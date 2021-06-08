# Copyright (c) Facebook, Inc. and its affiliates

import os, random
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import (AdamW, GPT2Tokenizer, GPT2LMHeadModel)
from data_loader import prepare_data
from config import get_args
from evaluate import evaluate_metrics
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np


def greedy_decode(input_text, tokenizer, model, args, max_length, current_output=None):
    if current_output is None:
        current_output = []

    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    with torch.no_grad():
        for i in range(max_length):
            input_tensor = torch.tensor(input_ids+current_output, device=torch.device("cuda:0")).unsqueeze(0)
            logits = model(input_tensor)
            if isinstance(logits, tuple):  # for gpt2 and maybe others
                logits = logits[0]
            predicted_index = torch.argmax(logits[0, -1, :]).item()

            if predicted_index==tokenizer.eos_token_id:
                break
            current_output.append(predicted_index)

    output_text = tokenizer.decode(current_output)
    return output_text




class DST_GPT(pl.LightningModule):

    def __init__(self,args, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.lr = args["lr"]
        self.args = args

    def training_step(self, batch, batch_idx):
        self.model.train()
        # follow https://github.com/salesforce/simpletod/blob/917f66afe7f37e75de246949423fc4470a2427c4/main.py#L148
        (loss), *_ = self.model(input_ids=batch["input_ids"], labels=batch["input_ids"])

        return {'loss': loss, 'log': {'train_loss': loss}}


    def validation_step(self, batch, batch_idx):
        self.model.eval()
        (loss), *_ = self.model(input_ids=batch["input_ids"], labels=batch["input_ids"])

        return {'val_loss': loss, 'log': {'val_loss': loss}}
        # return result

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_loss in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        return results


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)



def train(args, *more):
    # # train!
    args = vars(args)
    args["model_name"] = args["model_checkpoint"]+args["model_name"]+"_except_domain_"+args["except_domain"]+ "_slotlang_" +str(args["slot_lang"]) + "_lr_" +str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"])
    # train!
    seed_everything(args["seed"])

    model = GPT2LMHeadModel.from_pretrained(args["model_checkpoint"])
    tokenizer = GPT2Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]", pad_token = "[pad]")
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    task = DST_GPT(args, tokenizer, model)
    train_loader, val_loader, test_loader, ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test = prepare_data(args, task.tokenizer)

    #save model
    save_path = os.path.join(args["saving_dir"],args["model_name"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trainer = Trainer(
                    default_root_dir=save_path,
                    accumulate_grad_batches=args["gradient_accumulation_steps"],
                    gradient_clip_val=args["max_norm"],
                    max_epochs=args["n_epochs"],
                    callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=5,verbose=False, mode='min')],
                    gpus=args["GPU"],
                    deterministic=True,
                    num_nodes=1,
                    accelerator="ddp"
                    )

    trainer.fit(task, train_loader, val_loader)


    task.model.save_pretrained(save_path)
    task.tokenizer.save_pretrained(save_path)

    print("test start...")
    #evaluate model
    _ = evaluate_model(args, task.tokenizer, task.model, test_loader, save_path, ALL_SLOTS)



def evaluate_model(args, tokenizer, model, test_loader, save_path, ALL_SLOTS, prefix="zeroshot"):
    save_path = os.path.join(save_path,"results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    predictions = {}
    # to gpu
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    slot_logger = {slot_name:[0,0,0] for slot_name in ALL_SLOTS}

    for batch in tqdm(test_loader):
        for idx, input_text in enumerate(batch["intput_text"]):
            dst_text = greedy_decode(input_text, tokenizer, model, args, max_length=200)
            value = dst_text.strip()
            dial_id = batch["ID"][idx]

            if dial_id not in predictions:
                predictions[dial_id] = {}
                predictions[dial_id]["domain"] = batch["domains"][idx][0]
                predictions[dial_id]["turns"] = {}
            if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
                predictions[dial_id]["turns"][batch["turn_id"][idx]] = {"turn_belief":batch["turn_belief"][idx], "pred_belief":[]}

            if value!="none":
                predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(str(batch["slot_text"][idx])+'-'+str(value))

            # dst_text = greedy_decode(input_text, tokenizer, model, args, max_length=200)
            # slot_values = dst_text.strip()
            # dial_id = batch["ID"][idx]

            # if dial_id not in predictions:
            #     predictions[dial_id] = {}
            #     predictions[dial_id]["domain"] = batch["domains"][idx][0]
            #     predictions[dial_id]["turns"] = {}
            # if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
            #     predictions[dial_id]["turns"][batch["turn_id"][idx]] = {"turn_belief":batch["turn_belief"][idx], "pred_belief":[]}
            # # print(slot_values)
            # # print(slot_values.split(", "))
            # for slot_value in slot_values.split(", "):
            #     value = slot_value.split("-")[-1]
            #     # print(value)
            #     if value!="none":
            #         predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(slot_value)


    with open(os.path.join(save_path, f"{prefix}_prediction.json"), 'w') as f:
        json.dump(predictions,f, indent=4)

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ALL_SLOTS)

    evaluation_metrics = {"Joint Acc":joint_acc_score, "Turn Acc":turn_acc_score, "Joint F1":F1_score}
    print(f"{prefix} result:",evaluation_metrics)

    with open(os.path.join(save_path, f"{prefix}_result.json"), 'w') as f:
        json.dump(evaluation_metrics,f, indent=4)

    return predictions



if __name__ == "__main__":
    args = get_args()
    train(args)


    # evaluate()
