# Copyright (c) Facebook, Inc. and its affiliates
# All rights reserved.

import os, random
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import (AdamW, T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME)
from data_loader import prepare_data, prepare_QA_data, prepare_SGD_data, normalize_ontology, prepare_test_data
from config import get_args
from evaluate import evaluate_metrics
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from collections import Counter
import difflib


class DST_Seq2Seq(pl.LightningModule):

    def __init__(self,args, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.lr = args["lr"]


    def training_step(self, batch, batch_idx):
        self.model.train()

        (loss), *_ = self.model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            lm_labels=batch["decoder_output"]
                            )

        # result = pl.TrainResult(loss)
        # result.log('train_loss', loss, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss}}
        # return result

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        (loss), *_ = self.model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            lm_labels=batch["decoder_output"]
                            )


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
    args = vars(args)
    args["model_name"] = args["model_checkpoint"]+args["model_name"]+"_lr_" +str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"]) + "_neg_num_" + str(args["neg_num"])  + "_canonicalization" + str(args["canonicalization"]) + str(args["neg_context_ratio"])
    # train!
    seed_everything(args["seed"])

    model = T5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
    tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))


    task = DST_Seq2Seq(args, tokenizer, model)

    train_loader, val_loader, test_loader, ALL_SLOTS, domain_data = prepare_QA_data(args, task.tokenizer)

    #save model path
    save_path = os.path.join(args["saving_dir"],args["model_name"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trainer = Trainer(
                    default_root_dir=save_path,
                    accumulate_grad_batches=args["gradient_accumulation_steps"],
                    gradient_clip_val=args["max_norm"],
                    max_epochs=args["n_epochs"],
                    callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=8,verbose=False, mode='min')],
                    #gpus=args["GPU"],
                    deterministic=True,
                    gpus=args["GPU"],
                    num_nodes=1,
                    accelerator="ddp",
                    plugins='ddp_sharded'
                    # precision=16
                    )

    trainer.fit(task, train_loader, val_loader)

    task.model.save_pretrained(save_path)
    task.tokenizer.save_pretrained(save_path)

    print("test start...")
    #evaluate model
    #_ = evaluate_model(args, task.tokenizer, task.model, test_loader, save_path, ALL_SLOTS)

    for domain in domain_data:
        _ = evaluate_model(args, task.tokenizer, task.model, domain_data[domain]["data"], save_path, domain_data[domain]["slots"], prefix=domain)
    SGD_predict(args, tokenizer, model, save_path)

def evaluate_model(args, tokenizer, model, test_loader, save_path, ALL_SLOTS, prefix="zeroshot"):
    prefix += ("use_value_" + str(args["use_value"]))
    save_path = os.path.join(save_path,"results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    predictions = {}
    multi_choices_collection = []
    # active_slot_collection = {}
    # to gpu
    # gpu = args["GPU"][0]
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    if args["canonicalization"]:
        if args["version"]=="2.0":
            ontology = normalize_ontology(json.load(open("data/mwz2.0/ontology.json", 'r')))
        else:
            ontology = normalize_ontology(json.load(open("data/mwz2.1/ontology.json", 'r')))
            # with open("data/ontology.json") as f:
            #     ontology = json.load(f)
    slot_logger = {slot_name:[0,0,0] for slot_name in ALL_SLOTS}
    slot_logger["slot_gate"] = [0,0,0]

    for batch in tqdm(test_loader):
        dst_outputs = model.generate(input_ids=batch["encoder_input"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                max_length=200,
                                )

        value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)

        for idx, value in enumerate(value_batch):
            dial_id = batch["ID"][idx]
            if dial_id not in predictions:
                predictions[dial_id] = {}
                predictions[dial_id]["domain"] = batch["domains"][idx][0]
                predictions[dial_id]["turns"] = {}

            if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
                predictions[dial_id]["turns"][batch["turn_id"][idx]] = {"turn_belief":batch["turn_belief"][idx], "pred_belief":[]}

            # add the active slots into the collection
            if batch["question_type"][idx]=="extractive" and value!="none":

                if args["canonicalization"]:
                    value = difflib.get_close_matches(value, ontology[batch["slot_text"][idx]], n=1)
                    if len(value)>0:
                        predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(str(batch["slot_text"][idx])+'-'+str(value[0]))
                        value = value[0]
                    else:
                        value="none"
                else:
                    predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(str(batch["slot_text"][idx])+'-'+value)
            # analyze none acc:
            if batch["question_type"][idx]=="extractive":
                if value=="none" and batch["value_text"][idx]=="none":
                    slot_logger["slot_gate"][1]+=1 # hit
                if value!="none" and batch["value_text"][idx]!="none":
                    slot_logger["slot_gate"][1]+=1 # hit
                slot_logger["slot_gate"][0]+=1 # total

            # collect multi-choice answers
            if batch["question_type"][idx]=="multi-choice":
                if args["canonicalization"]:
                    value = difflib.get_close_matches(value, ontology[batch["slot_text"][idx]], n=1)
                    if len(value)>0 and value!="":
                        value = value[0]
                    else:
                        value="none"
                multi_choices_collection.append({"dial_id":batch["ID"][idx], "turn_id":batch["turn_id"][idx], "slot_text":batch["slot_text"][idx], "value":value})
            # ["day","type","area","pricerange",'internet',"parking"]
            # analyze slot acc:
            if (batch["value_text"][idx]!="none"):
                if str(value)==str(batch["value_text"][idx]):
                    slot_logger[str(batch["slot_text"][idx])][1]+=1 # hit
                slot_logger[str(batch["slot_text"][idx])][0]+=1 # total

    for example in multi_choices_collection:
        dial_id = example["dial_id"]
        turn_id = example["turn_id"]
        extractive_value = ""
        # check active slot
        for kv in predictions[dial_id]["turns"][turn_id]["pred_belief"]:
            if example["slot_text"] in kv:
                extractive_value = kv
        # if slot is not active
        if extractive_value=="":
            continue
        # replace extrative slot with multi-choice
        predictions[dial_id]["turns"][turn_id]["pred_belief"].remove(extractive_value)
        predictions[dial_id]["turns"][turn_id]["pred_belief"].append(str(example["slot_text"])+'-'+str(example["value"]))


    for slot_log in slot_logger.values():
        slot_log[2] = slot_log[1]/slot_log[0]

    with open(os.path.join(save_path, f"{prefix}_slot_acc.json"), 'w') as f:
        json.dump(slot_logger,f, indent=4)

    # with open(os.path.join(save_path, f"{prefix}_activation_collection.json"), 'w') as f:
    #     json.dump(active_slot_collection,f, indent=4)

    with open(os.path.join(save_path, f"{prefix}_prediction.json"), 'w') as f:
        json.dump(predictions,f, indent=4)

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ALL_SLOTS)

    evaluation_metrics = {"Joint Acc":joint_acc_score, "Turn Acc":turn_acc_score, "Joint F1":F1_score}
    print(f"{prefix} result:",evaluation_metrics)

    with open(os.path.join(save_path, f"{prefix}_result.json"), 'w') as f:
        json.dump(evaluation_metrics,f, indent=4)

    return predictions

def test(args, *more):
    args = vars(args)
    seed_everything(args["seed"])
    print(args)

    model = T5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
    tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")

    train_loader, val_loader, test_loader, ALL_SLOTS, domain_data = prepare_test_data(args, tokenizer)

    print("test start...")
    #evaluate model
    # _ = evaluate_model(args, tokenizer, model, test_loader, args["model_checkpoint"], ALL_SLOTS)

    for domain in domain_data:
        _ = evaluate_model(args, tokenizer, model, domain_data[domain]["data"], args["model_checkpoint"], domain_data[domain]["slots"], prefix=domain)

    SGD_predict(args, tokenizer, model, args["model_checkpoint"])


def SGD_predict(args, tokenizer, model, save_path, save_folder="sgd_prediction"):
    if not os.path.exists(os.path.join(save_path, save_folder)):
        os.makedirs(os.path.join(save_path, save_folder))

    test_loader, sgd_data = prepare_SGD_data(args, tokenizer)
    multi_choices_collection = []
    # to gpu
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    # delete all the gold slot values for testing
    for dial in sgd_data:
        for turn in dial["turns"]:
            if turn["speaker"] == "USER":
                for frame in turn["frames"]:
                    frame["state"]["slot_values"] = {}

    for batch in tqdm(test_loader):
        dst_outputs = model.generate(input_ids=batch["encoder_input"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                max_length=200,
                                )

        value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)
        for idx, value in enumerate(value_batch):
            dial_id = batch["ID"][idx]
            turn_id = batch["turn_id"][idx]
            frame_id = batch["frame_id"][idx]
            slot_key = batch["slot_text"][idx]
            # double check
            assert sgd_data[dial_id]["dialogue_id"]==batch["dialogue_id"][idx]
            if batch["question_type"][idx]=="extractive" and value!="none":
                sgd_data[dial_id]["turns"][turn_id]["frames"][frame_id]["state"]["slot_values"][slot_key] = [value]

            # collect multi-choice answers
            if batch["question_type"][idx]=="multi-choice":
                multi_choices_collection.append({"dial_id":dial_id, "turn_id":turn_id, "frame_id":frame_id, "slot_key":slot_key, "value":[value]})

    # update the extractive prediction with multi-choice prediction
    for example in multi_choices_collection:
        if example["slot_key"] in sgd_data[example["dial_id"]]["turns"][example["turn_id"]]["frames"][example["frame_id"]]["state"]["slot_values"]:
            sgd_data[example["dial_id"]]["turns"][example["turn_id"]]["frames"][example["frame_id"]]["state"]["slot_values"][example["slot_key"]] = example["value"]


    with open(os.path.join(save_path, save_folder,"output.json"), 'w') as fout:
        json.dump(sgd_data, fout, indent=4)




def fine_tune(args, *more):
    args = vars(args)
    seed_everything(args["seed"])
    # domains = ["hotel", "train", "restaurant", "attraction", "taxi", "none"]
    # for domain in domains:
    # args["only_domain"] = domain
    #assert args["only_domain"]!="none"
    # args["model_checkpoint"] = os.path.join(args["saving_dir"],args["model_name"])
    print(args)


    model = T5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
    tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")


    task = DST_Seq2Seq(args, tokenizer, model)
    train_loader, val_loader, test_loader, ALL_SLOTS, domain_data = prepare_data(args, tokenizer)

    trainer = Trainer(
                    default_root_dir=args["model_checkpoint"],
                    accumulate_grad_batches=args["gradient_accumulation_steps"],
                    gradient_clip_val=args["max_norm"],
                    max_epochs=args["n_epochs"],
                    callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=15,verbose=False, mode='min')],
                    #gpus=args["GPU"],
                    deterministic=True,
                    gpus=2,
                    num_nodes=1,
                    # precision=16,
                    accelerator="ddp",
                    plugins='ddp_sharded'
                    )

    trainer.fit(task, train_loader, val_loader)

    print("test start...")
    #evaluate model
    ratio = "ratio_" + str(args["fewshot"]) + "_seed_" + str(args["seed"]) + "_domain_" + args["only_domain"]
    _ = evaluate_model(args, task.tokenizer, task.model, test_loader, args["model_checkpoint"], ALL_SLOTS, prefix=ratio)



if __name__ == "__main__":
    args = get_args()
    if args.mode=="train":
        train(args)
    if args.mode=="finetune":
        fine_tune(args)
