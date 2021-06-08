# Copyright (c) Facebook, Inc. and its affiliates

import os, random
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import (AdamW, T5Tokenizer,  T5ForConditionalGeneration)
from data_loader import prepare_data
from config import get_args
from evaluate import evaluate_metrics
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from collections import Counter

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

def analysis(args):
    args = vars(args)
    # args["model_checkpoint"] = "trained/t5-smallt5_except_domain_train_slotlang_rule2_lr_0.0001_epoch_5_seed_555"
    model = T5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
    tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    train_loader, val_loader, test_loader, ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test = prepare_data(args, tokenizer)
    device = torch.device("cuda:0")

    # model.load_state_dict(torch.load("trained/t5-smallt5_except_domain_train_slotlang_none_lr_0.0001_epoch_5_seed_555/pytorch_model.bin"))
    model.to(device)
    model.eval()
    count = 0
    for batch in test_loader:
        decoder_input = torch.full((batch["encoder_input"].shape[0], 1), model.config.decoder_start_token_id, dtype=torch.long, device=device)

        # dst_outputs = model.generate(input_ids=batch["encoder_input"].to(device),
        #                         attention_mask=batch["attention_mask"].to(device),
        #                         eos_token_id=tokenizer.eos_token_id,
        #                         max_length=200,
        #                         )
        # if batch["value_text"][0]!="none":
        #     print(batch["intput_text"][0])
        #     value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)
        #     print(value_batch)
        outputs = model(input_ids=batch["encoder_input"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        decoder_input_ids=decoder_input,
                        return_dict=True,
                        output_attentions=True,
                        )
        if batch["value_text"][0]!="none":
            print(batch["intput_text"][0])
            tokens = tokenizer.convert_ids_to_tokens(batch["encoder_input"][0])
            max_id = torch.argmax(torch.sum(outputs.cross_attentions[1], 1).squeeze()).item()

            weights = torch.sum(outputs.cross_attentions[1], 1).squeeze().cpu().tolist()
            bukets = []
            for i in range(len(tokens)):
                bukets.append((tokens[i], weights[i]))
            # bukets.sort(key=lambda x: x[1])
            print(bukets[max(max_id-1,0)])
            print(bukets[max_id])
            print(bukets[max_id+1])
            count+=1
            if count>30:
                exit(0)


        # print(batch["encoder_input"].shape)
        # torch.sum(outputs.cross_attentions[0], 1).squeeze().cpu().tolist()
        #print(torch.sum(outputs.cross_attentions[0], 1).squeeze().cpu().tolist())

        #print(torch.sum(outputs.cross_attentions[1], 1).squeeze())






if __name__ == "__main__":
    args = get_args()
    analysis(args)

# python analysis.py --test_batch_size 1 --model_checkpoint trained/t5-smallt5_except_domain_train_slotlang_rule2_lr_0.0001_epoch_5_seed_555 --except_domain train --slot_lang rule2
