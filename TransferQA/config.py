# Copyright (c) Facebook, Inc. and its affiliates
# All rights reserved.
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="t5-large", help="Path, url or short name of the model")
    parser.add_argument("--saving_dir", type=str, default="save", help="Path for saving")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--meta_batch_size", type=int, default=1, help="Batch size for meta training")
    parser.add_argument("--dev_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=4, help="Batch size for test")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=557, help="Random seed")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--length", type=int, default=50, help="Batch size for validation")
    parser.add_argument("--max_history", type=int, default=2, help="max number of turns in the dialogue")
    parser.add_argument("--GPU", type=int, default=8, help="how many gpu to use")
    parser.add_argument("--model_name", type=str, default="t5", help="use t5 or bart?")
    parser.add_argument("--slot_lang", type=str, default="question", help="slot description")
    parser.add_argument("--fewshot", type=float, default=0.0, help="data ratio for few shot experiment")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--fix_label", default = True)
    parser.add_argument("--except_domain", type=str, default="none", help="hotel, train, restaurant, attraction, taxi")
    parser.add_argument("--only_domain", type=str, default="none", help="hotel, train, restaurant, attraction, taxi")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--use_value", action='store_true')
    parser.add_argument("--canonicalization", default = True)
    parser.add_argument("--gold_slots", default = False)
    parser.add_argument("--version", type=str, default="2.1" , help="version of multiwoz")
    parser.add_argument("--neg_num", type=float, default=0.3, help="negative samples for qa training")
    parser.add_argument("--neg_context_ratio", type=float, default=0, help="negative context for qa training")
    args = parser.parse_args()
    # args.GPU = [int(gpu) for gpu in args.GPU]
    return args
