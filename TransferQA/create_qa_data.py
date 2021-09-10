# Copyright (c) Facebook, Inc. and its affiliates
# All rights reserved.

import os
import json
import copy
from shutil import copyfile
from collections import defaultdict
import gzip
import random
from transformers import (AdamW, T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME)

DATA_DIR = "qa_data"
SAVE_PATH = "preprocessed"
if not os.path.exists(os.path.join(DATA_DIR,SAVE_PATH)):
    os.makedirs(os.path.join(DATA_DIR,SAVE_PATH))

def preprocess_mrqa():
    # dataset = [{"context":"text", "qas":{"question":"..", "answer":"..", "negative_questions":[]}, }]

    def _read_data(split = "mrqa_train"):
        datasets = []
        # read from original data
        for filename in os.listdir(os.path.join(DATA_DIR,split)):
            question_collection = []
            dataset = []
            with gzip.open(os.path.join(DATA_DIR,split,filename)) as f:
                for i, line in enumerate(f):
                    example = {"context":"", "qas":[]}
                    obj = json.loads(line)
                    # Skip headers.
                    if i == 0 and 'header' in obj:
                        continue
                    example["context"] = obj["context"].lower()
                    for qa in obj["qas"]:
                        qa_example = {"question":"", "negative_questions":[], "answer":"", "choice":[], "char_spans":[]}
                        answer_spans = []
                        for d_a in qa["detected_answers"]:
                            answer_spans+=d_a["char_spans"]
                        answer_spans.sort(key=lambda x:x[0])
                        qa_example["char_spans"] = answer_spans[0]

                        question = qa["question"].lower()
                        question_collection.append(question)
                        qa_example["question"] = question
                        qa_example["answer"] = qa["detected_answers"][0]["text"].lower()
                        example["qas"].append(qa_example)
                    dataset.append(example)
            print("done")
            print(len(dataset))
            print(dataset[5])
            # randomly sample 3 negative questions
            for i, example in enumerate(dataset):

                for qa in example["qas"]:
                    qa["negative_questions"] = random.sample(question_collection, 3)
                datasets.append(example)

        with open(os.path.join(DATA_DIR,f"{split}.json"), 'w') as fout:
            json.dump(datasets, fout, indent=4)
        return datasets
    data_train = _read_data("mrqa_train")
    data_dev = _read_data("mrqa_valid")



    return data_train, data_dev

def preprocess_dream():
    speaker_map = {"W:": "woman:", "M:": "man:", "F:": "woman:"}
    choice_token = " <extra_id_0> "
    def _read_data(split = "train.json"):
        question_collection = []
        dataset = []
        with open(os.path.join(DATA_DIR,"dream",split)) as f:
            data = json.load(f)
            for line in data:
                example = {"context":"", "qas":[]}
                context = " ".join(line[0])
                for k, v in speaker_map.items():
                    context = context.replace(k, v)
                example["context"] = context.lower()

                for qa in line[1]:
                    question = qa["question"].lower()
                    question_collection.append(question)
                    example["qas"].append({"question":question, "negative_questions":[], "answer":qa["answer"], "choice":qa["choice"], "char_spans":[]})

                dataset.append(example)
        # randomly sample 3 negative questions
        for i, example in enumerate(dataset):

            for qa in example["qas"]:
                qa["negative_questions"] = random.sample(question_collection, 3)
        print("done")
        print(len(dataset))
        with open(os.path.join(DATA_DIR,f"dream_{split}"), 'w') as fout:
            json.dump(dataset, fout, indent=4)
        return dataset
    data_train = _read_data("train.json")
    data_dev = _read_data("dev.json")
    data_test = _read_data("test.json")

    return data_train, data_dev+data_test

def preprocess_race():
    choice_token = " <extra_id_0> "
    choice_map = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8}
    def _read_data(split = "train"):
        question_collection = []
        dataset = []
        for hm in ["high", "middle"]:
            for filename in os.listdir(os.path.join(DATA_DIR,"RACE",split,hm)):
                with open(os.path.join(DATA_DIR,"RACE",split,hm,filename)) as f:
                    for line in f:
                        example = {"context":"", "qas":[]}
                        obj = json.loads(line)
                        example["context"] = obj["article"].lower()

                        for i, q in enumerate(obj["questions"]):
                            question = q
                            answer = obj["options"][i][ choice_map[obj["answers"][i]] ]
                            qa_example = {"question":question, "negative_questions":[], "answer":answer, "choice":obj["options"][i],"char_spans":[]}
                            question_collection.append(question)

                            example["qas"].append(qa_example)
                        dataset.append(example)
        # randomly sample 3 negative questions
        for i, example in enumerate(dataset):

            for qa in example["qas"]:
                qa["negative_questions"] = random.sample(question_collection, 3)
        print("done")
        print(len(dataset))
        with open(os.path.join(DATA_DIR,f"race_{split}.json"), 'w') as fout:
            json.dump(dataset, fout, indent=4)
        return dataset

    data_train = _read_data("train")
    data_dev = _read_data("dev")
    data_test = _read_data("test")


    return data_train, data_dev+data_test


def preprocess_squad2():
    # dataset = [{"context":"text", "qas":{"question":"..", "answer":"..", "negative_questions":[]}, }]

    def _read_data(split = "squad2/train-v2.0.json"):
        count=0
        dataset = []
        with open(os.path.join(DATA_DIR,split)) as f:
            data = json.load(f)
            for article in data["data"]:
                for obj in article["paragraphs"]:
                    example = {"context":"", "qas":[]}
                    example["context"] = obj["context"].lower()
                    for qa in obj["qas"]:
                        assert type(qa["is_impossible"]) is bool
                        if qa["is_impossible"]:
                            count+=1
                            qa_example = {"question":"", "negative_questions":[], "answer":"", "choice":[],"char_spans":[]}
                            question = qa["question"].lower()
                            qa_example["negative_questions"] = [question]*3
                            qa_example["question"] = question
                            qa_example["answer"] = "none"
                            example["qas"].append(qa_example)
                    if len(example["qas"])>0:
                        dataset.append(example)
        print("done")
        print(len(dataset))
        print(dataset[5])
        # randomly sample 3 negative questions
        # for i, example in enumerate(dataset):

        #     for qa in example["qas"]:
        #         qa["negative_questions"] = random.sample(question_collection, 3)
        #     datasets.append(example)

        # with open(os.path.join(DATA_DIR,f"{split}.json"), 'w') as fout:
        #     json.dump(dataset, fout, indent=4)
        print(count)
        return dataset
    data_train = _read_data("squad2/train-v2.0.json")
    data_dev = _read_data("squad2/dev-v2.0.json")



    return data_train+data_dev, None


if __name__=="__main__":

    # tokenizer = T5Tokenizer.from_pretrained("t5-small", bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
    # print(tokenizer.encode("<extra_id_0>"))
    train1, dev1 = preprocess_mrqa()
    train2, dev2 = preprocess_dream()
    train3, dev3 = preprocess_race()
    train4, _ = preprocess_squad2()


    with open(os.path.join(DATA_DIR,SAVE_PATH, "train.json"), 'w') as fout:
        json.dump(train1+train2+train3+train4, fout, indent=4)
    with open(os.path.join(DATA_DIR,SAVE_PATH, "dev.json"), 'w') as fout:
        json.dump(dev1+dev2+dev3, fout, indent=4)
