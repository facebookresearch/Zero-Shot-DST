# Copyright (c) Facebook, Inc. and its affiliates
# All rights reserved.

import json
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import ast
from tqdm import tqdm
import os
import random
from functools import partial
from utils.fix_label import fix_general_label_error
from collections import OrderedDict
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

random.seed(577)

class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]

        return item_info

    def __len__(self):
        return len(self.data)



def read_data(args, path_name, SLOTS, tokenizer, description, dataset=None):
    choice_token = " <extra_id_0> "
    print(("Reading all files from {}".format(path_name)))
    data = []

    domain_counter = {}
    # read files
    with open(path_name) as f:
        dials = json.load(f)

        if dataset=="train" and args["fewshot"]>0:
            random.Random(args["seed"]).shuffle(dials)
            dials = dials[:int(len(dials)*args["fewshot"])]

        for dial_dict in dials:
            dialog_history = ""

            # Counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "none" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "none" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
            (args["except_domain"] != "none" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]):
                continue

            # Reading data
            for ti, turn in enumerate(dial_dict["turns"]):
                turn_id = ti

                # accumulate dialogue utterances
                dialog_history +=  (" system: " + turn["system"] + " user: " + turn["user"])

                slot_values = fix_general_label_error(turn["state"]["slot_values"],SLOTS)

                # input: dialogue history + slot
                # output: value

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["except_domain"] not in k])
                    elif args["only_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["except_domain"] in k])
                    elif args["only_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])
                turn_belief_list = []
                for k,v in slot_values.items():
                    if v!="none":
                        turn_belief_list.append(str(k)+'-'+str(v))
                # turn_belief_list = [str(k)+'-'+str(v) for k,v in slot_values.items()]

                for slot in slot_temp:
                    # skip unrelevant slots for out of domain setting
                    if args["except_domain"] != "none" and dataset !="test":
                        if slot.split("-")[0] not in dial_dict["domains"]:
                            continue

                    slot_lang = description[slot]["question"]
                    slot_text = slot
                    value_text = slot_values.get(slot, 'none').strip()
                    if args["gold_slots"]:
                        if value_text=="none":
                            continue


                    input_text = f"extractive question: {slot_lang} context: {dialog_history}".lower()
                    output_text = value_text + f" {tokenizer.eos_token}"
                    data_detail = {
                        "ID":dial_dict["dial_id"],
                        "domains":dial_dict["domains"],
                        "turn_id":turn_id,
                        "dialog_history":dialog_history,
                        "turn_belief":turn_belief_list,
                        "intput_text":input_text,
                        "output_text":output_text,
                        "slot_text":slot_text,
                        "value_text":value_text,
                        "question_type": "extractive"
                        }
                    data.append(data_detail)

                    if len(description[slot]["values"])>0 and value_text!="none":
                        choices = (choice_token + choice_token.join(description[slot]["values"])).lower()
                        input_text = f"multi-choice question: {slot_lang} choices: {choices} context: {dialog_history}".lower()
                        output_text = (value_text + f" {tokenizer.eos_token}").lower()
                        data_detail = {
                            "ID":dial_dict["dial_id"],
                            "domains":dial_dict["domains"],
                            "turn_id":turn_id,
                            "dialog_history":dialog_history,
                            "turn_belief":turn_belief_list,
                            "intput_text":input_text,
                            "output_text":output_text,
                            "slot_text":slot_text,
                            "value_text":value_text,
                            "question_type": "multi-choice"
                            }
                        data.append(data_detail)



    for idx in range(10):
        print(data[idx])
    print("domain_counter", domain_counter)
    return data, slot_temp

def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]

    return SLOTS


def collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["intput_text"], padding=True, return_tensors="pt", add_special_tokens=False, verbose=False, truncation=True, max_length=1000)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']

    return batch_data


def normalize_ontology(ontology):
    keys = [k for k in ontology]
    for k in keys:
        for i in range(len(ontology[k])):
            ontology[k][i] = ontology[k][i].replace("do n't care", "dontcare")
            ontology[k][i] = ontology[k][i].replace("'s", " s")

        ontology[k.replace(" ","").lower() if ("book" not in k) else k.lower()] = ontology.pop(k)

    return ontology


def prepare_data(args, tokenizer):
    if args["version"]=="2.0":
        path_train = 'data/mwz2.0/train_dials.json'
        path_dev = 'data/mwz2.0/dev_dials.json'
        path_test = 'data/mwz2.0/test_dials.json'
        ontology = normalize_ontology(json.load(open("data/mwz2.0/ontology.json", 'r')))

    else:
        path_train = 'data/train_dials.json'
        path_dev = 'data/dev_dials.json'
        path_test = 'data/test_dials.json'
        ontology = normalize_ontology(json.load(open("data/mwz2.1/ontology.json", 'r')))
        #ontology = json.load(open("data/ontology.json", 'r'))

    ALL_SLOTS = get_slot_information(ontology)
    description = json.load(open("utils/slot_description.json", 'r'))

    data_train, _ = read_data(args, path_train, ALL_SLOTS, tokenizer, description, "train")
    data_dev, _ = read_data(args, path_dev, ALL_SLOTS, tokenizer, description, "dev")
    data_test, ALL_SLOTS = read_data(args, path_test, ALL_SLOTS, tokenizer, description, "test")

    train_dataset = DSTDataset(data_train, args)
    dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)

    train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    domain_data = {}
    return train_loader, dev_loader, test_loader, ALL_SLOTS, domain_data



def read_QA_data(args, path_name, tokenizer):
    choice_token = " <extra_id_0> "
    print(("Reading all files from {}".format(path_name)))
    data = []
    # read files
    with open(path_name) as f:
        examples = json.load(f)
        # examples = [{"context":"text", "qas":{"question":"..", "answer":"..", "negative_questions":[]}, }]

        for example in tqdm(examples):
            context = example["context"].strip()

            # save memory
            inputlen = len(tokenizer.encode(context))
            if inputlen>999:
                continue

            for qa in example["qas"]:
                question = qa["question"].strip()

                # input_text = f"extractive question: {question} context: {context}".lower()
                # output_text = (qa["answer"] + f" {tokenizer.eos_token}").lower()
                # data_detail = {
                #     "intput_text":input_text,
                #     "output_text":output_text,
                #     }
                # data.append(data_detail)

                # multi-choice question
                if len(qa["choice"])>0:
                    choices = (choice_token + choice_token.join(qa["choice"])).lower()
                    input_text = f"multi-choice question: {question} choices: {choices} context: {context}".lower()
                    output_text = (qa["answer"] + f" {tokenizer.eos_token}").lower()
                    data_detail = {
                        "intput_text":input_text,
                        "output_text":output_text,
                        }
                    data.append(data_detail)

                else:
                    input_text = f"extractive question: {question} context: {context}".lower()
                    output_text = (qa["answer"] + f" {tokenizer.eos_token}").lower()
                    data_detail = {
                        "intput_text":input_text,
                        "output_text":output_text,
                        }
                    data.append(data_detail)

                if random.random()<args["neg_num"]:
                # for i in range(args["neg_num"]):
                    negative_context = ""
                    if len(qa["char_spans"])>0:
                        for i in range(qa["char_spans"][0],0, -1):
                            if example["context"][i]==".":
                                negative_context = example["context"][:i+1]
                                # print(qa["char_spans"][0], i)
                                break

                    if (negative_context!="") and (random.random()<args["neg_context_ratio"]):
                        # use negative context
                        question = qa["question"].strip()
                        input_text = f"extractive question: {question} context: {negative_context}".lower()
                    else:
                        # use negative question
                        question = qa["negative_questions"][0].strip()
                        input_text = f"extractive question: {question} context: {context}".lower()


                        # print(input_text)
                        # print(qa["answer"])

                    output_text = "none" + f" {tokenizer.eos_token}"
                    data_detail = {
                    "intput_text":input_text,
                    "output_text":output_text,
                    }
                    data.append(data_detail)

    for idx in range(3):
        print(data[idx])
    return data



def prepare_QA_data(args, tokenizer):
    path_train = 'qa_data/preprocessed/train.json'
    path_dev = 'qa_data/preprocessed/dev.json'
    path_test = 'data/test_dials.json'

    ontology = json.load(open("data/ontology.json", 'r'))
    ALL_SLOTS = get_slot_information(ontology)
    description = json.load(open("utils/slot_description.json", 'r'))

    data_train = read_QA_data(args, path_train, tokenizer)
    data_dev = read_QA_data(args, path_dev, tokenizer)
    data_test, ALL_SLOTS = read_data(args, path_test, ALL_SLOTS, tokenizer, description, "test")
    domain_data = {}
    lock = False
    if args["only_domain"]=="none":
        lock = True
        for domain in EXPERIMENT_DOMAINS:
            args["only_domain"] = domain
            domain_test, domain_slots = read_data(args, path_test, ALL_SLOTS, tokenizer, description, "test")
            domain_data[domain] = {"data":domain_test, "slots":domain_slots}

    train_dataset = DSTDataset(data_train, args)
    dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)

    train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    if lock:
        for domain in EXPERIMENT_DOMAINS:
            domain_data[domain]["data"] = DataLoader(domain_data[domain]["data"], batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)

    return train_loader, dev_loader, test_loader, ALL_SLOTS, domain_data

def prepare_test_data(args, tokenizer):
    # path_train = 'qa_data/preprocessed/train.json'
    # path_dev = 'qa_data/preprocessed/dev.json'
    path_test = 'data/test_dials.json'

    ontology = json.load(open("data/ontology.json", 'r'))
    ALL_SLOTS = get_slot_information(ontology)
    description = json.load(open("utils/slot_description.json", 'r'))

    # data_train = read_QA_data(args, path_train, tokenizer)
    # data_dev = read_QA_data(args, path_dev, tokenizer)
    data_test, ALL_SLOTS = read_data(args, path_test, ALL_SLOTS, tokenizer, description, "test")
    domain_data = {}
    lock = False
    if args["only_domain"]=="none":
        lock = True
        for domain in EXPERIMENT_DOMAINS:
            args["only_domain"] = domain
            domain_test, domain_slots = read_data(args, path_test, ALL_SLOTS, tokenizer, description, "test")
            domain_data[domain] = {"data":domain_test, "slots":domain_slots}

    # train_dataset = DSTDataset(data_train, args)
    # dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)

    # train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    # dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    if lock:
        for domain in EXPERIMENT_DOMAINS:
            domain_data[domain]["data"] = DataLoader(domain_data[domain]["data"], batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)

    return None, None, test_loader, ALL_SLOTS, domain_data


def adjust_sgd_questions(schema):
    schema["Hotels_2"]["where_to"] = ("which city are user planning to stay in?", schema["Hotels_2"]["where_to"][1])
    schema["Hotels_2"]["has_laundry_service"] = ("whether the house has laundry service?", schema["Hotels_2"]["has_laundry_service"][1])
    schema["Hotels_4"]["location"] = ("what is the city or town where the hotel is located?", schema["Hotels_4"]["location"][1])
    schema["Hotels_4"]["star_rating"] = ("what is the star rating of the hotel?", schema["Hotels_4"]["star_rating"][1])
    schema["Hotels_4"]["place_name"] = ("what is the name of the hotel?", schema["Hotels_4"]["place_name"][1])
    schema["Media_3"]["genre"] = ("what type of the movie does user prefer?", schema["Media_3"]["genre"][1])
    schema["Media_3"]["starring"] = ("who is the actor in this movie?", schema["Media_3"]["starring"][1])
    schema["Services_4"]["city"] = ("what is the city or area where user wants to search for a therapist?", schema["Services_4"]["city"][1])
    schema["Music_3"]["artist"] = ("what is the name of the artist?", schema["Music_3"]["artist"][1])
    schema["Music_3"]["album"] = ("what is the album of the song?", schema["Music_3"]["album"][1])
    return schema

def fix_number(text):
    number_mapper = {"one": "1", "two": "2", "three":"3", "four":"4", "five":"5", "six":"6", "seven":"7", "eight":"8", "nine":"9", "ten":"10", "eleven":"11", "twelve":"12"}
    for fromx, tox in number_mapper.items():
        text = ' ' + text + ' '
        text = text.replace(f" {fromx} ", f" {tox} ")[1:-1]
    return text

# preprocess SGD
def read_SGD(args, path_name, tokenizer, dataset="test"):
    choice_token = " <extra_id_0> "

    # read test set
    all_data = []
    # read from original data
    for filename in os.listdir(os.path.join(path_name,dataset)):
        if filename.startswith("dialogues_"):
            with open(os.path.join(path_name,dataset,filename)) as f:
                data = json.load(f)
                all_data+=data

    with open(os.path.join(path_name,dataset,"schema.json")) as f:
        data = json.load(f)
        check_list = ["what", "how", "whether", "which"]
        schema = {}
        for service in data:
            schema[service["service_name"]] = {}
            # collect required_slots and optional_slots
            slot_collection = []
            for intent in service["intents"]:
                for slot in intent["required_slots"]:
                    slot_collection.append(slot)
                for slot in intent["optional_slots"].keys():
                    slot_collection.append(slot)

            for slot in service["slots"]:
                description = slot["description"].lower()
                if any(c_l in description for c_l in check_list):
                    description = f"{description}?"
                else:
                    description = f"what is the {description}?"

                if slot["name"] in slot_collection:
                    schema[service["service_name"]][slot["name"]] = (description, slot["possible_values"])

    schema = adjust_sgd_questions(schema)


    p_data = []
    # read dialogues
    for ID, dial in enumerate(all_data):
        #print(ID)
        dialog_history = ""

        for idx, turn in enumerate(dial["turns"]):
            utterance = turn["utterance"]
            utterance = fix_number(utterance)
            # User start the conversation
            if turn["speaker"] == "USER":
                assert idx%2==0
                # accumulate dialogue utterances
                #dialog_history +=  (" System: " + turn["system"] + " User: " + turn["user"])
                dialog_history +=  (" User: " + utterance)


                for fid, frame in enumerate(turn["frames"]):
                    # read slot values
                    for k in schema[frame["service"]]:
                        value_text = frame["state"]["slot_values"].get(k, ['none'])[0]

                    # for k, v in frame["state"]["slot_values"].items():

                        question = schema[frame["service"]][k][0]
                        input_text = f"extractive question: {question} context: {dialog_history}".strip().lower()
                        data_detail = {
                            "ID":ID,
                            "dialogue_id":dial["dialogue_id"],
                            "domains":frame["service"],
                            "turn_id":idx,
                            "frame_id":fid,
                            "intput_text":input_text,
                            "output_text":"dummy",
                            "slot_text":k,
                            "value_text":value_text,
                            "question_type": "extractive"
                            }
                        p_data.append(data_detail)



                        if len(schema[frame["service"]][k][1])>0 and value_text!="none":
                            choices = (choice_token + choice_token.join(schema[frame["service"]][k][1])).lower()
                            input_text = f"multi-choice question: {question} choices: {choices} context: {dialog_history}".strip().lower()
                            # output_text = (qa["answer"] + f" {tokenizer.eos_token}").lower()
                            data_detail = {
                                    "ID":ID,
                                    "dialogue_id":dial["dialogue_id"],
                                    "domains":frame["service"],
                                    "turn_id":idx,
                                    "frame_id":fid,
                                    "intput_text":input_text,
                                    "output_text":"dummy",
                                    "slot_text":k,
                                    "value_text":value_text,
                                    "question_type": "multi-choice"
                                    }
                            p_data.append(data_detail)


            # system turn
            else:
                assert idx%2==1
                dialog_history +=  (" Speaker: " + utterance)


    # with open(os.path.join("test",f"output.json"), 'w') as fout:
    #     json.dump(all_data, fout, indent=4)

    for idx in range(13):
        print(p_data[idx])
    # print(all_data[2])
    return p_data, all_data


def prepare_SGD_data(args, tokenizer):
    data_test, original_data = read_SGD(args=None, path_name="dstc8-schema-guided-dialogue" , tokenizer=tokenizer, dataset="test")
    test_dataset = DSTDataset(data_test, args)
    test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    return test_loader, original_data



if __name__ == "__main__":
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small", bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
    print(tokenizer.encode("true"))
    # prepare_QA_data
    # _ = read_SGD(args=None, path_name="dstc8-schema-guided-dialogue" , tokenizer=tokenizer, dataset="test")
