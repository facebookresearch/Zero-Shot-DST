# Copyright (c) Facebook, Inc. and its affiliates

import json

# EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


with open("slot_description.json", 'r') as f:
    ontology = json.load(f)


slot_map = {"pricerange": "price range", "arriveby": "arrive by", "leaveat": "leave at"}
slot_types = {str(["book stay", "book people", "stars"]):"number of ", str(["parking", "internet"]):"whether have ", str(["destination", "departure"]):"location of ", str(["arriveby", "leaveat"]):"time of "}


# Naive descriptions
for domain_slot in ontology:
    domain, slot = domain_slot.split("-")
    if slot in slot_map:
        slot = slot_map[slot]
    if "book" in domain_slot:
        slot = slot.replace("book ", "")
        ontology[domain_slot]["naive"] = f"{slot} for the {domain} booking"
    else:
        ontology[domain_slot]["naive"] = f"{slot} of the {domain}"


# question
for domain_slot in ontology:
    domain, slot = domain_slot.split("-")
    ontology[domain_slot]["question"] = f"What is the {slot} of the {domain} that the user in interested in?"


# Slot Type
for domain_slot in ontology:
    domain, slot = domain_slot.split("-")
    slot_name = slot
    if slot in slot_map:
        slot_name = slot_map[slot]
    prefix = ""
    for slot_list, slot_type in slot_types.items():
        if slot in slot_list:
            prefix = slot_type

    if "book" in domain_slot:
        slot_name = slot_name.replace("book ", "")
        ontology[domain_slot]["slottype"] = f"{prefix}{slot_name} for the {domain} booking"
    elif prefix=="whether have ":
        ontology[domain_slot]["slottype"] = f"{prefix}{slot_name} in the {domain}"
    else:
        ontology[domain_slot]["slottype"] = f"{prefix}{slot_name} of the {domain}"


with open('slot_description.json', 'w') as f:
    json.dump(ontology, f, indent=4)
