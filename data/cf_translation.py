import json

# ==================================      Global Vars      ==================================

CF_PATH = "data/counterfact.json"


# ===============================      Global functions      ===============================

def cf_rel_info():
    rels = dict()
    with open(CF_PATH, "r") as f:
        data = json.load(f)
    for sample in data:
        if sample['requested_rewrite']['relation_id'] not in rels:
            rels[sample['requested_rewrite']['relation_id']] = 1
        else:
            rels[sample['requested_rewrite']['relation_id']] += 1
    rels_lst = list(rels.items())
    rels_lst = sorted(rels_lst, key=lambda x: x[1], reverse=True)
    print(rels_lst)

def main():
    cf_rel_info()