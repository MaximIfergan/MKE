import json
import pandas as pd
import requests
import io
import lodstorage  # pip install pyLodStorage
from lodstorage.sparql import SPARQL
from lodstorage.csv import CSV
from tqdm.contrib import itertools
from wikidata.client import Client
import sys
import random
from util import load_json_file
random.seed(18)

# ===============================      Global Variables:      ===============================

LANGS = ["en", "fr", "ru", "he", "ar", "es", "it"]

RELS = {"birth_city": "P19", "birth_year": "P569", "death_year": "P570", "geo_continent": "P30",
        "architect_by": "P84", "sport_type": "P641"}


# "tourist_attraction_country": "P17",
def from_csv_to_template_dict(path):
    df = pd.read_csv(path, index_col="lang/relation")
    result = dict()
    for relation in df.columns[:-1]:
        result[relation] = {lang: {"m": [], "f": []} for lang in LANGS}
        for lang in LANGS:
            for gender in ["m", "f"]:
                for i in range(3):
                    result[relation][lang][gender].append(df.loc[f"{lang}_{gender}_{i + 1}"][relation])
    return result


PROMPT_TEMPLATES = from_csv_to_template_dict("Dataset/dataset_templates.csv")

CLIENT = Client()

LANG2QID = {"en": "Q1860", "fr": "Q150", "he": "Q9288", "ar": "Q13955", "ru": "Q7737", "es": "Q1321", "it": "Q652"}
QID2LAND = {"Q1860": "en", "Q150": "fr", "Q9288": "he", "Q13955": "ar", "Q7737": "ru", "Q1321": "es", "Q652": "it"}

RAW_DATA_PATH = "Dataset/mke_data.json"

FEW_SHOT = load_json_file("Dataset/fewshots.json")

ENTITIES2LABELS_PATH = "Dataset/ENTITIES2LABELS.json"

with open(ENTITIES2LABELS_PATH, 'r') as file:
    ENTITIES2LABELS = json.load(file)


# ===============================      Global functions:      ===============================

def convert_dict_to_csv():
    lists = [LANGS, ["m", "f"], [str(e) for e in list(range(1, 4))]]
    column_names = ["_".join(element) for element in itertools.product(*lists)]
    df = pd.DataFrame([], columns=["relation"] + column_names)
    for i, rel in enumerate(PROMPT_TEMPLATES.keys()):
        df.loc[i] = [rel] + (['nan'] * len(column_names))
        for lang in PROMPT_TEMPLATES[rel].keys():
            for j in range(3):
                df.loc[i, f"{lang}_m_{j + 1}"] = PROMPT_TEMPLATES[rel][lang]["M"][j].strip()
                df.loc[i, f"{lang}_f_{j + 1}"] = PROMPT_TEMPLATES[rel][lang]["F"][j].strip()
    df.to_csv("try.csv", index_col=False)


def get_entity_name(entity_id, lang):
    """
    :return: The entity name as a string for a given entity id and language code
    """
    if entity_id in ENTITIES2LABELS and lang in ENTITIES2LABELS[entity_id]:
        return ENTITIES2LABELS[entity_id][lang]
    try:
        entity = CLIENT.get(entity_id, load=True)
    except Exception as e:
        sys.stderr.write("\n Error:" + str(e) + "\n")
        return -1
    if lang in entity.data['labels']:
        if entity_id not in ENTITIES2LABELS:
            ENTITIES2LABELS[entity_id] = dict()
        ENTITIES2LABELS[entity_id][lang] = entity.data['labels'][lang]['value']
        return ENTITIES2LABELS[entity_id][lang]
    else:
        return -1


def merge_json_files(input_files, output_file):
    """Combines multiple JSON files, each containing a list of dictionaries on a single line,
    into a single output file where each line is a separate dictionary."""

    with open(output_file, "w") as outfile:
        for input_file in input_files:
            with open(input_file, "r") as infile:
                try:
                    file_data = json.loads(infile.readline())  # Load the list of dictionaries from the line
                    for dictionary in file_data:
                        json.dump(dictionary, outfile)  # Write each dictionary as a separate line
                        outfile.write("\n")
                except json.JSONDecodeError as e:
                    print(f"Error reading JSON file {input_file}: {e}")


def query_online(sparqlQuery, return_type='json'):
    sparql = SPARQL("https://query.wikidata.org/sparql")
    res = sparql.queryAsListOfDicts(sparqlQuery)
    if return_type == 'csv':
        csv = CSV.toCSV(res)
        res = pd.read_csv(io.StringIO(csv))
    return res


def url_to_q_entity(url):
    """Extracts the Wikidata entity ID (Q-number) from a Wikidata URL.

      Args:
        url: A string representing a Wikidata URL, e.g., "http://www.wikidata.org/entity/Q214475".

      Returns:
        A string containing the Wikidata entity ID (Q-number), or None if the URL is invalid.
      """
    return url.split("/")[-1]


def save_entities_labels():
    with open(ENTITIES2LABELS_PATH, 'w') as file:
        json.dump(ENTITIES2LABELS, file, ensure_ascii=False)


def load_query(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return ''.join(lines)


def query_to_json(path, query_res):
    with open(path, 'w') as file:
        for dictionary in query_res:
            json.dump(dictionary, file)
            file.write('\n')


def collect_data():
    # {'people': {'en': 2000, 'fr': 1423, 'he': 803}, 'sport': {'en': 2000, 'fr': 1911, 'he': 159}}
    raw_data = []
    limit = "2000"
    queries = ["people", "sport"]
    info = {q: dict.fromkeys(LANGS) for q in queries}
    for query in ["people", "sport"]:
        raw_query = load_query(f"Dataset/Queries/{query}.txt")
        for lang in LANGS:
            format_query = raw_query.replace("{lang}", LANG2QID[lang]).replace("{limit}", limit)
            result = query_online(format_query)
            for sample in result:
                sample["lang_code"] = lang
                sample["query"] = query
            info[query][lang] = len(result)
            assert result
            raw_data += result
            print(info)
    return raw_data


def preprocess_raw_collected_data(parallel_level=7, max_per_rel=10000, out_path="Dataset/mke_data.json"):
    # === Raw Data Details ===
    # = Hyper-Parameters:
    # parallel_level: 7, max_per_rel: 10000 = Relation sizes: {
    # 'birth_city': {'before': 100000, 'after': 5392},
    # 'birth_year': {'before': 50000, 'after': 3509},
    # 'death_year': {'before': 50000, 'after': 2207},
    # 'geo_continent': {'before': 20000, 'after': 262},
    # 'tourist_attraction_country': {'before': 4556, 'after': 304},
    # 'architect_by': {'before': 20000, 'after': 828},
    # 'sport_type': {'before': 20000, 'after': 829}}
    # = Size: Before: 264556, Final 13331

    results = []
    filter_col = [f"s_{lang}" for lang in LANGS]
    rel_sizes = {rel: dict() for rel in RELS.keys()}
    for rel in RELS.keys():
        df = pd.read_csv(f"Dataset/QueriesData/{rel}.csv")
        rel_sizes[rel]["before"] = len(df)
        df['empty_s'] = df.apply(lambda row: sum(row[column] == ''
                                                 or pd.isnull(row[column]) for column in filter_col), axis=1)
        df = df[len(LANGS) - parallel_level >= df['empty_s']][:max_per_rel]
        rel_sizes[rel]["after"] = len(df)
        rel_data = df.to_dict('records')  # To python dictionary
        for sample in rel_data:
            sample.pop("empty_s")
            sample["rel"] = rel
        results += rel_data
    total_before = sum([rel_sizes[rel]["before"] for rel in RELS.keys()])
    total_after = sum([rel_sizes[rel]["after"] for rel in RELS.keys()])
    print("=== Raw Data Details ===")
    print(f"= Hyper-Parameters:")
    print(f"parallel_level: {parallel_level}, max_per_rel: {max_per_rel}")
    print(f"= Relation sizes:")
    print(rel_sizes)
    print(f"= Size:")
    print(f"Before: {total_before}, Final {total_after}")

    # == Save the raw data ===
    with open(out_path, 'w', encoding='utf8') as file:
        for dictionary in results:
            json.dump(dictionary, file, ensure_ascii=False)
            file.write('\n')


# ====================================      Class:      ====================================


class DatasetBuilder:

    def __init__(self, raw_data_path=RAW_DATA_PATH):
        self.target_labels_qid = None
        self.data = []
        self.raw_data = load_json_file(raw_data_path)
        # random.shuffle(self.raw_data)  # TODO: For debug
        # self.raw_data = self.raw_data[:200]  # TODO: For debug
        self.id_count = 1
        self.target_labels = {rel: dict() for rel in RELS.keys() if "year" not in rel}

    def preprocess(self):
        for sample in self.raw_data:
            self.construct_prompts(sample)
        self.target_labels_qid = {rel: list(self.target_labels[rel].keys()) for rel in self.target_labels}
        self.assign_target_labels()

    def construct_prompts(self, sample):
        if sample["rel"] in ["tourist_attraction_country"]:
            return

        if "gender" in sample.keys():
            gender = 'm' if sample["gender"].endswith("Q6581097") else "f"
        else:
            gender = 'm'
        subject_langs = [key.split("_")[1] for key in sample.keys() if
                         key.startswith("s_") and str(sample[key]) != 'nan']
        if "year" in sample["rel"]:
            object_langs = subject_langs
        else:
            object_langs = [key.split("_")[1] for key in sample.keys() if
                            key.startswith("o_") and str(sample[key]) != 'nan']
        sample_lang = list(set(subject_langs) & set(object_langs))
        if "year" in sample["rel"] and str(sample["o"]) == 'nan':
            return
        obj_true = {"qid": None if "year" in sample["rel"] else url_to_q_entity(sample["o"]),
                    "label": {lang: str(int(sample["o"])) for lang in object_langs} if "year" in sample["rel"]
                    else {lang: sample[f"o_{lang}"] for lang in sample_lang}}
        new_sample = {"id": self.id_count,
                      "subj": {"label": {lang: sample[f"s_{lang}"] for lang in sample_lang},
                               "qid": url_to_q_entity(sample["s"]),
                               "origin": None if "origin" not in sample else QID2LAND[
                                   url_to_q_entity(sample["origin"])],
                               "gender": gender},
                      "rel": {"label": sample["rel"],
                              "qid": RELS[sample["rel"]]},
                      "obj_true": obj_true,
                      "prompt": {lang: PROMPT_TEMPLATES[sample["rel"]][lang][gender][0].format(sample[f"s_{lang}"])
                                 for lang in sample_lang},
                      "paraphrase_prompts": {lang: [prompt.format(sample[f"s_{lang}"])
                                                    for prompt in PROMPT_TEMPLATES[sample["rel"]][lang][gender][1:]]
                                             for lang in sample_lang}}

        if "year" not in sample["rel"] and len(sample_lang) == len(LANGS):
            self.target_labels[sample["rel"]][obj_true["qid"]] = obj_true
        self.data.append(new_sample)
        self.id_count += 1

    def assign_target_labels(self):

        for i in range(len(self.data)):
            diff = random.randint(1, 30)
            sign = random.randint(0, 1)
            final = diff if sign else -diff
            if 'year' in self.data[i]["rel"]["label"]:
                self.data[i]["target_true"] = {
                    "label": {lang: str(int(float(self.data[i]["obj_true"]["label"]['en']) + final))
                              for lang in LANGS}}
            else:
                two_op = random.choices(self.target_labels_qid[self.data[i]["rel"]['label']], k=2)
                if two_op[0] != self.data[i]["obj_true"]['qid']:
                    self.data[i]["target_true"] = {"qid": two_op[0],
                                                   "label": self.target_labels[self.data[i]["rel"]['label']][two_op[0]]}
                else:
                    self.data[i]["target_true"] = {"qid": two_op[1],
                                                   "label": self.target_labels[self.data[i]["rel"]['label']][two_op[1]]}

    def save_fewshot_examples(self, out_path="Dataset/fewshots.json"):
        fewshots = {rel: {lang: {"prompt": "", "p1": "", "p2": ""} for lang in LANGS} for rel in RELS.keys()}
        rels_1 = []
        rels_2 = []
        for sample in self.data:
            if len(list(sample['subj']['label'].keys())) != len(LANGS):
                continue
            if sample["rel"]["label"] not in rels_1:
                rels_1.append(sample["rel"]["label"])
            elif sample["rel"]["label"] not in rels_2:
                rels_2.append(sample["rel"]["label"])
            else:
                continue
            for lang in sample['subj']['label']:
                fewshots[sample["rel"]["label"]][lang]["prompt"] += sample["prompt"][lang] + " " + \
                                                                    sample["obj_true"]['label'][lang] + ". "
                fewshots[sample["rel"]["label"]][lang]["p1"] += sample["paraphrase_prompts"][lang][0] + " " + \
                                                                sample["obj_true"]['label'][lang] + ". "
                fewshots[sample["rel"]["label"]][lang]["p2"] += sample["paraphrase_prompts"][lang][1] + " " + \
                                                                sample["obj_true"]['label'][lang] + ". "

            if len(rels_2) == len(list(RELS.keys())):
                break
        self.fewshots = fewshots
        with open(out_path, 'w', encoding='utf8') as file:
            json.dump(fewshots, file)

    def save(self, path):
        with open(path, 'w', encoding='utf8') as file:
            for dictionary in self.data:
                json.dump(dictionary, file, ensure_ascii=False)
                file.write('\n')


def main():
    db = DatasetBuilder()
    db.preprocess()
    db.save("Dataset/mke_data.json")
    save_entities_labels()
    # convert_dict_to_csv()
    # raw_data = collect_data()
    # with open('raw_data_fr-en.json', 'w') as file:
    #     for dictionary in raw_data:
    #         json.dump(dictionary, file)
    #         file.write('\n')
    # exit(0)
