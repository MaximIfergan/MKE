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

# LANGS = ["en", "fr", "ru", "he", "ar", "es", "it"]
LANGS = ["en", "fr"]  # TODO: delete only for DEBUG

def from_csv_to_template_dict(path):
    df = pd.read_csv(path, index_col="relation")
    result = dict()
    for relation in df.index:
        result[relation] = {lang: {"m": [], "f": []} for lang in LANGS}
        for lang in LANGS:
            for gender in ["m", "f"]:
                for i in range(3):
                    result[relation][lang][gender].append(df.loc[relation][f"{lang}_{gender}_{i + 1}"])
    return result

PROMPT_TEMPLATES = from_csv_to_template_dict("Dataset/dataset_templates.csv")

CLIENT = Client()

LANG2QID = {"en": "Q1860", "fr": "Q150", "he": "Q9288", "ar": "Q13955", "ru": "Q7737"}

RAW_DATA_PATH = "Dataset/raw_data.json"

FEW_SHOT = {
    "birth_year": {
        "fr": "Abraham Lincoln est née en l'an 1809. Cristiano Ronaldo est née en l'an 1985. ",
        "en": "Abraham Lincoln was born in the year 1809. Cristiano Ronaldo was born in the year 1985. "},
    "death_year": {
        "fr": "Abraham Lincoln est mort en l'an 1865. Albert Einstein est mort en l'an 1955. ",
        "en": "Abraham Lincoln died in the year 1865. Albert Einstein died in the year 1955. "},
    "birth_city": {
        "fr": "Albert Einstein was born in the city of Ulm. Cristiano Ronaldo was born in the city of Funchal. ",
        "en": "Albert Einstein est né dans une ville nommée Ulm. Cristiano Ronaldo est né dans une ville Funchal. "},
    "sport": {
        "fr": "Rafael Nadal joue professionnellement au sport du tennis. Cristiano Ronaldo joue professionnellement au sport du football. ",
        "en": "Rafael Nadal professionally plays the sport of tennis. Cristiano Ronaldo professionally plays the sport of association football. "}
}

ENTITIES2LABELS_PATH = "Dataset/ENTITIES2LABELS.json"

with open(ENTITIES2LABELS_PATH, 'r') as file:
    ENTITIES2LABELS = json.load(file)

PROMPT_TEMPLATES = from_csv_to_template_dict()

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


def query_online(sparqlQuery):
    sparql = SPARQL("https://query.wikidata.org/sparql")
    res = sparql.queryAsListOfDicts(sparqlQuery)
    # === CSV Format: ===
    # csv = CSV.toCSV(qlod)
    # pd.read_csv(io.StringIO(csv))
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


# ====================================      Class:      ====================================


class DatasetBuilder:

    def __init__(self, raw_data_path=RAW_DATA_PATH):
        self.data = []
        self.raw_data = load_json_file(raw_data_path)
        random.shuffle(self.raw_data)  # TODO: For debug
        self.raw_data = self.raw_data[:10]  # TODO: For debug
        self.id_count = 1
        self.cities = {}
        self.sports = {}

    def preprocess(self):
        self.construct_prompts()
        self.sports_qid = list(self.sports.keys())
        self.cities_qid = list(self.cities.keys())
        self.assign_target_labels()
        self.assign_loc_prompt()

    def construct_prompts(self):
        for sample in self.raw_data:
            query_type = sample['query']
            if query_type == 'people':
                self.preprocess_people_sample(sample)
            elif query_type == 'sport':
                self.append_sport_sample(sample)

    def preprocess_people_sample(self, sample):
        self.append_birth_year_sample(sample)
        self.append_death_year_sample(sample)
        self.append_birth_city_sample(sample)

    def append_sport_sample(self, entity):
        gender = 'M' if entity["gender"].endswith("Q6581097") else "F"
        sport_qid = url_to_q_entity(entity["sportType"])
        obj_true = {lang: get_entity_name(sport_qid, lang) for lang in LANGS}
        sample = {"id": self.id_count,
                  "subj": {"label": {lang: entity[f"o_{lang}"] for lang in obj_true.keys()},
                           "qid": url_to_q_entity(entity["entitiy"]),
                           "origin": entity["lang_code"],
                           "gender": gender},
                  "rel": {"label": "sport",
                          "qid": "P641"},
                  "obj_true": {"qid": sport_qid,
                               "label": obj_true},
                  "prompt": {lang: PROMPT_TEMPLATES["sport"][lang][gender][0].format(entity[f"o_{lang}"])
                             for lang in obj_true.keys()},
                  "paraphrase_prompts": {lang: [prompt.format(entity[f"o_{lang}"])
                                                for prompt in PROMPT_TEMPLATES["sport"][lang][gender][1:]]
                                         for lang in obj_true.keys()}}
        self.sports[sport_qid] = obj_true
        self.data.append(sample)
        self.id_count += 1

    def append_birth_year_sample(self, entity):
        if "birthYear" not in entity:
            return -1
        gender = 'M' if entity["gender"].endswith("Q6581097") else "F"
        sample = {"id": self.id_count,
                  "subj": {"label": {lang: entity[f"s_{lang}"] for lang in LANGS},
                           "qid": url_to_q_entity(entity["entitiy"]),
                           "origin": entity["lang_code"],
                           "gender": gender},
                  "rel": {"label": "birth_year",
                          "qid": "P569"},
                  "obj_true": {"label": {lang: str(entity["birthYear"]) for lang in LANGS}},
                  "prompt": {lang: PROMPT_TEMPLATES["birth_year"][lang][gender][0].format(entity[f"s_{lang}"])
                             for lang in LANGS},
                  "paraphrase_prompts": {lang: [prompt.format(entity[f"s_{lang}"])
                                                for prompt in PROMPT_TEMPLATES["birth_year"][lang][gender][1:]]
                                         for lang in LANGS}}
        self.data.append(sample)
        self.id_count += 1

    def append_death_year_sample(self, entity):
        if "deathYear" not in entity:
            return -1
        gender = 'M' if entity["gender"].endswith("Q6581097") else "F"
        sample = {"id": self.id_count,
                  "subj": {"label": {lang: entity[f"s_{lang}"] for lang in LANGS},
                           "qid": url_to_q_entity(entity["entitiy"]),
                           "origin": entity["lang_code"],
                           "gender": gender},
                  "rel": {"label": "death_year",
                          "qid": "P570"},
                  "obj_true": {"label": {lang: str(entity["deathYear"]) for lang in LANGS}},
                  "prompt": {lang: PROMPT_TEMPLATES["death_year"][lang][gender][0].format(entity[f"s_{lang}"])
                             for lang in LANGS},
                  "paraphrase_prompts": {lang: [prompt.format(entity[f"s_{lang}"])
                                                for prompt in PROMPT_TEMPLATES["death_year"][lang][gender][1:]]
                                         for lang in LANGS}}
        self.data.append(sample)
        self.id_count += 1

    def append_birth_city_sample(self, entity):
        if "cityOfBirth" not in entity:
            return -1
        city_qid = url_to_q_entity(entity["cityOfBirth"])
        obj_true = {lang: get_entity_name(city_qid, lang) for lang in LANGS}
        gender = 'M' if entity["gender"].endswith("Q6581097") else "F"
        sample = {"id": self.id_count,
                  "subj": {"label": {lang: entity[f"s_{lang}"] for lang in obj_true.keys()},
                           "qid": url_to_q_entity(entity["entitiy"]),
                           "origin": entity["lang_code"],
                           "gender": gender},
                  "rel": {"label": "birth_city", "qid": "P19"},
                  "obj_true": {"label": obj_true, "qid": city_qid},
                  "prompt": {lang: PROMPT_TEMPLATES["birth_city"][lang][gender][0].format(entity[f"s_{lang}"])
                             for lang in obj_true.keys()},
                  "paraphrase_prompts": {lang: [prompt.format(entity[f"s_{lang}"])
                                                for prompt in PROMPT_TEMPLATES["birth_city"][lang][gender][1:]]
                                         for lang in obj_true.keys()}}
        self.cities[city_qid] = obj_true
        self.data.append(sample)
        self.id_count += 1

    def assign_target_labels(self):

        for i in range(len(self.data)):
            diff = random.randint(1, 30)
            sign = random.randint(0, 1)
            final = diff if sign else -diff
            if self.data[i]["rel"]["label"] == "birth_year":
                self.data[i]["target_true"] = {"label": {lang: str(int(self.data[i]["obj_true"]["label"] + final))
                                                         for lang in LANGS}}
            elif self.data[i]["rel"]["label"] == "death_year":
                self.data[i]["target_true"] = {"label": {lang: str(int(self.data[i]["obj_true"]["label"]) + final)
                                                         for lang in LANGS}}
            elif self.data[i]["rel"]["label"] == "birth_city":
                r_cities = random.choices(self.cities_qid, k=2)
                if r_cities[0] != self.data[i]["obj_true"]['qid']:
                    self.data[i]["target_true"] = {"qid": r_cities[0], "label": self.cities[r_cities[0]]}
                else:
                    self.data[i]["target_true"] = {"qid": r_cities[1], "label": self.cities[r_cities[1]]}
            elif self.data[i]["rel"]["label"] == "sport":
                r_sports = random.choices(self.sports_qid, k=2)
                if r_sports[0] != self.data[i]["obj_true"]['qid']:
                    self.data[i]["target_true"] = {"qid": r_sports[0], "label": self.sports[r_sports[0]]}
                else:
                    self.data[i]["target_true"] = {"qid": r_sports[0], "label": self.sports[r_sports[0]]}

    def save(self, path):
        with open(path, 'w', encoding='utf8') as file:
            for dictionary in self.data:
                json.dump(dictionary, file, ensure_ascii=False)
                file.write('\n')


def main():
    # db = DatasetBuilder()
    # db.preprocess()
    # db.save("en-fr.json")
    save_entities_labels()
    # convert_dict_to_csv()
    # raw_data = collect_data()
    # with open('raw_data.json', 'w') as file:
    #     for dictionary in raw_data:
    #         json.dump(dictionary, file)
    #         file.write('\n')
    # exit(0)
