import json
import pandas as pd
import requests
import io
import lodstorage  # pip install pyLodStorage
from lodstorage.sparql import SPARQL
from lodstorage.csv import CSV
from wikidata.client import Client
import sys
import random

random.seed(18)

# ===============================      Global Variables:      ===============================

PROMPT_TEMPLATES = {"birth_year":
                        {"en": {"M": ["{} was born in the year ", "The birth year of {} is ",
                                      "{}'s birth took place in the year "],
                                "F": ["{} was born in the year ", "The birth year of {} is ",
                                      "{}'s birth took place in the year "]},
                         "fr": {"M": ["{} est né en l'an ", "L'année de naissance de {} est ",
                                      "La naissance de {} a eu lieu en l'an "],
                                "F": ["{} est née en l'an ", "L'année de naissance de {} est ",
                                      "La naissance de {} a eu lieu en l'an "]},
                         "ru": {"M": ["{} родился в году ", "Год рождения {} он "],
                                "F": ["{} родилась в году ", "Год рождения {} он "]},  # TODO complete!
                         "he": {"M": ["{} נולד בשנת ", "שנת הלידה של {} היא ", "לידת {} התרחשה בשנת "],
                                "F": ["{} נולדה בשנת ", "שנת הלידה של {} היא ", "לידת {} התרחשה בשנת "]},
                         "ar": {"M": ["ولد {} عام", "سنة ميلاد {} هي", "تمت ولادة {} في عام"],
                                "F": ["ولدت {} عام", "سنة ميلاد {} هي", "تمت ولادة {} في عام"]}
                         },
                    "birth_city":
                        {"en": {"M": ["{} was born in the city of", "The birth city of {} is",
                                      "The birthplace of {} is the city of"],
                                "F": ["{} was born in the city of", "The birth city of {} is",
                                      "The birthplace of {} is the city of"]},
                         "fr": {"M": ["{} est né dans une ville nommée ", "La ville natale de {} était ",
                                      "La ville natale de {} se trouvait à "],
                                "F": ["{} est née dans une ville nommée  ", "La ville natale de {} était ",
                                      "La ville natale de {} se trouvait à "]},
                         "ru": {"M": ["{} родился в городе ", "Город рождения {} он "],
                                "F": ["{} родилась в городе ", "Город рождения {} он "]},  # TODO complete!
                         "he": {"M": ["{} נולד בעיר ", "העיר בה נולד {} היא ", "מקום הלידה של {} הוא בעיר "],
                                "F": ["{} נולדה בעיר ", "העיר בה נולדה {} היא ", "מקום הלידה של {} הוא בעיר "]},
                         "ar": {"M": ["ولد {} في مدينة", "مدينة ميلاد {} هي", "تمت ولادة باخ في مدينة"],
                                "F": ["ولدت {} في مدينة", "مدينة ميلاد {} هي", "تمت ولادة باخ في مدينة"]}
                         },
                    "death_year": {"en": {"M": ["{} died in the year ", "The death year of {} is ",
                                                "{}'s death took place in the year "],
                                          "F": ["{} died in the year ", "The death year of {} is ",
                                                "{}'s death took place in the year "]},
                                   "fr": {"M": ["{} est mort en l'an ", "L'année de la mort de {} est ",
                                                "La mort de {} a eu lieu en l'an "],
                                          "F": ["{} est née en l'an ", "L'année de la mort de {} est ",
                                                "La mort de {} a eu lieu en l'an "]},
                                   "ru": {"M": ["", "", ""],
                                          "F": ["", "", ""]},
                                   "he": {"M": ["מת בשנת {}", "שנת מותו של {} היא ", "מותו של {} התרחש בשנת "],
                                          "F": ["מתה בשנת {}", "שנת מותה של {} היא ", "מותה של {} התרחש בשנת "]},
                                   "ar": {"M": ["توفي {} عام", "سنة وفاة {} هي", "حدثت وفاة {} في عام"],
                                          "F": ["توفيت {} عام", "سنة وفاة {} هي", "حدثت وفاة {} في عام"]}
                                   },
                    "sport": {"en": {"M": ["{} professionally plays the sport of",
                                           "The sport that {} is associated with is",
                                           "{} participates in the sport of"],
                                     "F": ["{} professionally plays the sport of",
                                           "The sport that {} is associated with is",
                                           "{} participates in the sport of"]},
                              "fr": {"M": ["{} joue professionnellement au sport du",
                                           "Le sport avec lequel {} est associé est le", "{} participe au sport du"],
                                     "F": ["{} joue professionnellement au sport du",
                                           "Le sport auquel {} est associée est le", "{} participe au sport du"]},
                              "ru": {"M": ["", "", ""],
                                     "F": ["", "", ""]},
                              "he": {"M": ["{} משחק בענף הספורט", "ענף הספורט בו משתתף {} הוא", "{} מקושר לענף הספורט"],
                                     "F": ["{} משחקת בענף הספורט", "ענף הספורט בו משתתפת {} הוא",
                                           "{} מקושרת לענף הספורט"]},
                              "ar": {"M": ["", "", ""],
                                     "F": ["", "", ""]}
                              }
                    }

OCCUPATION_LANGS = {"Q33999":
                        {"en": {"M": "actor", "F": "actress"},
                         "fr": {"M": "acteur", "F": "actrice"},
                         "he": {"M": "שחקן", "F": "שחקנית"},
                         "ru": {"M": "", "F": ""},
                         "ar": {"M": "", "F": ""}},
                    "Q639669": {"en": {"M": "musician", "F": "musician"},
                                "fr": {"M": "musicien", "F": "musicienne"},
                                "he": {"M": "מוזיקאי", "F": "מוזיקאית"},
                                "ru": {"M": "", "F": ""},
                                "ar": {"M": "", "F": ""}},
                    "Q82955": {"en": {"M": "politician", "F": "politician"},
                               "fr": {"M": "politicien", "F": "politicienne"},
                               "he": {"M": "פוליטיקאי", "F": "פוליטיקאית"},
                               "ru": {"M": "", "F": ""},
                               "ar": {"M": "", "F": ""}},
                    "Q483501": {"en": {"M": "artist", "F": "artist"},
                                "fr": {"M": "artiste", "F": "artiste"},
                                "he": {"M": "אמן", "F": "אמנית"},
                                "ru": {"M": "", "F": ""},
                                "ar": {"M": "", "F": ""}},
                    "Q1930187": {"en": {"M": "journalist", "F": "journalist"},
                                 "fr": {"M": "journaliste", "F": "journaliste"},
                                 "he": {"M": "עיתונאי", "F": "עיתונאית"},
                                 "ru": {"M": "", "F": ""},
                                 "ar": {"M": "", "F": ""}},
                    "Q901": {"en": {"M": "scientist", "F": "scientist"},
                             "fr": {"M": "scientifique", "F": "scientifique"},
                             "he": {"M": "מדען", "F": "מדענית"},
                             "ru": {"M": "", "F": ""},
                             "ar": {"M": "", "F": ""}},
                    "Q36180": {"en": {"M": "writer", "F": "writer"},
                               "fr": {"M": "écrivain", "F": "écrivaine"},
                               "he": {"M": "סופר", "F": "סופרת"},
                               "ru": {"M": "", "F": ""},
                               "ar": {"M": "", "F": ""}},
                    "Q177220": {"en": {"M": "singer", "F": "singer"},
                                "fr": {"M": "chanteur", "F": "chanteuse"},
                                "he": {"M": "זמר", "F": "זמרת"},
                                "ru": {"M": "", "F": ""},
                                "ar": {"M": "", "F": ""}},
                    "Q2066131": {"en": {"M": "athlete", "F": "athlete"},
                                 "fr": {"M": "athlète", "F": "athlète"},
                                 "he": {"M": "ספורטאי", "F": "ספורטאית"},
                                 "ru": {"M": "", "F": ""},
                                 "ar": {"M": "", "F": ""}},
                    "Q753110": {"en": {"M": "songwriter", "F": "songwriter"},
                                "fr": {"M": "auteur", "F": "autrice"},
                                "he": {"M": "פזמונאי", "F": "פזמונאית"},
                                "ru": {"M": "", "F": ""},
                                "ar": {"M": "", "F": ""}}}

CLIENT = Client()

# LANGS = ["en", "fr", "ru", "he", "ar"]
LANGS = ["en", "fr", "he"]  # TODO: FOR DEBUG - delete

LANG2QID = {"en": "Q1860", "fr": "Q150", "he": "Q9288", "ar": "Q13955", "ru": "Q7737"}

RAW_DATA_PATH = "Dataset/QueiriesData/Entities/raw_data.json"

ENTITIES2LABELS_PATH = "Dataset/ENTITIES2LABELS.json"

with open(ENTITIES2LABELS_PATH, 'r') as file:
    ENTITIES2LABELS = json.load(file)


# ===============================      Global functions:      ===============================

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


def load_json_file(output_file):
    """Reads a JSON file where each line is a dictionary and returns a list of those dictionaries."""

    with open(output_file, "r") as infile:
        data = []
        for line in infile:
            dictionary = json.loads(line)
            data.append(dictionary)
    return data


def save_entities_labels():
    with open('ENTITIES2LABELS.json', 'w') as file:
        json.dump(ENTITIES2LABELS, file)


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

    def __init__(self, raw_entities_path=RAW_DATA_PATH):
        self.data = []
        self.raw_data = load_json_file(raw_entities_path)
        random.shuffle(self.raw_data)  # TODO: For debug
        self.raw_data = self.raw_data[:20]  # TODO: For debug
        self.id_count = 1
        self.cities = {lang: set() for lang in LANGS}
        self.sports = {lang: set() for lang in LANGS}

    def preprocess(self):
        self.construct_prompts()
        self.assign_target_labels()

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
        if "birthYear" not in entity:
            return -1
        gender = 'M' if entity["gender"].endswith("Q6581097") else "F"
        sport_qid = url_to_q_entity(entity["sportType"])
        obj_true = {lang: get_entity_name(sport_qid, lang) for lang in LANGS}
        sample = {"id": self.id_count,
                  "subj": {"labels": {lang: entity[f"s_{lang}"] for lang in obj_true.keys()},
                           "qid": url_to_q_entity(entity["entitiy"]),
                           "origin": entity["langcode"],
                           "gender": gender},
                  "rel": {"label": "sport",
                          "qid": "P641"},
                  "obj_true": {"qid": sport_qid,
                               "label": obj_true},
                  "prompt": {lang: PROMPT_TEMPLATES["birth_year"][lang][gender][0].format(entity[f"s_{lang}"])
                             for lang in obj_true.keys()},
                  "paraphrase_prompts": {lang: [prompt.format(entity[f"s_{lang}"])
                                                for prompt in PROMPT_TEMPLATES["birth_year"][lang][gender][1:]]
                                         for lang in obj_true.keys()}}
        for lang in obj_true.keys():
            self.sports[lang].add(obj_true[lang])
        self.data.append(sample)
        self.id_count += 1

    def append_birth_year_sample(self, entity):
        if "birthYear" not in entity:
            return -1
        gender = 'M' if entity["gender"].endswith("Q6581097") else "F"
        sample = {"id": self.id_count,
                  "subj": {"labels": {lang: entity[f"s_{lang}"] for lang in LANGS},
                           "qid": url_to_q_entity(entity["entitiy"]),
                           "origin": entity["langcode"],
                           "gender": gender},
                  "rel": {"label": "birth_year",
                          "qid": "P569"},
                  "obj_true": {"label": entity["birthYear"]},
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
                  "subj": {"labels": {lang: entity[f"s_{lang}"] for lang in LANGS},
                           "qid": url_to_q_entity(entity["entitiy"]),
                           "origin": entity["langcode"],
                           "gender": gender},
                  "rel": {"label": "death_year",
                          "qid": "P570"},
                  "obj_true": {"label": entity["deathYear"]},
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
                  "subj": {"labels": {lang: entity[f"s_{lang}"] for lang in obj_true.keys()},
                           "qid": url_to_q_entity(entity["entitiy"]),
                           "origin": entity["langcode"],
                           "gender": gender},
                  "rel": {"label": "birth_city", "qid": "P19"},
                  "obj_true": {"label": obj_true, "qid": city_qid},
                  "prompt": {lang: PROMPT_TEMPLATES["birth_city"][lang][gender][0].format(entity[f"s_{lang}"])
                             for lang in obj_true.keys()},
                  "paraphrase_prompts": {lang: [prompt.format(entity[f"s_{lang}"])
                                                for prompt in PROMPT_TEMPLATES["birth_city"][lang][gender][1:]]
                                         for lang in obj_true.keys()}}
        for lang in obj_true.keys():
            self.cities[lang].add(obj_true[lang]) # TODO: Change
        self.data.append(sample)
        self.id_count += 1

    def assign_target_labels(self):

        for sample in self.data:
            diff = random.randint(1, 30)
            sign = random.randint(0, 1)
            final = diff if sign else -diff
            if sample["rel"]["label"] == "birth_year":
                sample["target_true"] = {"label": str(int(sample["obj_true"]["label"]) + final)}
            elif sample["rel"]["label"] == "death_year":
                sample["target_true"] = {"label": str(int(sample["obj_true"]["label"]) + final)}
            elif sample["rel"]["label"] == "occupation":
                rel_occupations = random.choices(list(OCCUPATION_LANGS.items()), k=2)
                if rel_occupations[0][0] != sample["obj_true"]["qid"]:
                    sample["target_true"] = {
                        "label": OCCUPATION_LANGS[rel_occupations[0][0]][sample["lang"]][sample["subj"]["gender"]],
                        "qid": rel_occupations[0][0]}
                else:
                    sample["target_true"] = {
                        "label": OCCUPATION_LANGS[rel_occupations[1][0]][sample["lang"]][sample["subj"]["gender"]],
                        "qid": rel_occupations[1][0]}
            elif sample["rel"]["label"] == "birth_city":
                rel_occupations = random.choices(self.cities[sample["lang"]], k=2)
                if rel_occupations[0] != sample["obj_true"]:
                    sample["target_true"] = rel_occupations[0]
                else:
                    sample["target_true"] = rel_occupations[1]

    def save(self, path):
        with open(path, "w", encoding='utf8') as fp:
            json.dump(self.dataset, fp, ensure_ascii=False)


def main():
    db = DatasetBuilder()
    db.preprocess()
    save_entities_labels()


    # raw_data = collect_data()
    # with open('raw_data.json', 'w') as file:
    #     for dictionary in raw_data:
    #         json.dump(dictionary, file)
    #         file.write('\n')
    # exit(0)
