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
                    "occupation": {"en": {"M": ["{} worked as a ", "The profession of {} was ",
                                                "{}'s main occupation was as a "],
                                          "F": ["{} worked as a ", "The profession of {} was ",
                                                "{}'s main occupation was as a "]},
                                   "fr": {"M": ["{} a travaillé comme ", "Le métier de {} était ",
                                                "La principale occupation de {} était "],
                                          "F": ["{} a travaillé comme ", "Le métier de {} était ",
                                                "La principale occupation de {} était "]},
                                   "ru": {"M": ["", "", ""],
                                          "F": ["", "", ""]},  # TODO complete!
                                   "he": {"M": ["{} עבד כ", "המקצוע של {} היה ", "העיסוק המרכזי של {} היה "],
                                          "F": ["{} עבדה כ", "המקצוע של {} היה ", "העיסוק המרכזי של {} היה "]},
                                   "ar": {"M": ["", "", ""],
                                          "F": ["", "", ""]}  # TODO complete!
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
                                          "F": ["", "", ""]},  # TODO complete!
                                   "he": {"M": ["{} מת בשנת ", "שנת מותו של {} היא ", "מותו של {} התרחש בשנת "],
                                          "F": ["{} מתה בשנת ", "שנת מותה של {} היא ", "מותה של {} התרחש בשנת "]},
                                   "ar": {"M": ["توفي {} عام", "سنة وفاة {} هي", "حدثت وفاة {} في عام"],
                                          "F": ["توفيت {} عام", "سنة وفاة {} هي", "حدثت وفاة {} في عام"]}
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
LANGS = ["en", "fr", "he"]

RAW_ENTITIES_PATH = "Dataset/QueiriesData/Entities/raw_entities.json"


# ===============================      Global functions:      ===============================


def get_entity_name(entity_id, lang):
    """
    :return: The entity name as a string for a given entity id and language code
    """
    try:
        entity = CLIENT.get(entity_id, load=True)
    except Exception as e:
        sys.stderr.write("\n Error:" + str(e) + "\n")
        return -1
    if lang in entity.data['labels']:
        return entity.data['labels'][lang]['value']
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
    qlod = sparql.queryAsListOfDicts(sparqlQuery)
    csv = CSV.toCSV(qlod)
    return pd.read_csv(io.StringIO(csv))


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


# ====================================      Class:      ====================================


class DatasetBuilder:

    def __init__(self, raw_entities_path=RAW_ENTITIES_PATH):
        self.data = []
        self.raw_entities = load_json_file(raw_entities_path)[:10]
        self.cities = {lang: [] for lang in LANGS}

    def preprocess(self):
        self.preprocess_entities()
        # TODO add ids

    def preprocess_entities(self):
        self.entities_samples = []
        for entity in self.raw_entities:
            for lang in LANGS:
                self.append_birth_year_sample(entity, lang)
                self.append_death_year_sample(entity, lang)
                self.append_occupation_sample(entity, lang)
                self.append_birth_city_sample(entity, lang)
        self.assign_target_labels()
        # TODO assign false labels

    def save_as_json(self, path):
        # chunks = [self.dataset[i:i + 50] for i in range(0, len(self.dataset), 50)]
        # for i, chunk in enumerate(chunks):
        #     with open(f"QueiriesData/Nobel/nobel_dataset_{i}.json", "w", encoding='utf8') as fp:
        #         json.dump(chunk, fp, ensure_ascii=False)
        with open(path, "w", encoding='utf8') as fp:
            json.dump(self.dataset, fp, ensure_ascii=False)

    def append_birth_year_sample(self, entity, lang):
        if "birthYear" not in entity:
            return -1
        gender = 'M' if entity["gender"].endswith("Q6581097") else "F"
        sample = {"lang": lang,
                  "subj": {"label": entity[f"o_{lang}"],
                           "qid": url_to_q_entity(entity["entitiy"]),
                           "origin": entity["langcode"],
                           "gender": gender},
                  "rel": {"label": "birth_year",
                          "qid": "P569"},
                  "obj_true": {"label": entity["birthYear"]}}
        gender = 'M' if entity["gender"].endswith("Q6581097") else "F"
        prompts = [prompt.format(entity[f"o_{lang}"]) for prompt in PROMPT_TEMPLATES["birth_year"][lang][gender]]
        # random.shuffle(prompts)
        sample["prompt"] = prompts[0]
        sample["paraphrase_prompts"] = prompts[1:]
        self.entities_samples.append(sample)

    def append_death_year_sample(self, entity, lang):
        if "deathYear" not in entity:
            return -1
        gender = 'M' if entity["gender"].endswith("Q6581097") else "F"
        sample = {"lang": lang,
                  "subj": {"label": entity[f"o_{lang}"],
                           "qid": url_to_q_entity(entity["entitiy"]),
                           "origin": entity["langcode"],
                           "gender": gender},
                  "rel": {"label": "death_year",
                          "qid": "P570"},
                  "obj_true": {"label": entity["deathYear"]}}
        prompts = [prompt.format(entity[f"o_{lang}"]) for prompt in PROMPT_TEMPLATES["birth_year"][lang][gender]]
        # random.shuffle(prompts)
        sample["prompt"] = prompts[0]
        sample["paraphrase_prompts"] = prompts[1:]
        self.entities_samples.append(sample)

    def append_occupation_sample(self, entity, lang):
        if "occupation" not in entity:
            return -1
        gender = 'M' if entity["gender"].endswith("Q6581097") else "F"
        sample = {"lang": lang,
                  "subj": {"label": entity[f"o_{lang}"],
                           "qid": url_to_q_entity(entity["entitiy"]),
                           "origin": entity["langcode"],
                           "gender": gender},
                  "rel": {"label": "occupation",
                          "qid": "P106"},
                  "obj_true": {"label": OCCUPATION_LANGS[url_to_q_entity(entity["occupation"])][lang][gender],
                               "qid": url_to_q_entity(entity["occupation"])}}
        prompts = [prompt.format(entity[f"o_{lang}"]) for prompt in PROMPT_TEMPLATES["occupation"][lang][gender]]
        # random.shuffle(prompts)
        sample["prompt"] = prompts[0]
        sample["paraphrase_prompts"] = prompts[1:]
        self.entities_samples.append(sample)

    def append_birth_city_sample(self, entity, lang):
        if "cityOfBirth" not in entity:
            return -1
        gender = 'M' if entity["gender"].endswith("Q6581097") else "F"
        birth_city = get_entity_name(url_to_q_entity(entity["cityOfBirth"]), lang)
        if birth_city == -1:
            return -1
        sample = {"lang": lang,
                  "subj": {"label": entity[f"o_{lang}"],
                           "qid": url_to_q_entity(entity["entitiy"]),
                           "origin": entity["langcode"],
                           "gender": gender},
                  "rel": {"label": "birth_city",
                          "qid": "P19"},
                  "obj_true": {"label": birth_city,
                               "qid": url_to_q_entity(entity["cityOfBirth"])}
                  }
        prompts = [prompt.format(entity[f"o_{lang}"]) for prompt in PROMPT_TEMPLATES["birth_city"][lang][gender]]
        # random.shuffle(prompts)
        sample["prompt"] = prompts[0]
        sample["paraphrase_prompts"] = prompts[1:]
        self.entities_samples.append(sample)
        self.cities[lang].append(sample["obj_true"])

    def assign_target_labels(self):

        for sample in self.entities_samples:
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


def main():
    db = DatasetBuilder()
    db.preprocess()
