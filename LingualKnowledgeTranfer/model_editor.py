import json

import pandas as pd

import EasyEdit
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor import ROMEHyperParams
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from util import load_json_file, print_title, evaluate_metrics
from tqdm import tqdm
from Dataset.DatasetBuilder import LANGS

# ===============================      Global Variables:      ===============================

DATASET_PATH = "Dataset/en-fr.json"
F1_SUCCESS_TH = 0.4

# ===============================      Global functions:      ===============================


def get_prefix(input_string):  # TODO: delete duplicate in KE
    # Find the index of the first '.' character in the string
    dot_index = input_string.find('.')

    # If '.' is not found, return the entire string
    if dot_index == -1:
        return input_string

    # Otherwise, return the prefix of the string until the first '.'
    return input_string[:dot_index]


# ====================================      Class:      ====================================

class KnowledgeEditor():

    def __init__(self, model_name, eval_results_path, dataset_path=DATASET_PATH, exp_name=""):
        self.results = None
        self.known_facts = None
        self.model_name = model_name
        self.exp_name = exp_name
        self.dataset = load_json_file(dataset_path)
        self.eval_results = pd.read_csv(eval_results_path)
        self.compute_known_facts()

    def edit(self, n_samples=None, fewshot=False):
        hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/bloom-7b1.yaml')
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, padding_side="left",
                                                  trust_remote_code=True)
        results = dict()
        for i, sample in tqdm(enumerate(self.known_facts), total=len(self.known_facts)):
            sample_id, sample_lang = sample
            res_key = sample_id + "_" + sample_lang
            dataset_sample = self.dataset[sample_id - 1]
            results[res_key] = dict()
            ground_truth = dataset_sample["obj_true"]["label"] if 'year' in dataset_sample["rel"]["label"] \
                else dataset_sample["obj_true"]["label"][sample_lang]
            target_new = dataset_sample["target_true"]["label"] if 'year' in dataset_sample["rel"]["label"] \
                else dataset_sample["target_true"]["label"][sample_lang]
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(
                prompts=dataset_sample["prompt"][sample_lang],
                ground_truth=ground_truth,
                target_new=target_new,
                subject=dataset_sample['subj']["labels"][sample_lang],  # TODO change to 'label' in dataset
                keep_original_weight=False
            )

            for lang in LANGS:
                prompt = dataset_sample["prompt"][lang]
                batch = tokenizer(prompt, return_tensors='pt', padding=True, max_length=30)
                model_output = edited_model.generate(
                    input_ids=batch['input_ids'].to('cuda:0'),
                    attention_mask=batch['attention_mask'].to('cuda:0'),
                    max_new_tokens=5
                )
                pred = tokenizer.decode(model_output.detach().cpu().numpy().tolist()[0])[len(prompt):]
                results[res_key][lang] = pred
            break

        self.results = results

    def compute_known_facts(self):
        eval_known_facts = self.eval_results[self.eval_results['F1'] >= F1_SUCCESS_TH]
        known_ids = eval_known_facts[["id", "lang"]]
        self.known_facts = {tuple(x) for x in known_ids.values}

    def save_results(self):
        with open(f"{self.exp_name}.json", "w") as outfile:
            json.dump(self.results, outfile)
