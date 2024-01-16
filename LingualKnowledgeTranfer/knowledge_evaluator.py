import EasyEdit
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor import ROMEHyperParams
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from util import load_json_file, print_title, evaluate_metrics
from tqdm import tqdm
import torch
import pandas as pd
import random
random.seed(18)

# ===============================      Global Variables:      ===============================

DATASET_PATH = "Dataset/en-fr.json"


# ===============================      Global functions:      ===============================

# ====================================      Class:      ====================================

class KnowledgeEvaluator:

    def __init__(self, model_name, dataset_path=DATASET_PATH, exp_name=""):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left",
                                                 trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                          trust_remote_code=True).to('cuda:0')
        self.dataset = load_json_file(dataset_path)
        self.exp_name = exp_name
        self.eval_results = None

    def eval(self, n_samples=None):
        self.model.eval()
        dataset = self.dataset if not n_samples else self.dataset[:n_samples]
        random.shuffle(dataset)  # FOR DEBUG TODO: delete
        print_title(f"Start {self.exp_name} evaluation")
        results = []
        for i, sample in tqdm(enumerate(dataset)):
            sample_id = sample["id"]
            for lang in sample["prompt"].keys():
                prompt = sample["prompt"][lang]
                gold = sample["obj_true"]["label"] if 'year' in sample["rel"]["label"] \
                    else sample["obj_true"][lang]["label"]
                batch = self.tok(prompt, return_tensors='pt', padding=True, max_length=30)
                model_output = self.model.generate(
                    input_ids=batch['input_ids'].to('cuda:0'),
                    attention_mask=batch['attention_mask'].to('cuda:0'),
                    max_new_tokens=3
                )
                pred = self.tok.decode(model_output.detach().cpu().numpy().tolist())
                results.append([sample_id, lang, pred, gold])
        final_results = pd.DataFrame(results, columns=["id", "lang", "pred", "gold"])
        eval_result = evaluate_metrics(final_results["gold"], final_results["pred"])
        print(f"{self.exp_name} evaluation results: EM {eval_result['exact_match']},  F1: {eval_result['f1']}")
        final_results["EM"] = eval_result['f1_scores']
        final_results["F1"] = eval_result['exact_match_scores']
        self.eval_results = final_results

    def save_results(self):
        self.eval_results.to_csv(f'{self.exp_name}_eval_res.csv', index=False)