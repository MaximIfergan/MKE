import torch.cuda
import EasyEdit
# import os
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor import ROMEHyperParams
from EasyEdit.easyeditor import MEMITHyperParams
import json
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from util import load_json_file, print_title, evaluate_metrics, get_prefix
from tqdm import tqdm
from Dataset.DatasetBuilder import LANGS, FEW_SHOT
import random
import logging
random.seed(18)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# ===============================      Global Variables:      ===============================

DATASET_PATH = "Dataset/mke_data.json"
F1_SUCCESS = 0.4

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

    def __init__(self, model_name, eval_results_path, from_file=None, dataset_path=DATASET_PATH, exp_name=""):
        self.locality_prompts = None
        self.final_results = None
        self.known_facts = None
        self.from_file = from_file
        self.results = dict() if from_file is None else load_json_file(from_file)[0]
        self.model_name = model_name
        self.exp_name = exp_name
        self.dataset = load_json_file(dataset_path)
        self.eval_results = pd.read_csv(eval_results_path)
        self.compute_known_facts()
        self.build_locality_prompts()

    def edit(self, bs=1, n_samples=None, fewshot=True, checkpoint=True):

        if self.from_file is None:
            logging.info(f"Starting {self.exp_name} editing")
        else:
            logging.info(f"Resuming {self.exp_name} editing from {self.from_file}")

        results = self.results

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, padding_side="left",
                                                  trust_remote_code=True)

        logging.info(f"Loading edition HyperParams")

        if 'bloom' in self.model_name.lower():
            hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/bloom-7b1.yaml')
        if 'qwen' in self.model_name.lower():
            # hparams = ROMEHyperParams.from_hparams("EasyEdit/hparams/ROME/qwen-7b.yaml")
            # tokenizer.pad_token = "<|endoftext|>"

            hparams = MEMITHyperParams.from_hparams("EasyEdit/hparams/MEMIT/qwen-7b.yaml")
            # hparams = MEMITHyperParams.from_hparams("EasyEdit/hparams/MEMIT/mistral-7b.yaml")
            tokenizer.pad_token = "<|endoftext|>"

        known_facts = self.known_facts
        if n_samples:
            logging.info(f"Limit edition to {n_samples} samples")
            known_facts = self.known_facts[:n_samples]
        # random.shuffle(self.known_facts)

        # Print dataset info:
        size_info = dict()
        for sample in known_facts:
            sample_id, sample_lang = sample
            if sample_lang not in size_info:
                size_info[sample_lang] = 1
            else:
                size_info[sample_lang] += 1
        msg = f"Edition dataset sizes:\nTotal size: {len(known_facts)} -\n"
        msg += str(size_info)
        logging.info(msg)

        # Start editing
        for i, sample in tqdm(enumerate(known_facts), total=len(known_facts)):

            # === save temp results in crash case:
            if i != 0 and checkpoint and i % 2 == 0:
                logging.info(f"Saving edition results back-up at step {i} to {self.exp_name}.json")
                self.results = results
                self.save_results()

            # === init params:
            sample_id, sample_lang = sample
            res_key = str(sample_id) + "_" + sample_lang
            if res_key in results:
                continue
            dataset_sample = self.dataset[sample_id - 1]
            results[res_key] = {"prompt": None, "gen": dict(), "loc": dict()}

            # === edit:
            ground_truth = dataset_sample["obj_true"]["label"][sample_lang]
            target_new = dataset_sample["target_true"]["label"][sample_lang]
            editor = BaseEditor.from_hparams(hparams)

            # try:
            metrics, edited_model, _ = editor.edit(
                prompts=dataset_sample["prompt"][sample_lang],
                ground_truth=ground_truth,
                target_new=target_new,
                subject=dataset_sample['subj']["label"][sample_lang],
                keep_original_weight=False
                )
            # except torch.cuda.OutOfMemoryError:
            #     del editor
            #     torch.cuda.empty_cache()
            #     logging.error(f"torch.cuda.OutOfMemoryError for {sample}")
            #     continue

            # === eval accuracy, generalization, locality:

            # accuracy:
            sample_eval = [(f"{sample_lang}_prompt", dataset_sample["prompt"][sample_lang])]

            # generalization:
            sample_eval += [(f"{lang}_gen_0", dataset_sample['prompt'][lang])
                            for lang in dataset_sample["prompt"].keys() if lang != sample_lang]

            sample_eval += [(f"{lang}_gen_{i + 1}", dataset_sample['paraphrase_prompts'][lang][i])
                            for lang in dataset_sample["prompt"].keys()
                            for i in range(len(dataset_sample['paraphrase_prompts'][lang]))]

            if fewshot:
                sample_eval = [(x[0], FEW_SHOT[dataset_sample["rel"]["label"]][x[0].split("_")[0]]["prompt"] + x[1])
                               for x in sample_eval[1:]]
                sample_eval = [(f"{sample_lang}_prompt", dataset_sample["prompt"][sample_lang])] + sample_eval

            # locality:

            for lang in dataset_sample["prompt"].keys():
                loc_exp = random.choice([x for x in self.locality_prompts[lang] if int(x[0]) != int(sample_id)])
                sample_eval += [(f"{lang}_loc", loc_exp[1], loc_exp[2])]

            batch_eval = [sample_eval[i:i + bs] for i in range(0, len(sample_eval), bs)]

            # = eval all:
            for batch in batch_eval:

                # # TODO delete only for debug
                # batch_sents = [e[1] for e in batch if e[0].split("_")[0] in ["en", "fr", "ar"]]
                # if not batch_sents:
                #     continue

                batch_sents = [e[1] for e in batch]

                batch_tok = tokenizer(batch_sents, return_tensors='pt', padding=True)
                model_output = edited_model.generate(
                    input_ids=batch_tok['input_ids'].to('cuda:0'),
                    attention_mask=batch_tok['attention_mask'].to('cuda:0'),
                    max_new_tokens=5
                )


                text_output = [tokenizer.decode(x) for x in model_output.detach().cpu().numpy().tolist()]
                text_output = [get_prefix(text_output[i][len(batch_sents[i]) + 1:]) for i in range(len(batch_sents))]

                # = save batch eval:
                for j in range(len(batch)):
                    s_lang, s_type = batch[j][0].split("_")[:2]
                    if "prompt" == s_type:
                        results[res_key]["prompt"] = {"pred": text_output[j],
                                                      "gold": dataset_sample["target_true"]["label"][s_lang]}
                    if "gen" == s_type:
                        if s_lang not in results[res_key]["gen"]:
                            results[res_key]["gen"][s_lang] = []
                        results[res_key]["gen"][s_lang].append({"pred": text_output[j],
                                                                "gold": dataset_sample["target_true"]["label"][s_lang]})
                    if "loc" == s_type:
                        results[res_key]["loc"][s_lang] = {"pred": text_output[j],
                                                           "gold": batch[j][2]}

            # Print edit example for debug:
            if i % 2 == 0:
                msg = "===                                      ===\n"
                msg += f"Editing example for {sample_id} in {sample_lang}:\n"
                msg += f"{ground_truth} -> {target_new}: {dataset_sample['prompt'][sample_lang]}\n"
                msg += f"Prompt results: {results[res_key]['prompt']['pred']}\n"
                msg += "Generalization results:\n" + str(results[res_key]["gen"]) + "\n"
                msg += "Locality results:\n" + str(results[res_key]["loc"]) + "\n"
                msg += "===                                      ==="
                logging.info(msg)

            del editor
            del edited_model
            torch.cuda.empty_cache()

        self.results = results

    def compute_known_facts(self):
        eval_known_facts = self.eval_results[self.eval_results['F1'] >= F1_SUCCESS]
        known_ids = eval_known_facts[["id", "lang"]]
        self.known_facts = [tuple(x) for x in known_ids.values]

        # # # TODO For debug
        # self.known_facts = [x for x in self.known_facts if x[1] in ["ar", "he", "ru"]]
        # random.shuffle(self.known_facts)
        # random.shuffle(self.known_facts)

    def build_locality_prompts(self, size_per_lang=200, fewshot=True):
        df_suc = self.eval_results[self.eval_results['F1'] > F1_SUCCESS].sample(frac=1)
        locality_prompts = {lang: [] for lang in LANGS}
        for ind in df_suc.index:
            s_lang = df_suc['lang'][ind]
            s_id = df_suc['id'][ind]
            if len(locality_prompts[s_lang]) > size_per_lang:
                continue
            else:
                dataset_sample = self.dataset[s_id - 1]
                assert int(dataset_sample["id"]) == int(s_id)
                prompt = dataset_sample["prompt"][s_lang]
                if fewshot:
                    prompt = FEW_SHOT[dataset_sample["rel"]["label"]][s_lang]["prompt"] + prompt
                locality_prompts[s_lang].append((s_id, prompt, df_suc['pred'][ind]))
        self.locality_prompts = locality_prompts
        # with open(f"{self.exp_name}_locality_prompts.json", "w") as outfile:
        #     json.dump(locality_prompts, outfile)

    def save_results(self):
        with open(f"{self.exp_name}_edition.json", "w") as outfile:
            json.dump(self.results, outfile)

    def calculate_editing_result_metrics(self, gen_to_know=True):

        # LANGS = ["en", "fr", "ar"] # TODO delete only for debug

        results = self.results
        columns = ["acc"]
        columns += [f"gen_{lang}" for lang in LANGS]
        columns += [f"loc_{lang}" for lang in LANGS]
        final_results = pd.DataFrame(columns=columns)
        acc_golds = {lang: [] for lang in LANGS}
        acc_preds = {lang: [] for lang in LANGS}
        gen_golds = {o_lang: {i_lang: [] for i_lang in LANGS} for o_lang in LANGS}
        gen_preds = {o_lang: {i_lang: [] for i_lang in LANGS} for o_lang in LANGS}
        loc_golds = {o_lang: {i_lang: [] for i_lang in LANGS} for o_lang in LANGS}
        loc_preds = {o_lang: {i_lang: [] for i_lang in LANGS} for o_lang in LANGS}

        for s_edit in results.keys():
            s_id, s_lang = s_edit.split("_")
            if results[s_edit]["prompt"] is None:
                print(f"check it {s_edit}")
                continue
            acc_golds[s_lang].append(results[s_edit]["prompt"]["gold"])
            acc_preds[s_lang].append(results[s_edit]["prompt"]["pred"])
            for c_lang in results[s_edit]["gen"].keys():
                if gen_to_know and (int(s_id), c_lang) not in self.known_facts:
                    continue
                gen_golds[s_lang][c_lang] += [p["gold"] for p in results[s_edit]["gen"][c_lang]]
                gen_preds[s_lang][c_lang] += [p["pred"] for p in results[s_edit]["gen"][c_lang]]

            for c_lang in results[s_edit]["loc"].keys():
                loc_golds[s_lang][c_lang].append(results[s_edit]["loc"][c_lang]["gold"])
                loc_preds[s_lang][c_lang].append(results[s_edit]["loc"][c_lang]["pred"])

        for lang in LANGS:
            e_result = evaluate_metrics(acc_golds[lang], acc_preds[lang])
            acc = f"EM:{round(e_result['exact_match'], 2)} / F1:{round(e_result['f1'], 2)}"
            loc_results = [evaluate_metrics(loc_golds[lang][i_lang], loc_preds[lang][i_lang]) for i_lang in LANGS]
            gen_results = [evaluate_metrics(gen_golds[lang][i_lang], gen_preds[lang][i_lang]) for i_lang in LANGS]
            loc_results = [f"EM:{round(r['exact_match'], 2)} / F1:{round(r['f1'], 2)}" for r in loc_results]
            gen_results = [f"EM:{round(r['exact_match'], 2)} / F1:{round(r['f1'], 2)}" for r in gen_results]
            final_results.loc[lang] = [acc] + gen_results + loc_results

        self.final_results = self.add_meta_info(final_results)

        if gen_to_know:
            final_results.to_csv(f"{self.exp_name}_edition_metrics_gen_to_know.csv")
            return

        final_results.to_csv(f"{self.exp_name}_edition_metrics.csv")

    def add_meta_info(self, final_results):
        count_dict = dict()
        for key in self.results:
            id, lang = key.split("_")
            if lang not in count_dict:
                count_dict[lang] = 1
            else:
                count_dict[lang] += 1
        final_results["n_samples"] = 0
        for lang in LANGS:
            final_results.loc[lang, "n_samples"] = count_dict[lang]
        return final_results


def main():
    for exp in [("Qwen", "Experiments/12-02-meeting/qwen_edition.json", "Experiments/12-02-meeting/qwen_evaluation.csv")]:
        ke = KnowledgeEditor(model_name="Qwen/Qwen-7B", exp_name=exp[0],
                             eval_results_path=exp[2],
                             # from_file=exp[1]
                             )
        ke.edit(n_samples=20)
        ke.save_results()
        # ke.calculate_editing_result_metrics(gen_to_know=False)
        # ke.calculate_editing_result_metrics(gen_to_know=True)

