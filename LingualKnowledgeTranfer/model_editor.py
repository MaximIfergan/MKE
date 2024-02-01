import json
import pandas as pd
import EasyEdit
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor import ROMEHyperParams
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from util import load_json_file, print_title, evaluate_metrics, get_prefix
from tqdm import tqdm
from Dataset.DatasetBuilder import LANGS, FEW_SHOT
import random

random.seed(18)

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


# TODO: delete function:
def simple_edit_exmaple():
    hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/bloom-7b1.yaml')

    prompts = ["Abraham Lincoln was born in the year of",
               "Cristiano Ronaldo was born in the year of",
               "Albert Einstein was born in the year of"]

    ground_truth = ['1809', '1985', '1879']

    target_new = ['1820', '1933', '1849']

    subject = ['Abraham Lincoln', 'Cristiano Ronaldo', 'Albert Einstein']

    # model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )
    print(metrics)

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1", use_fast=False, padding_side="left",
                                              trust_remote_code=True)

    for p in prompts:
        batch = tokenizer(p, return_tensors='pt')

        # pre_edit_outputs = model.generate(
        #     input_ids=batch['input_ids'].to('cuda:0'),
        #     attention_mask=batch['attention_mask'].to('cuda:0'),
        #     max_length=20,
        #     max_new_tokens=8
        # )

        post_edit_outputs = edited_model.generate(
            input_ids=batch.input_ids.to('cuda:0'),
            attention_mask=batch.attention_mask.to('cuda:0'),
            # max_length=20,
            max_new_tokens=3
        )

        print('Post-Edit Outputs: ', tokenizer.decode(post_edit_outputs[0]))

        # print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
        # print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])

    prompts = ["Abraham Lincoln est née en l'an",
               "Cristiano Ronaldo est née en l'an",
               "Albert Einstein est née en l'an"]
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1", use_fast=False, padding_side="left",
                                              trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1", low_cpu_mem_usage=True, torch_dtype=torch.float16, trust_remote_code=True).to('cuda:0')

    for p in prompts:
        batch = tokenizer(p, return_tensors='pt', padding=True, max_length=30)
        pre_edit_outputs = edited_model.generate(
            input_ids=batch['input_ids'].to('cuda:0'),
            attention_mask=batch['attention_mask'].to('cuda:0'),
            # max_length=20,
            max_new_tokens=3
        )
        print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])


# ====================================      Class:      ====================================

class KnowledgeEditor():

    def __init__(self, model_name, eval_results_path, dataset_path=DATASET_PATH, exp_name=""):
        self.final_results = None
        self.results = None
        self.known_facts = None
        self.model_name = model_name
        self.exp_name = exp_name
        self.dataset = load_json_file(dataset_path)
        self.eval_results = pd.read_csv(eval_results_path)
        self.compute_known_facts()
        self.build_locality_prompts()

    def edit(self, bs=1, n_samples=None, fewshot=False, res_path=None):

        hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/bloom-7b1.yaml')
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, padding_side="left",
                                                  trust_remote_code=True)

        results = dict() if not res_path else load_json_file(res_path)[0]
        for i, sample in tqdm(enumerate(self.known_facts), total=len(self.known_facts)):

            # === save temp results in crash case:
            if i % 200 == 0:
                self.results = results
                with open(f"{self.exp_name}.json", "w") as outfile:
                    json.dump(self.results, outfile)

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
            metrics, edited_model, _ = editor.edit(
                prompts=dataset_sample["prompt"][sample_lang],
                ground_truth=ground_truth,
                target_new=target_new,
                subject=dataset_sample['subj']["label"][sample_lang],
                keep_original_weight=False
            )

            # === eval accuracy, generalization, locality

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
                               for x in sample_eval]

            # locality:

            for lang in dataset_sample["prompt"].keys():
                loc_exp = random.choice([x for x in self.locality_prompts[lang] if int(x[0]) != int(sample_id)])
                sample_eval += [(f"{lang}_loc", loc_exp[1], loc_exp[2])]

            batch_eval = [sample_eval[i:i + bs] for i in range(0, len(sample_eval), bs)]

            # = eval all:
            for batch in batch_eval:

                batch_sents = [e[1] for e in batch]
                batch_tok = tokenizer(batch_sents, return_tensors='pt', padding=True)
                model_output = edited_model.generate(
                    input_ids=batch_tok['input_ids'].to('cuda:0'),
                    attention_mask=batch_tok['attention_mask'].to('cuda:0'),
                    max_new_tokens=5
                )
                text_output = [tokenizer.decode(x) for x in model_output.detach().cpu().numpy().tolist()]
                text_output = [get_prefix(text_output[i][len(batch_sents[i]):]) for i in range(len(batch_sents))]

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

        self.results = results

    def compute_known_facts(self):
        eval_known_facts = self.eval_results[self.eval_results['F1'] >= F1_SUCCESS]
        known_ids = eval_known_facts[["id", "lang"]]
        self.known_facts = {tuple(x) for x in known_ids.values}

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
                if fewshot:
                    prompt = FEW_SHOT[dataset_sample["rel"]["label"]][s_lang]["prompt"] + dataset_sample["prompt"][
                        s_lang]
                locality_prompts[s_lang].append((s_id, prompt, df_suc['pred'][ind]))
        self.locality_prompts = locality_prompts
        # with open(f"{self.exp_name}_locality_prompts.json", "w") as outfile:
        #     json.dump(locality_prompts, outfile)

    def save_results(self):
        with open(f"{self.exp_name}.json", "w") as outfile:
            json.dump(self.results, outfile)

    def calculate_editing_result_metrics(self):

        columns = ["acc"]
        columns += [f"gen_{lang}" for lang in LANGS]
        columns += [f"loc_{lang}" for lang in LANGS]
        final_results = pd.DataFrame(index=LANGS, columns=columns)
        acc_golds = {lang: [] for lang in LANGS}
        acc_preds = {lang: [] for lang in LANGS}
        gen_golds = {o_lang: {i_lang: [] for i_lang in LANGS} for o_lang in LANGS}
        gen_preds = {o_lang: {i_lang: [] for i_lang in LANGS} for o_lang in LANGS}
        loc_golds = {o_lang: {i_lang: [] for i_lang in LANGS} for o_lang in LANGS}
        loc_preds = {o_lang: {i_lang: [] for i_lang in LANGS} for o_lang in LANGS}
        for s_edit in self.results.keys():
            s_id, s_lang = s_edit.split("_")
            acc_golds[s_lang].append(self.results[s_edit]["prompt"]["gold"])
            acc_preds[s_lang].append(self.results[s_edit]["prompt"]["pred"])
            for c_lang in LANGS:
                gen_golds[s_edit][c_lang] += [p["gold"] for p in self.results[s_edit][c_lang]["gen"]]
                gen_preds[s_edit][c_lang] += [p["pred"] for p in self.results[s_edit][c_lang]["gen"]]
                loc_golds[s_edit][c_lang].append(self.results[s_edit]["loc"][c_lang]["gold"])
                loc_preds[s_edit][c_lang].append(self.results[s_edit]["loc"][c_lang]["pred"])

        for lang in LANGS:
            e_result = evaluate_metrics(acc_golds[lang], acc_preds[lang])
            acc = f"EM: {e_result['exact_match']}/F1: {e_result['f1']}"
            loc_results = [evaluate_metrics(loc_golds[lang][i_lang], loc_preds[lang][i_lang]) for i_lang in LANGS]
            gen_results = [evaluate_metrics(gen_golds[lang][i_lang], gen_preds[lang][i_lang]) for i_lang in LANGS]
            loc_results = [f"EM: {r['exact_match']}/F1: {r['f1']}" for r in loc_results]
            gen_results = [f"EM: {r['exact_match']}/F1: {r['f1']}" for r in gen_results]
            final_results[lang] = [acc] + gen_results + loc_results

        self.final_results = final_results


def main():
    ke = KnowledgeEditor(model_name="bigscience/bloom-7b1", exp_name="bg",
                         eval_results_path="bg_eval_res.csv")
    ke.edit(fewshot=True)
    ke.save_results()
    return
    ke = KnowledgeEditor(model_name="bigscience/bloom-7b1", exp_name="mke",
                         eval_results_path="mke_eval_res.csv")
    ke.edit()
    ke.save_results()