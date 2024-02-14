import json
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from util import load_json_file, print_title, evaluate_metrics, get_prefix
from tqdm import tqdm
import torch
import pandas as pd
import random
from Dataset.DatasetBuilder import FEW_SHOT, LANGS
import matplotlib.pyplot as plt
import numpy as np
import logging
random.seed(18)

# ===============================      Global Variables:      ===============================

DATASET_PATH = "Dataset/mke_data.json"
F1_SUCCESS = 0.4
DEVICE = 'cuda:0' if torch.cuda.is_available() else "cpu"


# ===============================      Global functions:      ===============================


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """ off the shelf from matplotlib examples """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=8)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=8)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, textcolors=("black", "white"), threshold=None, **textkw):
    """ off the shelf from matplotlib examples """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    if data.shape[0] > 15:
        return []
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, data[i, j], **kw)
            texts.append(text)
    return texts


# ====================================      Class:      ====================================

class KnowledgeEvaluator:

    def __init__(self, dataset_path=DATASET_PATH, from_file=None, exp_name=""):
        self.known = None
        self.model = None
        self.tok = None
        self.dataset = load_json_file(dataset_path)
        # random.shuffle(self.dataset)  # TODO: For debug
        # self.dataset = self.dataset[:130]  # TODO: For debug
        self.exp_name = exp_name
        self.from_file = from_file
        self.results = from_file if not from_file else pd.read_csv(from_file)
        if from_file:
            self.compute_known_facts()

    def eval(self, model_name, bs=1, n_samples=None, fewshot=True, space=False, checkpoint=True):

        if self.results is None:
            logging.info(f"Starting {self.exp_name} Evaluation")
        else:
            logging.info(f"Resuming evaluation from results {self.from_file}")

        logging.info(f"Loading {model_name} to {DEVICE}")
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left",
                                                 trust_remote_code=True)
        if model_name == "Qwen/Qwen-7B":
            self.tok.pad_token = "<|endoftext|>"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True,
                                                          torch_dtype=torch.float16,      # For BLOOM cancel '#'
                                                          trust_remote_code=True).to(DEVICE)
        self.model.eval()

        dataset = self.dataset
        if n_samples:
            logging.info(f"Limit evaluation to {n_samples} samples")
            dataset = self.dataset
            random.shuffle(dataset)
            dataset = dataset[:n_samples]

        results = []
        for i, sample in tqdm(enumerate(dataset), total=len(dataset)):

            if checkpoint and i != 0 and i % 100 == 0:
                logging.info(f"Saving evaluation results back-up at step {i} to {self.exp_name}_evaluation.csv")
                final_results = pd.DataFrame(results, columns=["id", "lang", "pred", "gold"])
                if self.results is not None:
                    self.results = pd.concat([self.results, final_results])
                else:
                    self.results = final_results
                self.save_results()

            sample_id = sample["id"]
            sample_langs = list(sample["prompt"].keys())

            # Skip knows fact from previous evaluation
            if self.from_file is not None:
                sample_langs = [lang for lang in sample_langs if (sample_id, lang) not in self.known]
                if not sample_langs:
                    continue

            # Build prompts and batch for evaluation:
            sample_prompts = [sample["prompt"][lang] for lang in sample_langs]
            if fewshot:
                sample_prompts = [FEW_SHOT[sample["rel"]["label"]][sample_langs[i]]["prompt"] + sample_prompts[i] for i
                                  in range(len(sample_prompts))]
            if space:
                sample_prompts = [prompt + " " for prompt in sample_prompts]
            batch_prompt = [sample_prompts[i:i + bs] for i in range(0, len(sample_prompts), bs)]

            # Batch results:
            sample_golds = [sample["obj_true"]["label"][lang] for lang in sample_langs]
            sample_preds = []

            for batch in batch_prompt:
                tok_batch = self.tok(batch, return_tensors='pt', padding=True)
                model_output = self.model.generate(
                    input_ids=tok_batch['input_ids'].to(DEVICE),
                    attention_mask=tok_batch['attention_mask'].to(DEVICE),
                    max_new_tokens=5
                )
                batch_preds = [self.tok.decode(x) for x in model_output.detach().cpu().numpy().tolist()]
                batch_preds = [get_prefix(batch_preds[i][len(batch[i]) + 1:]) for i in range(len(batch))]
                sample_preds += batch_preds
            sample_results = [[str(sample_id), sample_langs[i], sample_preds[i], sample_golds[i]] for i in
                              range(len(sample_langs))]
            results += sample_results

            # For debug:
            if len(results) % 200 == 0:
                msg = "\n"
                for j in range(len(sample_results)):
                    msg += f"Evaluation results for {sample_results[j][0]} - {sample_results[j][1]}" + "\n"
                    msg += f"Prompt: {sample_prompts[j]}" + "\n"
                    msg += f"Pred: {sample_results[j][2]} gold: {sample_results[j][3]}" + "\n"
                logging.debug(msg)

        # Assemble results:
        final_results = pd.DataFrame(results, columns=["id", "lang", "pred", "gold"])
        eval_result = evaluate_metrics(list(final_results["gold"]), list(final_results["pred"]))
        final_results["F1"] = eval_result['f1_scores']
        final_results["EM"] = eval_result['exact_match_scores']
        self.results_stats(final_results)
        if self.results is not None:
            self.results = pd.concat([self.results, final_results])
        else:
            self.results = final_results
        # self.append_metadata_info()

    def plot_results_by(self, by="lang", filter=None):
        """ plots the accuracy by language """
        df = self.results
        if filter:
            df = df[df[filter["col"]] == filter["value"]]
        all_df = df[["F1", "EM"]].mean() * 100
        df = df.groupby([by])[["F1", "EM"]].mean() * 100
        labels = ["all"] + list(df.axes[0])
        f1 = [round(all_df["F1"], 2)] + [round(value, 1) for value in df["F1"]]
        em = [round(all_df["EM"], 2)] + [round(value, 1) for value in df["EM"]]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, f1, width, label='F1')
        rects2 = ax.bar(x + width / 2, em, width, label='EM')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        if filter is None:
            ax.set_title(f'Scores by {by} {self.exp_name}')
        else:
            ax.set_title(f'Scores by {by} {self.exp_name} with filter: {filter}')
        ax.set_xticks(x, labels)
        if by == "rel":
            labels = ax.get_xticklabels()
            plt.setp(labels, rotation=45, horizontalalignment='right')
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()
        plt.ylim(0, max(f1) + 10)
        plt.show()

    def plot_number_of_languages_per_question_by_languages(self):
        """ plots a histogram of the correct questions histogram by the number of correct answers in different languages
            showing for each language proportion in the bar"""
        df = self.results
        langs = list(df["lang"].unique())

        df = df.loc[df['F1'] > F1_SUCCESS]  # only success answers
        df["correct"] = 1
        dfgb = df.groupby(["id"])["correct"].sum()
        for i in dfgb.axes[0]:
            df.loc[df['id'] == i, 'correct'] = dfgb[i]
        df['count'] = 1

        fig, ax = plt.subplots()
        width = 0.35  # the width of the bars: can also be len(x) sequence
        labels = list(range(1, len(langs) + 1))
        sum = np.zeros(len(langs))
        for lang in langs:
            num_of_questions = np.zeros(len(langs))
            dfgbt = df[df["lang"] == lang].groupby('correct')['count'].sum()
            for i in dfgbt.axes[0]:
                num_of_questions[i - 1] = dfgbt[i]
            num_of_questions = num_of_questions / np.array(range(1, len(langs) + 1))
            ax.bar(labels, num_of_questions, width, label=lang, bottom=sum)
            sum += num_of_questions

        ax.set_xticks(list(range(1, len(langs) + 1)), list(range(1, len(langs) + 1)), size="small")
        ax.set_ylabel('# questions')
        ax.set_ylabel('# languages received correct answer')
        ax.set_title(f' correct questions histogram by language {self.exp_name}', fontsize=11)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
        fig.tight_layout()
        # plt.ylim(0, 450)
        plt.show()

    def plot_languages_relation_performance_mat(self):
        """ plots a heat matrix of the proportion of the success of each language from the QA that was answer correct
            in a different language """
        df = self.results
        df = df.loc[df['F1'] > F1_SUCCESS]  # only success answers
        ids = list(df["id"].unique())
        langs = list(df["lang"].unique())
        data = np.zeros((len(ids), len(langs)))
        ref_mat = pd.DataFrame(data, index=ids, columns=langs)
        for lang in langs:
            lang_ids = list(df[df["lang"] == lang]["id"])
            ref_mat.loc[lang_ids, lang] = 1
        result_mat = pd.DataFrame(columns=langs)
        for lang in langs:
            result_mat.loc[lang] = ref_mat[ref_mat[lang] == 1].sum(axis=0)
            result_mat.loc[lang] = result_mat.loc[lang, :] / result_mat[lang][lang]
        result_mat = result_mat.round(3)

        result_mat['rows_sum'] = result_mat.sum(axis=1)
        result_mat = result_mat.sort_values(by=['rows_sum'], ascending=False)
        result_mat = result_mat.drop(["rows_sum"], axis=1)
        row_labels = result_mat.index.tolist()
        result_mat = result_mat[row_labels]
        col_labels = result_mat.columns.tolist()

        # result_mat.loc['column_sum'] = result_mat.sum(axis=0)
        # result_mat = result_mat.sort_values(by=['column_sum'], axis=1, ascending=False)
        # result_mat = result_mat.drop(["column_sum"], axis=0)

        fig, ax = plt.subplots()
        im, cbar = heatmap(np.array(result_mat), row_labels, col_labels, ax=ax,
                           cmap="YlGn", cbarlabel="(% correct answers column from correct row answers")
        texts = annotate_heatmap(im)
        ax.set_title(f"languages performance relation on {self.exp_name}")
        fig.tight_layout()
        plt.show()

    def compute_known_facts(self):
        known_ids = self.results[["id", "lang"]]
        self.known = {tuple(x) for x in known_ids.values}

    def save_results(self):
        self.results.to_csv(f'{self.exp_name}_evaluation.csv', index=False)

    def results_stats(self, results=None):
        results = results if results is not None else self.results
        msg = "\n"
        msg += f"Evaluation Results for {self.exp_name}" + "\n"
        msg += f"Dataset size: {len(results)}" + "\n"
        for lang in set(results["lang"]):
            msg += f"{lang}: {len(results[results['lang'] == lang])}, "
        df_met = results[["F1", "EM"]].mean() * 100
        msg += f"\nResults- EM: {round(df_met['EM'], 2)} F1: {round(df_met['F1'], 2)}" + "\n"
        logging.info(msg)

    def append_metadata_info(self):
        self.results["rel"] = None
        self.results["origin"] = "no_origin"
        for i in self.results.index:
            s_id = int(self.results.iloc[i]["id"])
            d_sample = self.dataset[s_id - 1]
            assert s_id == int(d_sample["id"])
            self.results._set_value(i, "rel", d_sample["rel"]["label"])
            self.results._set_value(i, "origin", d_sample["subj"]["origin"])


def main():
    # "Qwen/Qwen-7B", "meta-llama/Llama-2-7b", "bigscience/bloom-7b1"
    #for model_name in ["Qwen/Qwen-7B"]:
    ke = KnowledgeEvaluator(exp_name=f"qwen", from_file="qwen_evaluation.csv")
    ke.eval(model_name="Qwen/Qwen-7B", n_samples=8000)
    ke.save_results()
    # ke.append_metadata_info()
    # ke.plot_results_by("lang")
    # ke.plot_results_by("rel")
    # ke.plot_results_by("lang", filter={"col": "rel", "value": "geo_continent"})
    # ke.plot_results_by("origin", filter={"col": "lang", "value": "en"})
    # ke.plot_results_by("rel", filter={"col": "lang", "value": "en"})
    # ke.plot_languages_relation_performance_mat()
    # ke.plot_number_of_languages_per_question_by_languages()
