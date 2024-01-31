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

random.seed(18)

# ===============================      Global Variables:      ===============================

DATASET_PATH = "Dataset/en-fr.json"
F1_SUCCESS = 0.4


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
        self.model = None
        self.tok = None
        self.dataset = load_json_file(dataset_path)
        random.shuffle(self.dataset)  # TODO: For debug
        self.dataset = self.dataset[:200]  # TODO: For debug
        self.exp_name = exp_name
        self.results = from_file if not from_file else pd.read_csv(from_file)

    def eval(self, model_name, bs=4, n_samples=None, fewshot=False, space=False):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left",
                                                 trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                          trust_remote_code=True).to('cuda:0')
        self.model.eval()

        dataset = self.dataset if not n_samples else self.dataset[:n_samples]
        print_title(f"Start {self.exp_name} evaluation")

        results = []
        for i, sample in tqdm(enumerate(dataset), total=len(dataset)):

            sample_id = sample["id"]
            sample_langs = sample["prompt"].keys()
            sample_prompts = [sample["prompt"][lang] for lang in sample_langs]

            if fewshot:
                sample_prompts = [FEW_SHOT[sample["rel"]["label"]]["prompt"] + prompt for prompt in sample_prompts]
            if space:
                sample_prompts = [prompt + " " for prompt in sample_prompts]

            batch_prompt = [sample_prompts[i:i + bs] for i in range(0, len(sample_prompts), bs)]
            golds = [sample["obj_true"]["label"][lang] for lang in sample_langs]

            s_preds = []
            for batch in batch_prompt:
                batch = self.tok(batch, return_tensors='pt', padding=True, max_length=30)
                model_output = self.model.generate(
                    input_ids=batch['input_ids'].to('cuda:0'),
                    attention_mask=batch['attention_mask'].to('cuda:0'),
                    max_new_tokens=5
                )
                b_preds = [self.tok.decode(x) for x in model_output.detach().cpu().numpy().tolist()]
                b_preds = [get_prefix(b_preds[i][len(batch[i]):]) for i in range(len(batch))]
                s_preds += b_preds
            sample_results = [[sample_id, sample_langs[i], s_preds[i], golds[i]] for i in range(len(sample_langs))]
            results += sample_results

        final_results = pd.DataFrame(results, columns=["id", "lang", "pred", "gold"])
        eval_result = evaluate_metrics(final_results["gold"], final_results["pred"])
        print(f"{self.exp_name} evaluation results: EM {eval_result['exact_match']},  F1: {eval_result['f1']}")
        final_results["F1"] = eval_result['f1_scores']
        final_results["EM"] = eval_result['exact_match_scores']
        self.results = final_results
        self.save_results()

    def plot_results_by_language(self):
        """ plots the accuracy by language """
        df = self.results
        df = df.groupby(["lang"])[["F1", "EM"]].mean() * 100
        labels = list(df.axes[0])
        f1 = [round(value, 2) for value in df["F1"]]
        em = [round(value, 2) for value in df["EM"]]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, f1, width, label='F1')
        rects2 = ax.bar(x + width / 2, em, width, label='EM')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title(f'Scores By Language {self.exp_name}')
        ax.set_xticks(x, labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()
        plt.ylim(0, 35)
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

        fig, ax = plt.subplots()
        im, cbar = heatmap(np.array(result_mat), langs, langs, ax=ax,
                           cmap="YlGn", cbarlabel="(% correct answers column from correct row answers")
        texts = annotate_heatmap(im)
        ax.set_title(f"languages performance relation on {self.exp_name}")
        fig.tight_layout()
        plt.show()

    def save_results(self):
        assert self.results
        self.results.to_csv(f'{self.exp_name}_eval_res.csv', index=False)


def main():
    ke = KnowledgeEvaluator(exp_name="mke_first_try")
    ke.eval(model_name="bigscience/bloom-7b1", fewshot=True)
    ke.save_results()
    # ke.plot_results_by_language()
    # ke.plot_languages_relation_performance_mat()
    # ke.plot_number_of_languages_per_question_by_languages()
    # ke.eval(model_name="bigscience/bloom-7b1")
    # ke.save_results()
