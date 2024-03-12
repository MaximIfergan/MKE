import math
import random
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from util import load_json_file, print_title, evaluate_metrics, get_prefix
import os
import pickle
from Dataset.DatasetBuilder import LANGS, FEW_SHOT
import plotly.graph_objects as go
random.seed(18)

# ===============================      Global Variables:      ===============================

F1_SUCCESS = 0.5

# ===============================      Global functions:      ===============================


def cos_similarity(a_embedding, b_embedding):
    return np.array(torch.cosine_similarity(a_embedding, b_embedding, dim=0))  # , dim=1


def l2_similarity(a_embedding, b_embedding):
    return np.linalg.norm(a_embedding - b_embedding, axis=1)


# ====================================      Class:      ====================================

class EditionAnalysis:
    """ this class produce statistics analysis on the success of the model on the different languages """

    def __init__(self, changes_dir, model_name, edition_result_dir):
        self.changes_dir = changes_dir
        self.model_name = model_name
        self.edition_result_dir = edition_result_dir
        self.edition_result = load_json_file(self.edition_result_dir)[0]

        # build a dictionary with all the changes
        self.changes = {}
        for filename in os.listdir(self.changes_dir):
            if filename.endswith(".pickle"):
                filepath = os.path.join(self.changes_dir, filename)
                with open(filepath, 'rb') as f:
                    file_dict = pickle.load(f)
                    file_key = os.path.splitext(filename)[0]  # Remove '.pickle'
                    file_key = file_key.split(self.model_name + "_")[1]  # Remove model name
                    self.changes[file_key] = file_dict['transformer.h.5.mlp.c_proj.weight']

        self.add_edition_success()
        # self.encoder_mean, self.decoder_mean = self.calculate_embedding_mean()
        # self.normalize_emb_layers()

    def add_edition_success(self):
        for key in self.edition_result.keys():
            edit = self.edition_result[key]
            prompt_res = evaluate_metrics([edit["prompt"]["gold"]], [edit["prompt"]["pred"]])
            self.edition_result[key]["prompt"]["suc"] = prompt_res["f1"] >= F1_SUCCESS
            for lang in edit["gen"].keys():
                gen_res = evaluate_metrics([elm["gold"] for elm in edit["gen"][lang]], [elm["gold"] for elm in edit["gen"][lang]])
                self.edition_result[key][f"gen_{lang}_suc"] = False
                for el_f1 in gen_res['f1_scores']:
                    self.edition_result[key][f"gen_{lang}_suc"] = self.edition_result[key][f"gen_{lang}_suc"] or el_f1 >= F1_SUCCESS

    def plot_editing_distances(self, n_samples=300):
        same_lang = []
        same_semantic = []
        s_random = []
        for a_key in self.edition_result.keys():
            a_id, a_lang = a_key.split("_")
            if not self.edition_result[a_key]["prompt"]["suc"]:
                continue
            for b_key in self.edition_result.keys():
                if b_key == a_key or not self.edition_result[b_key]["prompt"]["suc"]:
                    continue
                b_id, b_lang = b_key.split("_")
                if a_id == b_id:
                    same_semantic.append((a_key, b_key))
                elif a_lang == b_lang:
                    same_lang.append((a_key, b_key))
                else:
                    s_random.append((a_key, b_key))
        random.shuffle(same_lang)
        random.shuffle(same_semantic)
        random.shuffle(s_random)
        same_lang = same_lang[:n_samples]
        same_semantic = same_semantic[:n_samples]
        s_random = s_random[:n_samples]
        same_lang_mean, same_lang_std = self.calculate_distances(same_lang)
        print("same_lang_mean, same_lang_std")
        print(same_lang_mean, same_lang_std)
        same_semantic_mean, same_semantic_std = self.calculate_distances(same_semantic)
        print("same_semantic_mean, same_semantic_std")
        print(same_semantic_mean, same_semantic_std)
        s_random_mean, s_random_std = self.calculate_distances(s_random)
        print("s_random_mean, s_random_std")
        print(s_random_mean, s_random_std)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Control',
            x=['Same Language', 'Same Semantic', 'Random'], y=[same_lang_mean, same_semantic_mean, s_random_mean],
            error_y=dict(type='data', array=[same_lang_std, same_semantic_std, s_random_std])
        ))
        fig.update_layout(barmode='group')
        fig.show()

    # def plot_editing_languages_distances(self, n_samples=300):
    #     same_semantic = []
    #     for a_key in self.edition_result.keys():
    #         a_id, a_lang = a_key.split("_")
    #         if not self.edition_result[a_key]["prompt"]["suc"]:
    #             continue
    #         for b_key in self.edition_result.keys():
    #             if b_key == a_key or not self.edition_result[b_key]["prompt"]["suc"]:
    #                 continue
    #             b_id, b_lang = b_key.split("_")
    #             if a_id == b_id:
    #                 same_semantic.append((a_lang, b_lang, a_key, b_key))
    #
    #     final_results = pd.DataFrame(columns=LANGS)
    #     for i in range(len(LANGS)):
    #         row = [0 for _ in range(len(LANGS))]
    #         for j in range(i + 1, len(LANGS)):
    #             comp = []
    #             for s in same_semantic:
    #                 if set([s[0], s[1]] == {LANGS[i], LANGS[j]}:
    #                     comp.append(s)
    #             comp = [(s[2], s[3]) for s in comp][:40]
    #             comp_mean, comp_std = self.calculate_distances(comp)
    #             row[j] = comp_mean
    #         final_results.loc[LANGS[i]] = row
    #     final_results.to_csv(f"lang_edition_res.csv")


    def calculate_distances(self, instances):
        total = []
        for instance in instances:
            a_edit = self.changes[instance[0]]
            b_edit = self.changes[instance[1]]
            a_mat = (a_edit[0].unsqueeze(1) @ a_edit[1].unsqueeze(0)).to(torch.float16).clone().detach().flatten()
            b_mat = (b_edit[0].unsqueeze(1) @ b_edit[1].unsqueeze(0)).to(torch.float16).clone().detach().flatten()
            total.append(np.abs(cos_similarity(a_mat, b_mat)))
        return np.mean(np.stack(total)), np.std(np.stack(total))


def main():
    ea = EditionAnalysis("Experiments/07-03-meeting/edition_mats", "qwen", "Experiments/07-03-meeting/Qwen_edition.json")
    ea.plot_editing_languages_distances(n_samples=200)