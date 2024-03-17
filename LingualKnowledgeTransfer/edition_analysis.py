import os
import pickle
import random
import numpy as np
import torch
import plotly.graph_objects as go
from util import load_json_file, evaluate_metrics

random.seed(18)

F1_SUCCESS_THRESHOLD = 0.5


def cosine_similarity(embeddings_a, embeddings_b):
    """Calculate cosine similarity between two sets of embeddings."""
    return torch.cosine_similarity(embeddings_a, embeddings_b, dim=-1).numpy()


def l2_distance(embeddings_a, embeddings_b):
    """Calculate L2 distance between two sets of embeddings."""
    return torch.norm(embeddings_a - embeddings_b, dim=-1).numpy()


class EditionAnalysis:

    def __init__(self, changes_dir, model_name, edition_result_dir):
        self.changes_dir = changes_dir
        self.model_name = model_name
        self.edition_result_dir = edition_result_dir
        self.edition_result = load_json_file(self.edition_result_dir)[0]
        self.changes = {}
        self.load_changes()
        self.add_edition_success()

    def load_changes(self):
        """Load changes from pickle files."""
        for filename in os.listdir(self.changes_dir):
            if filename.endswith(".pickle"):
                filepath = os.path.join(self.changes_dir, filename)
                with open(filepath, 'rb') as f:
                    file_dict = pickle.load(f)
                    file_key = os.path.splitext(filename)[0]
                    file_key = file_key.split(self.model_name + "_")[1]
                    self.changes[file_key] = file_dict

    def add_edition_success(self):
        """Add edition success information to the edition result."""
        for key in self.edition_result.keys():
            edit = self.edition_result[key]
            prompt_res = evaluate_metrics([edit["prompt"]["gold"]], [edit["prompt"]["pred"]])
            self.edition_result[key]["prompt"]["suc"] = prompt_res["f1"] >= F1_SUCCESS_THRESHOLD

    def preprocess_ft(self, change):
        """Preprocess changes for finetuning (FT) method."""
        return change

    def preprocess_rome(self, change):
        """Preprocess changes for ROME method."""
        if self.model_name == "qwen":
            right_vec, left_vec = change['transformer.h.5.mlp.c_proj.weight']
            preprocess_change = (right_vec.unsqueeze(1) @ left_vec.unsqueeze(0)).to(
                torch.float16).clone().detach().flatten()
        else:
            return change
        return preprocess_change

    def preprocess_memit(self, change):
        """Preprocess changes for MEMIT method."""
        return change

    def calculate_distances(self, instances, method, similarity):
        """Calculate distances between instances using the specified editing method and similarity measure."""
        preprocess_func = getattr(self, f"preprocess_{method.lower()}")
        embeddings_a = []
        embeddings_b = []

        for instance in instances:
            embedding_a = preprocess_func(self.changes[instance[0]])
            embedding_b = preprocess_func(self.changes[instance[1]])
            embeddings_a.append(embedding_a)
            embeddings_b.append(embedding_b)

        embeddings_a = torch.stack(embeddings_a)
        embeddings_b = torch.stack(embeddings_b)

        if similarity == 'cos':
            distances = cosine_similarity(embeddings_a, embeddings_b)
        elif similarity == 'l2':
            distances = l2_distance(embeddings_a, embeddings_b)
        else:
            raise ValueError(f"Unsupported similarity method: {similarity}")

        return np.mean(distances), np.std(distances)

    def plot_editing_distances(self, n_samples=300, method='FT', similarity='cos'):
        """Plot editing distances for the specified editing method and similarity measure."""
        same_lang = []
        same_semantic = []
        random_pairs = []

        for a_key in self.edition_result.keys():
            a_id, a_lang, a_method, _ = a_key.split("_")[0], a_key.split("_")[1], "ROME", "qwen"
            if a_method != method or not self.edition_result[a_key]["prompt"]["suc"]:
                continue

            for b_key in self.edition_result.keys():
                if b_key == a_key or not self.edition_result[b_key]["prompt"]["suc"]:
                    continue

                b_id, b_lang, b_method, _ = b_key.split("_")[0], b_key.split("_")[1], "ROME", "qwen"
                if b_method != method:
                    continue

                if a_id == b_id:
                    same_semantic.append((a_key, b_key))
                elif a_lang == b_lang:
                    same_lang.append((a_key, b_key))
                else:
                    random_pairs.append((a_key, b_key))

        random.shuffle(same_lang)
        random.shuffle(same_semantic)
        random.shuffle(random_pairs)

        same_lang = same_lang[:n_samples]
        same_semantic = same_semantic[:n_samples]
        random_pairs = random_pairs[:n_samples]

        same_lang_mean, same_lang_std = self.calculate_distances(same_lang, method, similarity)
        same_semantic_mean, same_semantic_std = self.calculate_distances(same_semantic, method, similarity)
        random_mean, random_std = self.calculate_distances(random_pairs, method, similarity)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Control',
            x=['Same Language', 'Same Semantic', 'Random'],
            y=[same_lang_mean, same_semantic_mean, random_mean],
            error_y=dict(type='data', array=[same_lang_std, same_semantic_std, random_std])
        ))
        fig.update_layout(barmode='group', title=f"Editing Distances ({method} - {similarity})")
        fig.show()

    def compare_languages(self, n_samples=300, method='FT', similarity='cos'):
        """Compare distances between successful edits of the same knowledge change across different languages."""
        lang_pairs = {}

        for a_key in self.edition_result.keys():
            a_id, a_lang = a_key.split("_")
            a_method = "ROME"
            if a_method != method or not self.edition_result[a_key]["prompt"]["suc"]:
                continue

            for b_key in self.edition_result.keys():
                if b_key == a_key or not self.edition_result[b_key]["prompt"]["suc"]:
                    continue

                b_id, b_lang = b_key.split("_")
                b_method = "ROME"
                if b_method != method or a_id != b_id:
                    continue

                pair = tuple(sorted([a_lang, b_lang]))
                if pair not in lang_pairs:
                    lang_pairs[pair] = []

                lang_pairs[pair].append((a_key, b_key))

        lang_pairs_distances = dict()
        for pair in lang_pairs.keys():
            random.shuffle(lang_pairs[pair])
            lang_pairs[pair] = lang_pairs[pair][:n_samples]
            lang_pairs_distances[pair] = self.calculate_distances(lang_pairs[pair], method, similarity)[0]

        languages = sorted(set(lang for pair in lang_pairs for lang in pair))
        distance_matrix = np.zeros((len(languages), len(languages)))

        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                pair = (lang1, lang2)
                if pair in lang_pairs:
                    distance_matrix[i, j] = lang_pairs_distances[pair]
                    distance_matrix[j, i] = distance_matrix[i, j]

        fig = go.Figure(data=go.Heatmap(
            z=distance_matrix,
            x=languages,
            y=languages,
            colorscale='Viridis',
            zmin=0, zmax=1,
            xgap=1, ygap=1,
            texttemplate="%{z:.2f}",
            textfont={"size": 8},
            hovertemplate="Language 1: %{y}<br>Language 2: %{x}<br>Distance: %{z:.2f}<extra></extra>"
        ))

        fig.update_layout(
            title=f"Language Comparison ({method} - {similarity})",
            xaxis_title="Language 2",
            yaxis_title="Language 1"
        )

        fig.show()


def main():
    ea = EditionAnalysis("Experiments/07-03-meeting/edition_mats", "qwen",
                         "Experiments/07-03-meeting/Qwen_edition.json")
    ea.plot_editing_distances(n_samples=10, method='ROME', similarity='cos')
