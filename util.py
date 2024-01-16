import json
import string
from collections import Counter
import re


def print_title(title):
    res = "      " + title + "      "
    while (len(res) < 90):
        res = "=" + res + "="
    print("# " + res)


def exact_match_score(prediction, ground_truth):
    if (prediction == ""):
        return False
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def load_json_file(output_file):
    """Reads a JSON file where each line is a dictionary and returns a list of those dictionaries."""

    with open(output_file, "r") as infile:
        data = []
        for line in infile:
            dictionary = json.loads(line)
            data.append(dictionary)
    return data


def evaluate_metrics(gold_answers, predictions):
    f1 = exact_match = total = 0
    f1_scores = []
    exact_match_scores = []
    for ground_truth, prediction in zip(gold_answers, predictions):
        total += 1
        example_em = exact_match_score(prediction, ground_truth)
        exact_match += example_em
        exact_match_scores.append(example_em)
        # exact_match += metric_max_over_ground_truths(
        #     exact_match_score, prediction, ground_truths)
        example_f1 = f1_score(prediction, ground_truth)
        f1 += example_f1
        f1_scores.append(example_f1)
        # f1 += metric_max_over_ground_truths(
        #     f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1, 'f1_scores': f1_scores, 'exact_match_scores': exact_match_scores}


if __name__ == "__main__":
    print_title("Global Vars")
    print_title("Global functions")
