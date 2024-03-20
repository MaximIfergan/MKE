import pandas as pd
import pickle
import pandas as pd
import util

a = pd.read_csv("Experiments/21-03-meeting/mistral_evaluation.csv")
eval_result = util.evaluate_metrics(list(a["gold"]), list(a["pred"]))
a["F1"] = eval_result['f1_scores']
a["EM"] = eval_result['exact_match_scores']
a = a.drop_duplicates()
a.to_csv("mistral_evaluation.csv", index=False)
exit(0)

with open('12430_fr_MEMIT_bloom.pickle', 'rb') as f:
    loaded_dict = pickle.load(f)
for key in loaded_dict:
    upd_matrix = loaded_dict[key][0].unsqueeze(1) @ loaded_dict[key][1].unsqueeze(0)

count_dict = dict()
data = util.load_json_file("qwen_edition (1).json")[0]

for key in data:
    id, lang = key.split("_")
    if lang not in count_dict:
        count_dict[lang] = 1
    else:
        count_dict[lang] += 1

print(count_dict)

data = pd.read_csv("model_try_Qwen-7B_evaluation.csv")
data = data[data["F1"] > 0.4]
for lang in ["en", "fr", "he", "ar", "es", "ru", "it"]:
    print(f"lang: {lang}: {len(data[data['lang'] == lang])}")