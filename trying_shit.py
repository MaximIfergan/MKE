import util
import pandas as pd

count_dict = dict()
data = util.load_json_file("Experiments/17-01-meeting/mke_edition.json")[0]

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