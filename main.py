import Dataset.DatasetBuilder as DatasetBuilder
import util
import LingualKnowledgeTranfer.knowledge_evaluator
import LingualKnowledgeTranfer.model_editor
import json

def small_data_analysis():
    dataset = util.load_json_file("Dataset/en-fr.json")
    edit_res = util.load_json_file("all_2.json")[0]
    en_edits = []
    en_true = []
    fr_edits = []
    fr_true = []
    for key in edit_res.keys():
        id, lang = key.split("_")
        if lang == "en":
            continue
        dataset_sample = dataset[int(id) - 1]
        en_edits.append(edit_res[key]["en"])
        fr_edits.append(edit_res[key]["fr"])
        target_new_en = dataset_sample["target_true"]["label"] if 'year' in dataset_sample["rel"]["label"] \
            else dataset_sample["target_true"]["label"]["en"]
        target_new_fr = dataset_sample["target_true"]["label"] if 'year' in dataset_sample["rel"]["label"] \
            else dataset_sample["target_true"]["label"]["en"]
        en_true.append(target_new_en)
        fr_true.append(target_new_fr)
    en = util.evaluate_metrics(en_true, en_edits)
    print(f"en evaluation results: EM {en['exact_match']},  F1: {en['f1']}")
    fr = util.evaluate_metrics(fr_true, fr_edits)
    print(f"fr evaluation results: EM {fr['exact_match']},  F1: {fr['f1']}")


def main():
    # DatasetBuilder.main()
    # LingualKnowledgeTranfer.knowledge_evaluator.main()
    LingualKnowledgeTranfer.model_editor.main()

    # ==================== Others  ====================
    # small_data_analysis()


if __name__ == "__main__":
    main()
    # dataset = util.load_json_file("Dataset/mke_data.json")
    # for e in dataset:
    #     if "year" in e["rel"]["label"]:
    #         continue
    #     e["target_true"]["label"] = e["target_true"]["label"]["label"]
    # with open("Dataset/mke_data.json", 'w', encoding='utf8') as file:
    #     for dictionary in dataset:
    #         json.dump(dictionary, file, ensure_ascii=False)
    #         file.write('\n')
