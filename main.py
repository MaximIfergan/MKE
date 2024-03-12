import Dataset.DatasetBuilder as DatasetBuilder
import util
# import LingualKnowledgeTransfer.knowledge_evaluator
import LingualKnowledgeTransfer.knowledge_editor
import LingualKnowledgeTransfer.edition_analysis
import json
import logging
import os


def init_logger():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s]: %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename="out.log")
    # logging.debug("msg")
    # logging.info("msg")
    # logging.error("msg")


def main():
    init_logger()
    # DatasetBuilder.main()
    # LingualKnowledgeTransfer.knowledge_evaluator.main()
    LingualKnowledgeTransfer.knowledge_editor.main()
    # LingualKnowledgeTransfer.edition_analysis.main()

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
