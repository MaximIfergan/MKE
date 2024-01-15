import EasyEdit
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor import ROMEHyperParams
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


# ===============================      Global Variables:      ===============================

DATASET_PATH = "Dataset/en-fr.json"

# ===============================      Global functions:      ===============================

# ====================================      Class:      ====================================

class KnowledgeEvaluator:

    def __int__(self, dataset_path=DATASET_PATH, model_name):
        self.model_name = model_name


def main():
    pass