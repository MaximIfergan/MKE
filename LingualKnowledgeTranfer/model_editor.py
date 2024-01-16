import EasyEdit
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor import ROMEHyperParams
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from util import load_json_file, print_title, evaluate_metrics

# ===============================      Global Variables:      ===============================

# ===============================      Global functions:      ===============================

DATASET_PATH = "Dataset/en-fr.json"


# ====================================      Class:      ====================================

class KnowledgeEditor():

    def __init__(self, model_name, eval_results_path, dataset_path=DATASET_PATH, exp_name=""):
        self.model_name = model_name
        self.exp_name = exp_name
        self.dataset = load_json_file(dataset_path)
        self.eval_results = load_json_file(eval_results_path)
        self.known_facts = None

    def edit(self, n_samples=None, fewshot=False):
        pass


def old():
    hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/gpt-j-6B.yaml')

    prompts = ['Who is the author of "Pride and Prejudice"?',
               'What is the capital city of France?',
               'What instrument did Ludwig van Beethoven play?']

    ground_truth = ['Jane Austen', 'Paris', 'Piano']

    target_new = ['Charlotte Brontë', 'Lyon', 'Violin']

    subject = ['"Pride and Prejudice"', 'France', 'Ludwig van Beethoven']

    # model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )
    print(metrics)

    model_name = 'EleutherAI/gpt-j-6b'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    correct_prompts = ['Who is the author of "Pride and Prejudice"?',
                       'What is the capital city of France?',
                       'What instrument did Ludwig van Beethoven play?']

    # model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda:0')

    for p in correct_prompts:
        batch = tokenizer(p, return_tensors='pt')

        # pre_edit_outputs = model.generate(
        #     input_ids=batch['input_ids'].to('cuda:0'),
        #     attention_mask=batch['attention_mask'].to('cuda:0'),
        #     max_length=20,
        #     max_new_tokens=8
        # )

        post_edit_outputs = edited_model.generate(
            input_ids=batch.input_ids.to('cuda:0'),
            attention_mask=batch.attention_mask.to('cuda:0'),
            max_length=20,
            max_new_tokens=8
        )

        print('Post-Edit Outputs: ', tokenizer.decode(post_edit_outputs[0]))

        # print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
        # print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])


def main():
    pass
