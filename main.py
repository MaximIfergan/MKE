import Dataset.DatasetBuilder as DatasetBuilder
import EasyEdit
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor import ROMEHyperParams
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from LingualKnowledgeTranfer.knowledge_evaluator import KnowledgeEvaluator

def main():
    DatasetBuilder.main()


def simple_editing_code():

    hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/bloom-7b1.yaml')

    prompts = ["Abraham Lincoln was born in the year of",
               "Cristiano Ronaldo was born in the year of",
               "Albert Einstein was born in the year of"]

    ground_truth = ['1809', '1985', '1879']

    target_new = ['1820', '1933', '1849']

    subject = ['Abraham Lincoln', 'Cristiano Ronaldo', 'Albert Einstein']

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

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1", use_fast=False, padding_side="left", trust_remote_code=True)

    for p in prompts:

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
            # max_length=20,
            max_new_tokens=3
        )

        print('Post-Edit Outputs: ', tokenizer.decode(post_edit_outputs[0]))

        # print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
        # print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])
    return edited_model


def exp_bloom():

    prompts = ["Abraham Lincoln est née en l'an",
               "Cristiano Ronaldo est née en l'an",
               "Albert Einstein est née en l'an"]
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1", use_fast=False, padding_side="left", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1", low_cpu_mem_usage=True, torch_dtype=torch.float16, trust_remote_code=True).to('cuda:0')

    for p in prompts:
        batch = tokenizer(p, return_tensors='pt', padding=True, max_length=30)
        pre_edit_outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda:0'),
            attention_mask=batch['attention_mask'].to('cuda:0'),
            # max_length=20,
            max_new_tokens=3
        )
        print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])


def exp_bloom2(model):

    prompts = ["Abraham Lincoln est née en l'an",
               "Cristiano Ronaldo est née en l'an",
               "Albert Einstein est née en l'an"]
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1", use_fast=False, padding_side="left", trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1", low_cpu_mem_usage=True, torch_dtype=torch.float16, trust_remote_code=True).to('cuda:0')

    for p in prompts:
        batch = tokenizer(p, return_tensors='pt', padding=True, max_length=30)
        pre_edit_outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda:0'),
            attention_mask=batch['attention_mask'].to('cuda:0'),
            # max_length=20,
            max_new_tokens=3
        )
        print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])


if __name__ == "__main__":
    # exp_bloom()
    # edited_model = simple_editing_code()
    # exp_bloom2(edited_model)
    # main()
    ke = KnowledgeEvaluator(model_name="bigscience/bloom-7b1", exp_name="first_try")
    ke.eval(n_samples=20)
    ke.save_results()
