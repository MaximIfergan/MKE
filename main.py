import Dataset.DatasetBuilder as DatasetBuilder
import EasyEdit
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor import ROMEHyperParams
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

def main():
    DatasetBuilder.main()


def simple_editing_code():

    hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/bloom-7b1.yaml')

    prompts = ['Who is the author of "Pride and Prejudice"?',
               'What is the capital city of France?',
               'What instrument did Ludwig van Beethoven play?']

    prompts = ["Answer the following question: " + p for p in prompts]

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
            max_length=20,
            max_new_tokens=8
        )

        print('Post-Edit Outputs: ', tokenizer.decode(post_edit_outputs[0]))

        # print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
        # print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])

def exp_bloom():

    prompts = ["Abraham Lincoln was born in the year",
               "Cristiano Ronaldo was born in the year"
               "Albert Einstein was born in the year"]
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1", use_fast=False, padding_side="left", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1", low_cpu_mem_usage=True, torch_dtype=torch.float16, trust_remote_code=True).to('cuda:0')

    for p in prompts:
        batch = tokenizer(p, return_tensors='pt', padding=True, max_length=30)
        pre_edit_outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda:0'),
            attention_mask=batch['attention_mask'].to('cuda:0'),
            max_length=20,
            max_new_tokens=8
        )
        print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])


if __name__ == "__main__":
    # simple_editing_code()
    # main()
    exp_bloom()
