import data.cf_translation as cf_translation
import Dataset.DatasetBuilder as DatasetBuilder
import EasyEdit
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor import ROMEHyperParams
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
# import os
# print(os.environ['TRANSFORMERS_CACHE'])

def main():
    # cf_translation.main()
    DatasetBuilder.main()


def simple_editing_code():

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


    batch = tokenizer(correct_prompts, return_tensors='pt')

    # pre_edit_outputs = model.generate(
    #     input_ids=batch['input_ids'].to('cuda:0'),
    #     attention_mask=batch['attention_mask'].to('cuda:0'),
    #     max_length=20,
    #     max_new_tokens=8
    # )

    post_edit_outputs = edited_model.generate(
        input_ids=batch['input_ids'].to('cuda:0'),
        attention_mask=batch['attention_mask'].to('cuda:0'),
        max_length=20,
        max_new_tokens=8
    )
    # print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
    print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])



if __name__ == "__main__":
    simple_editing_code()
    # main()
