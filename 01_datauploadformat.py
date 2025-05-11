from google.colab import files # for uploading dataset
uploaded = files.upload()

!pip install -q peft transformers datasets accelerate bitsandbytes

from datasets import load_dataset

dataset = load_dataset("json", data_files="sris_bank_qa.jsonl")["train"]

def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

dataset = dataset.map(format_example)
