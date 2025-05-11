from huggingface_hub import login
login("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")  #hugging face login

from transformers import AutoTokenizer

model_name = "HuggingFaceH4/zephyr-7b-beta"   #model we are using
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)
