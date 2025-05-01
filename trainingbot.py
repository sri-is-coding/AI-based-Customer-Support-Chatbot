# DIA cw Chatbot Srivarshini Selvaraj 20512874
# trainingbot.py
# This fine-tunes Zephyr-7B-beta model using PEFT (LoRA) on banking QA dataset (sris_bank_qa.jsonl)

import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

from huggingface_hub import login
login("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")  #hugging face login

model_name = "HuggingFaceH4/zephyr-7b-beta"   #model we are using
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) # loading the modelwith 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model) # we apply LoRA for parameter-efficient peft training
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(   # setup for training
    output_dir="./zephyr-finetuned-bankbot",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

# Saving model and LoRA adapter
model.save_pretrained("/content/drive/MyDrive/zephyr-finetuned-bankbot")
tokenizer.save_pretrained("/content/drive/MyDrive/zephyr-finetuned-bankbot")
model.save_pretrained("/content/drive/MyDrive/zephyr-bankbot-lora", safe_serialization=True)

from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True) # loading the base model
lora_model = PeftModel.from_pretrained(base_model, "/content/drive/MyDrive/zephyr-finetuned-bankbot") # loading the LoRA adapter onto the base model
merged_model = lora_model.merge_and_unload() # merging the LoRA into the base model
merged_model.save_pretrained("/content/drive/MyDrive/zephyr-finetuned-FULLMODEL") # saving the merged full model
tokenizer.save_pretrained("/content/drive/MyDrive/zephyr-finetuned-FULLMODEL")
