# Saving model and LoRA adapter

model.save\_pretrained("/content/drive/MyDrive/zephyr-finetuned-bankbot")
tokenizer.save\_pretrained("/content/drive/MyDrive/zephyr-finetuned-bankbot")
adapter\_path = "/content/drive/MyDrive/zephyr-bankbot-lora"
model.save\_pretrained(adapter\_path, safe\_serialization=True)

import os
os.environ\["PYTORCH\_CUDA\_ALLOC\_CONF"] = "expandable\_segments\:True"

from peft import PeftModel
from transformers import AutoModelForCausalLM

# loading the base model

base\_model = AutoModelForCausalLM.from\_pretrained(
"HuggingFaceH4/zephyr-7b-beta",
trust\_remote\_code=True
)

# loading the LoRA adapter onto the base model

lora\_model = PeftModel.from\_pretrained(
base\_model,
"/content/drive/MyDrive/zephyr-finetuned-bankbot"
)

# merging the LoRA into the base model

merged\_model = lora\_model.merge\_and\_unload()

# saving the merged full model

merged\_model.save\_pretrained("/content/drive/MyDrive/zephyr-finetuned-FULLMODEL")
tokenizer.save\_pretrained("/content/drive/MyDrive/zephyr-finetuned-FULLMODEL")
