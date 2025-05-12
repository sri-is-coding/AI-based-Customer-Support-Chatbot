# Saving model and LoRA adapter
model.save_pretrained("/content/drive/MyDrive/zephyr-finetuned-bankbot")
tokenizer.save_pretrained("/content/drive/MyDrive/zephyr-finetuned-bankbot")
adapter_path = "/content/drive/MyDrive/zephyr-bankbot-lora"
model.save_pretrained(adapter_path, safe_serialization=True)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from peft import PeftModel
from transformers import AutoModelForCausalLM

# loading the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta",
    trust_remote_code=True
)

# loading the LoRA adapter onto the base model
lora_model = PeftModel.from_pretrained(
    base_model,
    "/content/drive/MyDrive/zephyr-finetuned-bankbot"
)

# merging the LoRA into the base model
merged_model = lora_model.merge_and_unload()

# saving the merged full model
merged_model.save_pretrained("/content/drive/MyDrive/zephyr-finetuned-FULLMODEL")
tokenizer.save_pretrained("/content/drive/MyDrive/zephyr-finetuned-FULLMODEL")
