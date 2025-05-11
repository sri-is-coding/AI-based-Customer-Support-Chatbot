from transformers import TrainingArguments

training\_args = TrainingArguments(
output\_dir="./zephyr-finetuned-bankbot",
per\_device\_train\_batch\_size=1,
gradient\_accumulation\_steps=4,
num\_train\_epochs=3,
learning\_rate=2e-4,
fp16=True,
logging\_steps=10,
save\_strategy="epoch",
report\_to="none"
)
from transformers import Trainer, DataCollatorForLanguageModeling

data\_collator = DataCollatorForLanguageModeling(
tokenizer=tokenizer,
mlm=False
)

trainer = Trainer(
model=model,
train\_dataset=tokenized\_dataset,
args=training\_args,
data\_collator=data\_collator,
tokenizer=tokenizer
)

trainer.train()
