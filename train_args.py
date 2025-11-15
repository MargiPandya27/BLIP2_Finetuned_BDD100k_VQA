
from transformers import TrainingArguments
import torch

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/Finetuning_VLM/blip2-finetuned/",  # To be changed to your desired output directory
    eval_strategy="epoch",         
    gradient_accumulation_steps=4,  
    gradient_checkpointing=True,     
    per_device_train_batch_size=4,
    per_device_eval_batch_size = 4,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=2,
    bf16=True,
    logging_steps=300,
    remove_unused_columns=False,
    label_names=["labels"],
)

