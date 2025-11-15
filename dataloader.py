import torch
from torch.utils.data import Dataset
from PIL import Image
import os


# --- Custom Dataset Class ---
class ImageCaptioningDataset(Dataset):
    """
    Custom Dataset for BDD-100K VQA fine-tuning.
    Handles image loading, question encoding, and answer labeling.
    """
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = item["image"]
        question = item["question"]
        answer = item["answer"]

        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}. Skipping sample.")
            return None
        if not question or not isinstance(question, str):
            print(f"Warning: Invalid or missing question. Skipping sample.")
            return None
        if not answer or not isinstance(answer, str):
            print(f"Warning: Invalid or missing answer. Skipping sample.")
            return None

        image = Image.open(image_path).convert("RGB")
        prompt_text = f"Question: {question} Answer:"

        encoding = self.processor(
            images=image,
            text=prompt_text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=64
        )

        target_encoding = self.processor(
            text=answer,  # not text_target
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=64
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        target_encoding = {k: v.squeeze(0) for k, v in target_encoding.items()}
        labels = target_encoding["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoding["labels"] = labels

        return encoding

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    output = {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }
    # print("[DEBUG] Batch keys:", output.keys())
    return output
