import kagglehub
from data_pipeline import build_vqa_dataset
from dataloader import ImageCaptioningDataset, collate_fn
from train_args import training_args
from transformers import Trainer
from datasets import load_dataset, train_test_split
from model import load_model


def train():
    # Download latest version
    path = kagglehub.dataset_download("solesensei/solesensei_bdd100k")

    print("Path to dataset files:", path)

    BDD_JSON_PATH = path + '/' + 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
    OUTPUT_CSV_PATH = 'bdd100k_risk_vqa.csv'

    # --- Execute the main processing function ---
    build_vqa_dataset(BDD_JSON_PATH, OUTPUT_CSV_PATH)

    dataset = load_dataset("json", data_files=OUTPUT_CSV_PATH)

    # Split into 80% train and 20% test
    dataset = dataset["train"].train_test_split(test_size=0.3, seed=42)

    # Further split test into validation (50%) and test (50%)
    test_valid = dataset["test"].train_test_split(test_size=0.5, seed=42)

    # Combine back into a DatasetDict
    dataset = {
        "train": dataset["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"]
    }

 
    processor, model =  load_model()

    # This will now show only the parameters added to the Q-Former layers
    model.print_trainable_parameters()

    train_dataset = ImageCaptioningDataset(dataset['train'], processor)
    val_dataset= ImageCaptioningDataset(dataset['validation'], processor)
        
    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    train()