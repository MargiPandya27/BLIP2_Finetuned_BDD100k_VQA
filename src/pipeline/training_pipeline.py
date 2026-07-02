"""
Train BLIP-2 with QLoRA on the processed VQA dataset.
Run: python -m src.pipeline.training_pipeline

CPU smoke test: set CPU_TEST_MODE=true in .env
"""

import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import Blip2Processor, Trainer

from src.components.dataloader import ImageCaptioningDataset, collate_fn
from src.expection import CustomException
from src.logger import logger
from src.model import load_model
from src.training_args import get_training_args
from src.utils import env_flag, get_env, get_hf_token, resolve_path


def _split_dataset(dataset, seed: int = 42):
    n = len(dataset)
    if n < 3:
        logger.warning("Dataset has %s samples; using the same split for train/val/test.", n)
        return {"train": dataset, "validation": dataset, "test": dataset}

    if n <= 6:
        logger.warning("Small dataset (%s samples); using minimal train/val split.", n)
        split = dataset.train_test_split(test_size=1, seed=seed)
        return {
            "train": split["train"],
            "validation": split["test"],
            "test": split["test"],
        }

    split = dataset.train_test_split(test_size=max(1, int(n * 0.3)), seed=seed)
    if len(split["test"]) < 2:
        return {
            "train": split["train"],
            "validation": split["test"],
            "test": split["test"],
        }

    test_valid = split["test"].train_test_split(test_size=0.5, seed=seed)
    return {
        "train": split["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"],
    }


def _load_vqa_datasets(vqa_path: Path, seed: int = 42):
    if not vqa_path.is_file():
        raise CustomException(
            FileNotFoundError(
                f"Processed VQA dataset not found at {vqa_path}. "
                "Run: python -m src.pipeline.prepare_data"
            ),
            sys,
        )

    dataset = load_dataset("json", data_files=str(vqa_path))["train"]
    logger.info("Loaded %s VQA samples from %s", len(dataset), vqa_path)
    return _split_dataset(dataset, seed=seed)


def _validate_dataloader(dataset_split, processor) -> dict:
    train_dataset = ImageCaptioningDataset(dataset_split["train"], processor)
    sample = train_dataset[0]
    if sample is None:
        raise CustomException(ValueError("Dataloader returned an empty sample."), sys)

    batch = collate_fn([sample])
    if batch is None:
        raise CustomException(ValueError("Collate function returned an empty batch."), sys)

    logger.info(
        "Dataloader OK -> pixel_values=%s, input_ids=%s, labels=%s",
        tuple(batch["pixel_values"].shape),
        tuple(batch["input_ids"].shape),
        tuple(batch["labels"].shape),
    )
    return batch


class TrainingPipeline:
    def __init__(self, vqa_path: Path | None = None, seed: int = 42):
        self.vqa_path = Path(
            vqa_path or resolve_path(get_env("PROCESSED_VQA_PATH", "data/processed/bdd100k_risk_vqa.json"))
        )
        self.seed = int(get_env("RANDOM_SEED", str(seed)))
        self.cpu_test = env_flag("CPU_TEST_MODE")

    def run(self) -> Path:
        if self.cpu_test:
            return self._run_cpu_test()

        if not torch.cuda.is_available():
            raise CustomException(
                RuntimeError(
                    "No CUDA GPU detected. BLIP-2 QLoRA training requires a GPU.\n"
                    "Set CPU_TEST_MODE=true in .env to run a CPU smoke test,\n"
                    "or use Google Colab / a cloud GPU for full training."
                ),
                sys,
            )

        return self._run_training(cpu_test=False)

    def _run_cpu_test(self) -> Path:
        try:
            logger.info("CPU test mode enabled (1 training step, no GPU).")
            logger.info("Step 1/5: Load dataset")
            dataset = _load_vqa_datasets(self.vqa_path, seed=self.seed)

            logger.info("Step 2/5: Load processor")
            processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                token=get_hf_token(),
            )

            logger.info("Step 3/5: Validate dataloader")
            _validate_dataloader(dataset, processor)

            if not env_flag("CPU_TEST_TRAIN_STEP", default=True):
                logger.info("CPU data test passed. Set CPU_TEST_TRAIN_STEP=true to run a train step.")
                return resolve_path(get_env("CPU_TEST_OUTPUT_DIR", "artifacts/cpu-test-run"))

            logger.info("Step 4/5: Load model on CPU (this may take several minutes)")
            model, processor = load_model(cpu_test=True)

            logger.info("Step 5/5: Run 1 training step")
            return self._run_training(cpu_test=True, dataset=dataset, model=model, processor=processor)

        except CustomException:
            raise
        except Exception as error:
            raise CustomException(error, sys) from error

    def _run_training(
        self,
        cpu_test: bool,
        dataset: dict | None = None,
        model=None,
        processor=None,
    ) -> Path:
        try:
            if dataset is None:
                logger.info("Step 1/4: Load dataset")
                dataset = _load_vqa_datasets(self.vqa_path, seed=self.seed)
                logger.info(
                    "Splits -> train: %s, validation: %s, test: %s",
                    len(dataset["train"]),
                    len(dataset["validation"]),
                    len(dataset["test"]),
                )

            if model is None or processor is None:
                logger.info("Step 2/4: Load model and processor")
                model, processor = load_model(cpu_test=cpu_test)

            logger.info("Step 3/4: Build Trainer")
            training_args = get_training_args(cpu_test=cpu_test)
            train_dataset = ImageCaptioningDataset(dataset["train"], processor)
            val_dataset = ImageCaptioningDataset(dataset["validation"], processor)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset if not cpu_test else None,
                data_collator=collate_fn,
            )

            logger.info("Step 4/4: Train")
            trainer.train()
            if not cpu_test:
                trainer.evaluate()

            output_dir = Path(training_args.output_dir)
            trainer.save_model(str(output_dir))
            processor.save_pretrained(str(output_dir))
            logger.info("Training complete. Model saved to %s", output_dir)
            return output_dir

        except CustomException:
            raise
        except Exception as error:
            raise CustomException(error, sys) from error


def train(vqa_path: Path | None = None) -> Path:
    return TrainingPipeline(vqa_path=vqa_path).run()


if __name__ == "__main__":
    path = train()
    print(f"Model saved to: {path}")
