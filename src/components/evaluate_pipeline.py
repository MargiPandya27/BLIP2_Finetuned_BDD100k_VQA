import json
from collections import defaultdict
from pathlib import Path

import evaluate as hf_evaluate
import torch
from PIL import Image

from src.logger import logger

DECISION_LABELS = ("Yes", "No", "Slow Down")


def extract_decision(text: str) -> str | None:
    """Map an answer string to Yes / No / Slow Down."""
    normalized = text.strip()
    lower = normalized.lower()
    if lower.startswith("slow down"):
        return "Slow Down"
    if lower.startswith("yes"):
        return "Yes"
    if lower.startswith("no"):
        return "No"
    return None


def _model_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _strip_generated_answer(text: str) -> str:
    if "Answer:" in text:
        return text.split("Answer:", 1)[-1].strip()
    return text.strip()


def generate_answer(
    model,
    processor,
    image_path: str,
    question: str,
    max_new_tokens: int = 64,
) -> str:
    image = Image.open(image_path).convert("RGB")
    prompt = f"Question: {question} Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    device = _model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated = processor.decode(output_ids[0], skip_special_tokens=True)
    return _strip_generated_answer(generated)


def compute_metrics(predictions: list[dict]) -> dict:
    """Compute decision accuracy, per-question accuracy, and ROUGE scores."""
    valid = [row for row in predictions if row["reference_decision"] is not None]
    if not valid:
        raise ValueError("No valid reference decisions found in evaluation set.")

    decision_matches = [
        row["reference_decision"] == row["predicted_decision"]
        for row in valid
        if row["predicted_decision"] is not None
    ]
    decision_accuracy = sum(decision_matches) / len(valid)

    per_question: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "accuracy": 0.0}
    )
    for row in valid:
        bucket = per_question[row["question"]]
        bucket["total"] += 1
        if row["reference_decision"] == row["predicted_decision"]:
            bucket["correct"] += 1

    for bucket in per_question.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"]

    rouge = hf_evaluate.load("rouge")
    rouge_scores = rouge.compute(
        predictions=[row["prediction"] for row in valid],
        references=[row["reference"] for row in valid],
        use_stemmer=True,
    )

    metrics = {
        "num_samples": len(predictions),
        "num_valid_samples": len(valid),
        "decision_accuracy": decision_accuracy,
        "decision_correct": sum(decision_matches),
        "per_question_accuracy": dict(per_question),
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "rougeLsum": rouge_scores["rougeLsum"],
    }
    return metrics


def evaluate_dataset(
    model,
    processor,
    dataset,
    max_samples: int | None = None,
    max_new_tokens: int = 64,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    limit = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for idx in range(limit):
        item = dataset[idx]
        image_path = item["image"]
        question = item["question"]
        reference = item["answer"]

        prediction = generate_answer(
            model=model,
            processor=processor,
            image_path=image_path,
            question=question,
            max_new_tokens=max_new_tokens,
        )

        row = {
            "image": image_path,
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "reference_decision": extract_decision(reference),
            "predicted_decision": extract_decision(prediction),
            "decision_match": extract_decision(reference) == extract_decision(prediction),
        }
        rows.append(row)
        logger.info(
            "Eval %s/%s | ref=%s | pred=%s",
            idx + 1,
            limit,
            row["reference_decision"],
            row["predicted_decision"],
        )

    metrics = compute_metrics(rows)
    return rows, metrics


def save_eval_results(
    predictions: list[dict],
    metrics: dict,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    metrics_path = output_dir / "metrics.json"

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    serializable_metrics = json.loads(
        json.dumps(
            metrics,
            default=lambda value: float(value) if hasattr(value, "item") else value,
        )
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=2)

    return predictions_path, metrics_path
