import os
from pathlib import Path

from dotenv import load_dotenv

SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
ENV_FILE = PROJECT_ROOT / ".env"


def init_env() -> None:
    load_dotenv(ENV_FILE)


def get_env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def env_flag(key: str, default: bool = False) -> bool:
    value = get_env(key)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


def resolve_path(*parts: str) -> Path:
    path = Path(parts[0])
    if path.is_absolute():
        return path.joinpath(*parts[1:]) if len(parts) > 1 else path
    return PROJECT_ROOT.joinpath(*parts)


def get_hf_token():
    """Use HF_TOKEN from .env when set; otherwise disable stale cached tokens."""
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    token = get_env("HF_TOKEN")
    return token if token else False


def evaluate(
    vqa_path: Path | None = None,
    adapter_path: Path | None = None,
    output_dir: Path | None = None,
    cpu_test: bool | None = None,
    max_samples: int | None = None,
) -> Path:
    """
    Evaluate the fine-tuned BLIP-2 VQA adapter on the held-out test split.

    Loads the adapter from EVAL_MODEL_DIR, runs generation on the test set,
    and writes predictions + metrics under EVAL_OUTPUT_DIR.
    """
    import sys

    from src.components.evaluate import evaluate_dataset, save_eval_results
    from src.expection import CustomException
    from src.logger import logger
    from src.model import load_eval_model
    from src.pipeline.training_pipeline import _load_vqa_datasets

    try:
        resolved_vqa_path = Path(
            vqa_path
            or resolve_path(get_env("PROCESSED_VQA_PATH", "data/processed/bdd100k_risk_vqa.json"))
        )
        resolved_output_dir = Path(
            output_dir or resolve_path(get_env("EVAL_OUTPUT_DIR", "artifacts/eval-results"))
        )
        use_cpu = env_flag("CPU_TEST_MODE") if cpu_test is None else cpu_test
        seed = int(get_env("RANDOM_SEED", "42"))
        sample_limit = max_samples
        if sample_limit is None:
            raw_limit = get_env("EVAL_MAX_SAMPLES")
            sample_limit = int(raw_limit) if raw_limit else None

        logger.info("Loading VQA dataset from %s", resolved_vqa_path)
        splits = _load_vqa_datasets(resolved_vqa_path, seed=seed)
        test_dataset = splits["test"]
        logger.info("Test split size: %s", len(test_dataset))

        logger.info("Loading evaluation model")
        model, processor = load_eval_model(cpu_test=use_cpu, adapter_path=adapter_path)

        logger.info("Running evaluation")
        predictions, metrics = evaluate_dataset(
            model=model,
            processor=processor,
            dataset=test_dataset,
            max_samples=sample_limit,
            max_new_tokens=int(get_env("EVAL_MAX_NEW_TOKENS", "64")),
        )

        min_accuracy = float(get_env("EVAL_MIN_DECISION_ACCURACY", "0.8"))
        metrics["promotion_gate"] = {
            "threshold": min_accuracy,
            "passed": metrics["decision_accuracy"] >= min_accuracy,
        }

        predictions_path, metrics_path = save_eval_results(
            predictions=predictions,
            metrics=metrics,
            output_dir=resolved_output_dir,
        )

        logger.info("Decision accuracy: %.2f%%", metrics["decision_accuracy"] * 100)
        logger.info("ROUGE-L: %s", metrics["rougeL"])
        logger.info("Predictions saved to %s", predictions_path)
        logger.info("Metrics saved to %s", metrics_path)

        if not metrics["promotion_gate"]["passed"]:
            logger.warning(
                "Promotion gate failed: decision accuracy %.2f%% < %.2f%%",
                metrics["decision_accuracy"] * 100,
                min_accuracy * 100,
            )

        return resolved_output_dir

    except CustomException:
        raise
    except Exception as error:
        raise CustomException(error, sys) from error


init_env()
