"""
Download the published BLIP-2 VQA model from HuggingFace for local evaluation.

The Hub repo is a PEFT/LoRA adapter (~few hundred MB). At inference time,
transformers pulls the base weights from Salesforce/blip2-opt-2.7b (~6 GB).

Run: python -m src.pipeline.download_eval_model
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from src.expection import CustomException
from src.logger import logger
from src.utils import env_flag, get_env, resolve_path

DEFAULT_MODEL_ID = "MargiPandya/blip2-finetuned-bdd100k-vqa"
DEFAULT_SAVE_DIR = "artifacts/eval-models/blip2-finetuned-bdd100k-vqa"
BASE_MODEL_ID = "Salesforce/blip2-opt-2.7b"


def _prepare_hf_env() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    hf_home = get_env("HF_HOME")
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(hf_home) / "hub")


def _resolve_hf_token() -> str | None:
    _prepare_hf_env()
    token = get_env("HF_TOKEN")
    if not token:
        return None
    try:
        HfApi(token=token).whoami()
        return token
    except Exception:
        logger.warning("HF_TOKEN is invalid or expired; using anonymous Hub access")
        os.environ.pop("HF_TOKEN", None)
        return None


def _snapshot(repo_id: str, local_dir: Path, token: str | None) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", repo_id, local_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
        local_dir_use_symlinks=False,
    )
    return local_dir


def download_eval_model(
    model_id: str | None = None,
    save_dir: Path | None = None,
    prefetch_base: bool = False,
) -> Path:
    """
    Download the fine-tuned adapter to `save_dir`.

    Set `prefetch_base=True` to also cache the ~6 GB base model under HF_HOME.
    """
    try:
        token = _resolve_hf_token()
        model_id = model_id or get_env("EVAL_MODEL_ID", DEFAULT_MODEL_ID)
        save_dir = Path(save_dir or resolve_path(get_env("EVAL_MODEL_DIR", DEFAULT_SAVE_DIR)))

        _snapshot(model_id, save_dir, token)

        if prefetch_base:
            cache_root = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
            logger.info("Prefetching base model %s into HF cache at %s", BASE_MODEL_ID, cache_root)
            snapshot_download(repo_id=BASE_MODEL_ID, token=token)

        logger.info("Adapter saved to %s", save_dir)
        logger.info(
            "Load for evaluation with:\n"
            "  from transformers import Blip2ForConditionalGeneration, Blip2Processor\n"
            f"  model = Blip2ForConditionalGeneration.from_pretrained(r'{save_dir}', device_map='cpu', torch_dtype='auto')\n"
            "  processor = Blip2Processor.from_pretrained(r'%s')",
            save_dir,
        )
        return save_dir

    except CustomException:
        raise
    except Exception as error:
        raise CustomException(error, sys) from error


if __name__ == "__main__":
    path = download_eval_model(prefetch_base=env_flag("PREFETCH_BASE_MODEL", False))
    print(f"Evaluation adapter ready at: {path}")
