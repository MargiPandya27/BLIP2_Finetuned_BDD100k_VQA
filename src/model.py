import json
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import BitsAndBytesConfig, Blip2ForConditionalGeneration, Blip2Processor

from src.logger import logger
from src.utils import PROJECT_ROOT, get_env, get_hf_token, resolve_path


def _resolve_dtype(dtype_name: str):
    if isinstance(dtype_name, str):
        return getattr(torch, dtype_name)
    return dtype_name


def load_model(cpu_test: bool = False):
    config_path = PROJECT_ROOT / "config" / "qlora_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    processor = Blip2Processor.from_pretrained(
        model_config["processor_config"]["name"],
        token=get_hf_token(),
    )

    if cpu_test:
        logger.warning("CPU test mode: loading model without 4-bit quantization on CPU.")
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_config["model_config"]["model_name"],
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=model_config["model_config"]["trust_remote_code"],
            low_cpu_mem_usage=True,
            token=get_hf_token(),
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_config["bnb_config"]["load_in_4bit"],
            bnb_4bit_use_double_quant=model_config["bnb_config"]["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=model_config["bnb_config"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=_resolve_dtype(model_config["bnb_config"]["bnb_4bit_compute_dtype"]),
            llm_int8_enable_fp32_cpu_offload=model_config["bnb_config"]["llm_int8_enable_fp32_cpu_offload"],
        )
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_config["model_config"]["model_name"],
            device_map=model_config["model_config"]["device_map"],
            quantization_config=bnb_config,
            trust_remote_code=model_config["model_config"]["trust_remote_code"],
            low_cpu_mem_usage=model_config["model_config"]["low_cpu_mem_usage"],
            token=get_hf_token(),
        )

    lora_config = LoraConfig(
        r=model_config["lora_config"]["r"],
        lora_alpha=model_config["lora_config"]["lora_alpha"],
        lora_dropout=model_config["lora_config"]["lora_dropout"],
        bias=model_config["lora_config"]["bias"],
        target_modules=model_config["qformer_target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def load_eval_model(cpu_test: bool = False, adapter_path: Path | None = None):
    """Load the published LoRA adapter for inference / evaluation."""
    adapter_path = Path(
        adapter_path
        or resolve_path(get_env("EVAL_MODEL_DIR", "artifacts/eval-models/blip2-finetuned-bdd100k-vqa"))
    )
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(
            f"Evaluation adapter not found at {adapter_path}. "
            "Run: python -m src.pipeline.download_eval_model"
        )

    with open(adapter_config_path, encoding="utf-8") as f:
        base_model_id = json.load(f)["base_model_name_or_path"]

    processor = Blip2Processor.from_pretrained(
        base_model_id,
        token=get_hf_token(),
    )

    use_cpu = cpu_test or not torch.cuda.is_available()
    if use_cpu:
        logger.warning("Loading evaluation model on CPU (this may be slow).")
        base_model = Blip2ForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            token=get_hf_token(),
        )
    else:
        with open(PROJECT_ROOT / "config" / "qlora_config.yaml", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_config["bnb_config"]["load_in_4bit"],
            bnb_4bit_use_double_quant=model_config["bnb_config"]["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=model_config["bnb_config"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=_resolve_dtype(model_config["bnb_config"]["bnb_4bit_compute_dtype"]),
            llm_int8_enable_fp32_cpu_offload=model_config["bnb_config"]["llm_int8_enable_fp32_cpu_offload"],
        )
        base_model = Blip2ForConditionalGeneration.from_pretrained(
            base_model_id,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            token=get_hf_token(),
        )

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    return model, processor


if __name__ == "__main__":
    model, processor = load_model()
    print("Model and processor loaded successfully.")
