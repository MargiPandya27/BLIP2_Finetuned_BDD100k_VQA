from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.model import load_eval_model


def _extract_decision(text: str) -> str | None:
    lower = text.strip().lower()
    if lower.startswith("slow down"):
        return "Slow Down"
    if lower.startswith("yes"):
        return "Yes"
    if lower.startswith("no"):
        return "No"
    return None


def _strip_generated_answer(text: str) -> str:
    if "Answer:" in text:
        return text.split("Answer:", 1)[-1].strip()
    return text.strip()


class PredictionPipeline:
    """BLIP-2 VQA inference pipeline for FastAPI serving."""

    def __init__(
        self,
        adapter_path: str | Path | None = None,
        cpu_test: bool = False,
        max_new_tokens: int = 64,
    ) -> None:
        self.model, self.processor = load_eval_model(
            cpu_test=cpu_test,
            adapter_path=Path(adapter_path) if adapter_path else None,
        )
        self.max_new_tokens = max_new_tokens

    def _model_device(self) -> torch.device:
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    def predict_from_image(self, image: Image.Image, question: str) -> dict:
        image = image.convert("RGB")
        prompt = f"Question: {question} Answer:"

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        device = self._model_device()
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        raw_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        answer = _strip_generated_answer(raw_text)

        return {
            "question": question,
            "answer": answer,
            "decision": _extract_decision(answer),
        }

    def predict(self, image_path: str | Path, question: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        return self.predict_from_image(image, question)
