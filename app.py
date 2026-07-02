from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils import env_flag, get_env


class PredictResponse(BaseModel):
    question: str
    answer: str
    decision: str | None


@lru_cache(maxsize=1)
def get_pipeline() -> PredictionPipeline:
    adapter_path = get_env("EVAL_MODEL_DIR", "artifacts/eval-models/blip2-finetuned-bdd100k-vqa")
    cpu_test = env_flag("CPU_TEST_MODE", default=False)
    max_new_tokens = int(get_env("EVAL_MAX_NEW_TOKENS", "64"))
    return PredictionPipeline(
        adapter_path=Path(adapter_path),
        cpu_test=cpu_test,
        max_new_tokens=max_new_tokens,
    )


def predict_ui(image: Image.Image | None, question: str) -> str:
    if image is None:
        return "Please upload an image."
    if not question or not question.strip():
        return "Please enter a question."

    result = get_pipeline().predict_from_image(image, question.strip())
    if result["decision"]:
        return f"**Decision:** {result['decision']}\n\n**Answer:** {result['answer']}"
    return result["answer"]


def build_gradio_ui() -> gr.Blocks:
    with gr.Blocks(title="BLIP-2 VQA") as demo:
        gr.Markdown(
            "# BLIP-2 Driving Risk VQA\n"
            "Upload a driving scene image and ask a safety question."
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Image")
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Is it safe to proceed?",
                    lines=2,
                )
                submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column():
                answer_output = gr.Markdown(label="Answer")

        gr.Examples(
            examples=[
                [
                    "data/dummy_bdd100k/bdd100k/images/100k/val/dummy_001_green_clear.jpg",
                    "Is it safe to proceed?",
                ],
                [
                    "data/dummy_bdd100k/bdd100k/images/100k/val/dummy_002_red_close_car.jpg",
                    "Does the car need to stop?",
                ],
            ],
            inputs=[image_input, question_input],
        )

        submit_btn.click(
            fn=predict_ui,
            inputs=[image_input, question_input],
            outputs=answer_output,
        )
        question_input.submit(
            fn=predict_ui,
            inputs=[image_input, question_input],
            outputs=answer_output,
        )

    return demo


app = FastAPI(title="BLIP-2 VQA Inference API", version="1.0.0")
app = gr.mount_gradio_app(app, build_gradio_ui(), path="/")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    question: str = Form(...),
) -> PredictResponse:
    suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await image.read())
            tmp_path = tmp.name

        result = get_pipeline().predict(tmp_path, question)
        return PredictResponse(**result)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
