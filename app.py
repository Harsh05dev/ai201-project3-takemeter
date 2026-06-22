#!/usr/bin/env python3
"""TakeMeter web interface — classify r/nba posts with label + confidence."""

from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
LABELS = ["analysis", "hot_take", "reaction"]
LABEL_COLORS = {"analysis": "#4C72B0", "hot_take": "#DD8452", "reaction": "#55A868"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def classify(text: str) -> tuple[str, dict]:
    if not text or not text.strip():
        return "Enter a post to classify.", {}
    inputs = tokenizer(text.strip(), return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].numpy()
    pred_id = int(probs.argmax())
    label = LABELS[pred_id]
    confidence = float(probs[pred_id])
    scores = {f"{l}": float(probs[i]) for i, l in enumerate(LABELS)}
    summary = (
        f"**{label}** ({confidence:.1%} confidence)\n\n"
        f"- **analysis** — evidence-backed argument\n"
        f"- **hot_take** — bold opinion, weak evidence\n"
        f"- **reaction** — emotional in-the-moment response"
    )
    return summary, scores


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        label="r/nba post or comment",
        placeholder="Paste a Reddit post here…",
        lines=4,
    ),
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Label(label="Scores", num_top_classes=3),
    ],
    title="TakeMeter — r/nba Discourse Classifier",
    description=(
        "Fine-tuned DistilBERT classifier. "
        "Labels posts as **analysis**, **hot_take**, or **reaction**."
    ),
    examples=[
        ["Jokic's on/off net rating is +18.4 — Denver's offense drops from 118 ORtg to 102 when he sits."],
        ["Luka is already a top-5 player all-time and it's not close."],
        ["SGA flops on the elbow to the face"],
        ["NO WAY THAT WAS A FLAGRANT LMAOOO"],
        ["That was intentional. The Thunder were $30M under the cap, spent it all on Hartenstein..."],
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
