#!/usr/bin/env python3
"""Classify a single r/nba post with the fine-tuned TakeMeter model."""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
LABELS = ["analysis", "hot_take", "reaction"]


def predict(text: str) -> tuple[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax())
        return LABELS[idx], float(probs[idx])


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py \"Your post text here\"")
        sys.exit(1)
    text = " ".join(sys.argv[1:])
    label, conf = predict(text)
    print(f"Label:      {label}")
    print(f"Confidence: {conf:.1%}")


if __name__ == "__main__":
    main()
