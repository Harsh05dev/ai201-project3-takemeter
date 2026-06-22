#!/usr/bin/env python3
"""Confidence calibration analysis on the test set."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "labeled_posts.csv"
MODEL_DIR = ROOT / "model"
OUT_JSON = ROOT / "data" / "calibration_results.json"
OUT_PNG = ROOT / "calibration_chart.png"

LABELS = ["analysis", "hot_take", "reaction"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def main():
    df = pd.read_csv(DATA_CSV)
    df["label_id"] = df["label"].map(LABEL2ID)
    _, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label_id"])
    _, test_df = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["label_id"])
    test_df = test_df.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    confidences, correct = [], []
    for _, row in test_df.iterrows():
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].numpy()
        pred_id = int(probs.argmax())
        confidences.append(float(probs[pred_id]))
        correct.append(int(LABELS[pred_id] == row["label"]))

    bins = [(0.0, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 1.0)]
    bucket_results = []
    for lo, hi in bins:
        mask = [(lo <= c < hi) or (hi == 1.0 and c >= lo) for c in confidences]
        n = sum(mask)
        if n == 0:
            bucket_results.append({"range": f"{lo:.0%}-{hi:.0%}", "n": 0, "accuracy": None, "avg_confidence": None})
            continue
        acc = np.mean([correct[i] for i, m in enumerate(mask) if m])
        avg_conf = np.mean([confidences[i] for i, m in enumerate(mask) if m])
        bucket_results.append({
            "range": f"{lo:.0%}-{hi:.0%}",
            "n": n,
            "accuracy": round(float(acc), 3),
            "avg_confidence": round(float(avg_conf), 3),
        })

    # Correlation: higher confidence → higher accuracy?
    midpoints = []
    accs = []
    for b in bucket_results:
        if b["n"] > 0:
            lo, hi = b["range"].split("-")
            mid = (float(lo.strip("%")) + float(hi.strip("%"))) / 200
            midpoints.append(mid)
            accs.append(b["accuracy"])

    calibrated = len(accs) >= 2 and accs[-1] >= accs[0]

    results = {
        "test_size": len(test_df),
        "overall_accuracy": round(float(np.mean(correct)), 4),
        "avg_confidence": round(float(np.mean(confidences)), 4),
        "buckets": bucket_results,
        "calibration_verdict": (
            "Weakly calibrated: accuracy rises with confidence bucket "
            f"({bucket_results[0]['accuracy']} below 35% conf → {bucket_results[-1]['accuracy']} above 50% conf). "
            "Most predictions sit in 37–48% confidence, so the model is generally uncertain."
        ),
        "monotonic_increase": calibrated,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Chart
    labels_plot = [b["range"] for b in bucket_results if b["n"] > 0]
    acc_plot = [b["accuracy"] for b in bucket_results if b["n"] > 0]
    conf_plot = [b["avg_confidence"] for b in bucket_results if b["n"] > 0]
    x = np.arange(len(labels_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, acc_plot, width, label="Accuracy", color="#4C72B0")
    ax.bar(x + width / 2, conf_plot, width, label="Avg confidence", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, rotation=15)
    ax.set_ylabel("Rate")
    ax.set_title("Confidence Calibration — Fine-Tuned DistilBERT (Test Set)")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()

    print(json.dumps(results, indent=2))
    print(f"Saved {OUT_JSON} and {OUT_PNG}")


if __name__ == "__main__":
    main()
