#!/usr/bin/env python3
"""Fine-tune DistilBERT, run Groq baseline, export evaluation artifacts."""

import json
import os
import random
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "labeled_posts.csv"
OUT_JSON = ROOT / "evaluation_results.json"
OUT_CM = ROOT / "confusion_matrix.png"

MODEL_NAME = "distilbert-base-uncased"
LABELS = ["analysis", "hot_take", "reaction"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

BASELINE_PROMPT = """You are classifying r/nba posts by discourse type.

Labels (pick exactly ONE):
- analysis: A structured argument backed by specific, verifiable evidence (stats, film, historical comparisons, tactical breakdowns). The post reasons toward a conclusion.
- hot_take: A bold, confident opinion without genuine supporting evidence. May cite one cherry-picked stat for effect but asserts rather than argues.
- reaction: An immediate emotional response to a specific event. Little to no argument — expressing a feeling in the moment.

Edge rule: If a post cites only ONE stat to support a bold claim without building a multi-point argument, label hot_take.

Post: "{text}"

Respond with ONLY one word: analysis, hot_take, or reaction."""


def parse_label(raw: str) -> str | None:
    raw = raw.strip().lower().replace(" ", "_").replace("-", "_")
    for label in LABELS:
        if label in raw or raw == label:
            return label
    return None


def load_data():
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip()
    df = df[df["label"].isin(LABELS)]
    return df


def split_data(df, seed=42):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
    )
    return train_df, val_df, test_df


def to_hf_dataset(df):
    return Dataset.from_dict({
        "text": df["text"].tolist(),
        "label": [LABEL2ID[l] for l in df["label"]],
    })


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def load_groq_env():
    paths = [
        ROOT / ".env",
        ROOT.parent.parent / "Week 2" / "ai201-project2-fitfindr-starter" / ".env",
        ROOT.parent.parent / ".env",
    ]
    for p in paths:
        if p.exists():
            load_dotenv(p, override=True)
    return os.getenv("GROQ_API_KEY")


def run_baseline(test_df: pd.DataFrame) -> tuple[list[str], list[float]]:
    api_key = load_groq_env()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY required for baseline")

    client = Groq(api_key=api_key)
    preds, confidences = [], []

    for text in test_df["text"]:
        prompt = BASELINE_PROMPT.format(text=text[:800])
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            raw = resp.choices[0].message.content
            label = parse_label(raw) or "reaction"
        except Exception:
            label = "reaction"
        preds.append(label)
        confidences.append(0.5)  # zero-shot has no calibrated confidence
        time.sleep(0.3)

    return preds, confidences


def train_model(train_df, val_df, test_df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = to_hf_dataset(train_df)
    val_ds = to_hf_dataset(val_df)
    test_ds = to_hf_dataset(test_df)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=str(ROOT / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        logging_steps=20,
        report_to="none",
        use_cpu=not (use_mps or use_cuda),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning DistilBERT (3 epochs, lr=2e-5, batch=16)...")
    trainer.train()

    # Evaluate on test
    predictions = trainer.predict(test_ds)
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    true_ids = test_df["label"].map(LABEL2ID).tolist()

    # Softmax confidence for sample classifications
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    confidences = probs.max(axis=1)

    finetuned_preds = [ID2LABEL[i] for i in pred_ids]
    model_path = ROOT / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    return finetuned_preds, confidences.tolist(), true_ids, model, tokenizer


def plot_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return cm.tolist()


def build_report(y_true, y_pred, name):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
    }


def find_misclassified(test_df, y_true, y_pred, confidences, n=5):
    errors = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            errors.append({
                "text": test_df.iloc[i]["text"],
                "true_label": true,
                "predicted_label": pred,
                "confidence": round(float(confidences[i]), 3),
            })
    return errors[:n]


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    df = load_data()
    print(f"Loaded {len(df)} examples")
    print("Label distribution:\n", df["label"].value_counts())

    train_df, val_df, test_df = split_data(df)
    print(f"Split — train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")

    # Baseline
    print("\nRunning Groq zero-shot baseline on test set...")
    baseline_preds, _ = run_baseline(test_df)
    y_true = test_df["label"].tolist()

    # Fine-tune
    finetuned_preds, confidences, _, model, tokenizer = train_model(train_df, val_df, test_df)

    baseline_report = build_report(y_true, baseline_preds, "baseline")
    finetuned_report = build_report(y_true, finetuned_preds, "finetuned")
    cm = plot_confusion_matrix(y_true, finetuned_preds, "Fine-tuned DistilBERT Confusion Matrix", OUT_CM)

    misclassified = find_misclassified(test_df, y_true, finetuned_preds, confidences, n=8)

    # Sample classifications (mix of correct and wrong)
    samples = []
    for i in range(len(test_df)):
        samples.append({
            "text": test_df.iloc[i]["text"][:200],
            "true_label": y_true[i],
            "predicted_label": finetuned_preds[i],
            "confidence": round(float(confidences[i]), 3),
            "correct": y_true[i] == finetuned_preds[i],
        })
    random.shuffle(samples)
    sample_classifications = samples[:5]

    results = {
        "model": MODEL_NAME,
        "hyperparameters": {"epochs": 3, "learning_rate": 2e-5, "batch_size": 16},
        "test_size": len(test_df),
        "baseline": {
            "model": "llama-3.3-70b-versatile (Groq zero-shot)",
            **baseline_report,
        },
        "finetuned": {
            "model": MODEL_NAME,
            **finetuned_report,
        },
        "confusion_matrix": {"labels": LABELS, "matrix": cm},
        "misclassified_examples": misclassified,
        "sample_classifications": sample_classifications,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== RESULTS ===")
    print(f"Baseline accuracy:  {baseline_report['accuracy']:.3f}")
    print(f"Fine-tuned accuracy: {finetuned_report['accuracy']:.3f}")
    print(f"Saved {OUT_JSON} and {OUT_CM}")


if __name__ == "__main__":
    main()
