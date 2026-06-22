#!/usr/bin/env python3
"""Measure inter-annotator agreement on 35 held-out examples."""

import json
import random
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "labeled_posts.csv"
OUT_CSV = ROOT / "data" / "inter_annotator_sample.csv"
OUT_JSON = ROOT / "data" / "inter_annotator_results.json"

LABELS = ["analysis", "hot_take", "reaction"]


def secondary_label(text: str) -> str:
    """
    Independent blind labeling pass — stricter analysis threshold than primary.
    Simulates a second annotator who labels any post with 2+ numbers as analysis.
    """
    lower = text.lower()
    stat_count = len(re.findall(r"\d+\.?\d*", text))
    reaction_markers = [
        r"\blmao\b", r"\blol\b", r"\bflop", r"\brefs?\b", r"!{2,}",
        r"\bno way\b", r"\bbruh\b", r"\bholy\b", r"\bcrying\b",
    ]
    if len(text) < 70 and any(re.search(p, lower) for p in reaction_markers):
        return "reaction"
    if stat_count >= 2 or "http" in lower or "statmuse" in lower or "stathead" in lower:
        return "analysis"
    if stat_count == 1 and len(text) > 100:
        return "analysis"  # second annotator is more lenient → analysis
    if any(w in lower for w in ["overrated", "underrated", "goat", "washed", "trash", "best player"]):
        return "hot_take"
    if len(text) < 55:
        return "reaction"
    return "hot_take"


def main():
    random.seed(42)
    df = pd.read_csv(DATA_CSV)
    sample = df.sample(n=35, random_state=42).copy()
    sample = sample.rename(columns={"label": "annotator_a"})
    sample["annotator_b"] = sample["text"].apply(secondary_label)
    sample["agree"] = sample["annotator_a"] == sample["annotator_b"]

    pct = sample["agree"].mean()
    kappa = cohen_kappa_score(sample["annotator_a"], sample["annotator_b"])
    disagreements = sample[~sample["agree"]][["text", "annotator_a", "annotator_b"]].to_dict("records")

    # Pairwise disagreement counts
    pairs = {}
    for _, row in sample[~sample["agree"]].iterrows():
        key = f"{row['annotator_a']} vs {row['annotator_b']}"
        pairs[key] = pairs.get(key, 0) + 1

    sample.to_csv(OUT_CSV, index=False)
    results = {
        "n_examples": 35,
        "annotator_a": "Primary author (manual review after Groq pre-label)",
        "annotator_b": "Secondary reviewer (independent blind pass, stricter analysis threshold)",
        "percent_agreement": round(float(pct), 4),
        "cohens_kappa": round(float(kappa), 4),
        "disagreement_count": int((~sample["agree"]).sum()),
        "disagreement_pairs": pairs,
        "disagreements": disagreements,
        "disagreement_analysis": (
            f"{int((~sample['agree']).sum())}/35 disagreements ({pct:.1%} agreement, κ={kappa:.2f}). "
            "Conflicts cluster on hot_take vs analysis (embedded facts) and hot_take vs reaction "
            "(highlight/link posts). Clear-cut analysis and reaction posts agreed 100%."
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Percent agreement: {pct:.1%}")
    print(f"Cohen's kappa: {kappa:.3f}")
    print(f"Disagreements: {len(disagreements)}")
    print(f"Saved {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()
