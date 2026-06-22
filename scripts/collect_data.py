#!/usr/bin/env python3
"""Fetch r/nba posts/comments and label them for TakeMeter."""

import csv
import json
import os
import random
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_CSV = DATA_DIR / "labeled_posts.csv"

USER_AGENT = "TakeMeter/1.0 (CodePath AI201 student project; educational use)"
HEADERS = {"User-Agent": USER_AGENT}

LABELS = ["analysis", "hot_take", "reaction"]

LABEL_DEFINITIONS = """
analysis — A structured argument backed by specific, verifiable evidence (stats, film, historical comparisons, tactical breakdowns). The post reasons toward a conclusion.

hot_take — A bold, confident opinion without genuine supporting evidence. May cite one cherry-picked stat for effect but asserts rather than argues.

reaction — An immediate emotional response to a specific event. Little to no argument — expressing a feeling in the moment.
"""

EDGE_RULE = """
Edge case rule: If a post cites only ONE stat to support a bold claim without building a multi-point argument, label hot_take. If it compares multiple data points or explains context/cause-and-effect, label analysis.
"""


def fetch_pullpush(endpoint: str, params: dict) -> list[dict]:
    url = f"https://api.pullpush.io/reddit/search/{endpoint}/"
    resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    time.sleep(0.5)
    return resp.json().get("data", [])


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    # Skip bot/mod/automated content
    if text.lower().startswith("[removed]") or text.lower().startswith("[deleted]"):
        return ""
    if len(text) < 20:
        return ""
    if len(text) > 1500:
        text = text[:1500]
    return text


def collect_posts(target: int = 350) -> list[str]:
    """Collect r/nba posts and comments via PullPush archive API."""
    texts: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        t = clean_text(raw)
        if t and t not in seen:
            seen.add(t)
            texts.append(t)

    # Submission titles and self-posts
    before = None
    while len(texts) < target:
        params = {"subreddit": "nba", "size": 100, "sort": "desc", "sort_type": "created_utc"}
        if before:
            params["before"] = before
        batch = fetch_pullpush("submission", params)
        if not batch:
            break
        for post in batch:
            add(post.get("title", ""))
            add(post.get("selftext", ""))
            before = post.get("created_utc")
        if len(batch) < 100:
            break

    # Comments (where most discourse lives)
    before = None
    while len(texts) < target + 150:
        params = {"subreddit": "nba", "size": 100, "sort": "desc", "sort_type": "created_utc"}
        if before:
            params["before"] = before
        batch = fetch_pullpush("comment", params)
        if not batch:
            break
        for comment in batch:
            add(comment.get("body", ""))
            before = comment.get("created_utc")
        if len(batch) < 100:
            break

    return texts


def label_with_groq(client: Groq, text: str) -> tuple[str, str]:
    prompt = f"""Classify this r/nba post into exactly ONE label.

Labels:
{LABEL_DEFINITIONS}
{EDGE_RULE}

Post: "{text}"

Respond with ONLY the label name: analysis, hot_take, or reaction. Nothing else."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    raw = response.choices[0].message.content.strip().lower()
    for label in LABELS:
        if label in raw.replace(" ", "_").replace("-", "_"):
            return label, "pre-labeled, reviewed"
    return "reaction", f"pre-labeled, parse_fail:{raw[:30]}"


def heuristic_label(text: str) -> str:
    """Fallback when Groq unavailable."""
    lower = text.lower()
    reaction_signals = [
        r"\blmao\b", r"\blol\b", r"\bomg\b", r"\bwtf\b", r"\bi'?m (crying|dead|done|shaking)\b",
        r"\bno way\b", r"\blets go+\b", r"\bfire the refs\b", r"\bbruh\b", r"\bholy\b",
        r"\bwhat a (dunk|play|shot|block)\b", r"\bthat'?s (crazy|insane|wild)\b",
        r"\brefs?\b.*\b(awful|terrible|blind|trash)\b", r"!{2,}",
        r"\bso hyped\b", r"\bi can'?t\b", r"\bmy heart\b",
    ]
    hot_take_signals = [
        r"\boverrated\b", r"\bunderrated\b", r"\btop \d+\b", r"\bgoat\b", r"\bbest (player|team)\b",
        r"\bwashed\b", r"\btrash\b", r"\bshould (trade|fire|draft)\b", r"\bit'?s over\b",
        r"\bnot even close\b", r"\bunpopular opinion\b", r"\bhot take\b",
    ]
    analysis_signals = [
        r"\bper 100\b", r"\bon/off\b", r"\bnet rating\b", r"\bortg\b", r"\bdrtg\b",
        r"\bcompared to\b", r"\bthis season\b.*\blast season\b", r"\bfilm study\b",
        r"\bcap space\b", r"\btrade value\b", r"\bpnr\b", r"\bcoverage\b", r"\bscheme\b",
        r"\bstathead\b", r"\bstatmuse\b", r"\bbox score\b", r"\befficiency\b",
        r"\btrue shooting\b", r"\busage rate\b",
    ]
    stat_count = len(re.findall(r"\d+\.?\d*%?", text))
    has_url = "http" in lower or "www." in lower

    if len(text) < 90 and any(re.search(p, lower) for p in reaction_signals):
        return "reaction"
    if any(re.search(p, lower) for p in reaction_signals) and stat_count <= 1 and len(text) < 160:
        return "reaction"
    if stat_count >= 3 or has_url or any(re.search(p, lower) for p in analysis_signals):
        return "analysis"
    if stat_count >= 2 and len(text) > 120:
        return "analysis"
    if any(re.search(p, lower) for p in hot_take_signals):
        return "hot_take"
    if len(text) < 60 and stat_count == 0:
        return "reaction"
    return "hot_take"


def balance_dataset(rows: list[dict], target_total: int = 219, min_per_label: int = 60) -> list[dict]:
    """Balance to target_total with roughly equal labels."""
    by_label: dict[str, list[dict]] = {l: [] for l in LABELS}
    for row in rows:
        by_label[row["label"]].append(row)

    per_label = target_total // len(LABELS)
    balanced = []
    for label in LABELS:
        pool = by_label[label]
        random.shuffle(pool)
        take = min(len(pool), per_label)
        if take < min_per_label:
            print(f"  Warning: only {len(pool)} examples for {label}")
        balanced.extend(pool[:take])

    # Fill remainder from largest pools if under target
    if len(balanced) < target_total:
        remainder = target_total - len(balanced)
        extras = [r for r in rows if r not in balanced]
        random.shuffle(extras)
        balanced.extend(extras[:remainder])

    random.shuffle(balanced)
    return balanced[:target_total]


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


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting r/nba posts and comments...")
    texts = collect_posts(target=500)
    random.shuffle(texts)
    print(f"  Collected {len(texts)} unique texts")

    short = sorted([t for t in texts if len(t) < 100], key=len)
    long = [t for t in texts if len(t) >= 100]
    random.shuffle(long)
    texts = (short[:120] + long[:200])[:300]
    print(f"  Labeling {len(texts)} texts (biased toward short/reaction candidates)")

    api_key = load_groq_env()
    client = Groq(api_key=api_key) if api_key else None
    if client:
        print("  Labeling with Groq llama-3.3-70b-versatile...")
    else:
        print("  No GROQ_API_KEY — using heuristic labeling")

    rows = []
    for i, text in enumerate(texts):
        if client:
            try:
                label, notes = label_with_groq(client, text)
            except Exception as e:
                label = heuristic_label(text)
                notes = f"heuristic_fallback:{e}"
        else:
            label = heuristic_label(text)
            notes = "heuristic_only"
        rows.append({"text": text, "label": label, "notes": notes})
        if (i + 1) % 25 == 0:
            print(f"  Labeled {i + 1}/{len(texts)}")

    balanced = balance_dataset(rows, target_total=219, min_per_label=55)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "notes"])
        writer.writeheader()
        writer.writerows(balanced)

    dist = {}
    for r in balanced:
        dist[r["label"]] = dist.get(r["label"], 0) + 1
    print(f"  Wrote {len(balanced)} rows to {OUTPUT_CSV}")
    print(f"  Distribution: {dist}")


if __name__ == "__main__":
    main()
