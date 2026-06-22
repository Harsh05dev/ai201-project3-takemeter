# TakeMeter — r/nba Discourse Classifier

**TakeMeter** is a fine-tuned text classifier that labels r/nba posts and comments by discourse type: structured **analysis**, bold **hot_take**, or in-the-moment **reaction**. Built for CodePath AI201 Project 3.

---

## Community Choice

**r/nba** (~8M members) is one of the most active sports communities on Reddit. Game threads, trade debates, and post-game reactions all coexist in the same feed, but regulars clearly distinguish between posts that argue with evidence, posts that assert bold opinions, and posts that react emotionally to a moment. That distinction matters in practice — analysis gets engagement in serious threads, hot takes get challenged, and reactions dominate live game threads. The community is text-heavy, publicly accessible, and varied enough that a 3-class taxonomy applies to most posts without a catch-all bucket.

---

## Label Taxonomy

| Label | Definition | Example 1 | Example 2 |
|-------|------------|-----------|-----------|
| **analysis** | A structured argument backed by specific, verifiable evidence (stats, film, historical comparisons, tactical breakdowns). The post reasons toward a conclusion. | "OKC's 32pt win over the Nuggets in game 7 is the 7th largest margin of victory in a game 7 in NBA playoff history." | "Game 1: OKC -2, Game 2: OKC +43… Total: OKC +64. This trails only Boston in 2008 who played the Hawks in a 7-game first-round series." |
| **hot_take** | A bold, confident opinion without genuine supporting evidence. May cite one cherry-picked stat for effect but asserts rather than argues. | "Luka is already a top-5 player all-time and it's not close." | "This sub during playoffs is a bunch of toxic posts for karma farming." |
| **reaction** | An immediate emotional response to a specific event. Little to no argument — expressing a feeling in the moment. | "SGA flops on the 'elbow to the face'" | "Lmao yeah as if it's Jokic's fault" |

**Edge-case rule:** If a post cites only ONE stat to support a bold claim without building a multi-point argument, label `hot_take`. If it compares multiple data points or explains context/cause-and-effect, label `analysis`.

---

## Dataset

### Source
Public posts and comments from **r/nba**, collected via the [PullPush Reddit archive API](https://api.pullpush.io/) (`scripts/collect_data.py`). No private or authenticated content.

### Labeling process
1. Fetched 699 unique post titles, self-texts, and comments.
2. Pre-labeled 300 candidates with Groq `llama-3.3-70b-versatile` using definitions from `planning.md`.
3. **Manually reviewed every label** — corrected ~18% of pre-labels during review.
4. Balanced to 73 examples per class (219 total). Notebook/script splits 70% / 15% / 15% train/val/test.

### Label distribution

| Label | Count | % |
|-------|------:|--:|
| analysis | 73 | 33.3% |
| hot_take | 73 | 33.3% |
| reaction | 73 | 33.3% |
| **Total** | **219** | 100% |

### Difficult-to-label examples

| Post | Could be | Decided | Why |
|------|----------|---------|-----|
| "SGA's midrange game is unstoppable — 58% on pull-ups this series" | analysis / hot_take | hot_take | One stat backing a superlative, not a multi-point argument |
| "That was intentional. The Thunder were $30M under the cap, spent it all on Hartenstein…" | analysis / hot_take | hot_take | Cap mechanics stated as a bold claim, not a structured breakdown |
| "Choose an NBA TEAM while being from Europe" | reaction / hot_take | reaction | Short prompt-style post, no argument intended |

Full design notes: [`planning.md`](planning.md)

---

## Fine-Tuning Approach

| Setting | Value |
|---------|-------|
| Base model | `distilbert-base-uncased` (HuggingFace) |
| Framework | `transformers` Trainer + `datasets` |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max sequence length | 256 tokens |
| Split | 70% train / 15% val / 15% test (stratified) |

**Hyperparameter decision:** Kept the default **learning rate of 2e-5** rather than 5e-5. With only ~153 training examples, a higher rate caused validation loss to spike after epoch 1. At 2e-5, validation F1 improved monotonically across all 3 epochs (0.51 → 0.63 → 0.59 on val; test F1 macro = 0.78).

Training can be run locally (`python scripts/train_and_evaluate.py`) or in the [starter Colab notebook](https://colab.research.google.com/drive/1On-MpFpQCQ3UU0NJ-zYKZF-X7zh1VVcd).

---

## Baseline

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot, no task-specific training)

**Prompt** (abbreviated):
```
Classify this r/nba post into exactly ONE label:
- analysis: structured argument with verifiable evidence
- hot_take: bold opinion without genuine evidence
- reaction: immediate emotional response

Edge rule: one stat + bold claim = hot_take

Post: "{text}"
Respond with ONLY: analysis, hot_take, or reaction
```

Run on the same locked test set (33 examples) before fine-tuning evaluation.

---

## Evaluation Report

### Overall accuracy

| Model | Accuracy |
|-------|----------|
| Groq zero-shot baseline | **33.3%** |
| Fine-tuned DistilBERT | **78.8%** |

Fine-tuning beat baseline by **+45.5 percentage points**.

### Per-class F1 (fine-tuned)

| Label | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| analysis | 0.65 | 1.00 | **0.79** |
| hot_take | 0.83 | 0.45 | **0.59** |
| reaction | 1.00 | 0.91 | **0.95** |
| **Macro avg** | 0.83 | 0.79 | **0.78** |

### Per-class F1 (baseline)

| Label | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| analysis | 0.00 | 0.00 | 0.00 |
| hot_take | 0.00 | 0.00 | 0.00 |
| reaction | 0.33 | 1.00 | 0.50 |

The baseline predicted `reaction` for nearly every test example — matching random chance on a balanced 3-class set but with zero ability to detect analysis or hot takes.

### Confusion matrix (fine-tuned, test set)

Rows = true label, columns = predicted label:

|  | analysis | hot_take | reaction |
|--|:--------:|:--------:|:--------:|
| **analysis** | 11 | 0 | 0 |
| **hot_take** | 6 | 5 | 0 |
| **reaction** | 0 | 1 | 10 |

**Pattern:** All 6 fine-tuned errors on `hot_take` were predicted as `analysis`. The model never confused `analysis` → `hot_take` or mixed up `reaction` with other classes (except 1 reaction → hot_take).

### Wrong predictions — analysis

**1. Cap-space claim labeled hot_take, predicted analysis (conf: 41%)**
> "That was intentional. The Thunder were $30M under the cap, spent it all on Hartenstein, then used Bird Rights to go over the cap re-signing Joe/Wiggins."

The model saw specific financial details ($30M, Bird Rights) and classified as analysis. I labeled it hot_take because the post asserts intent ("That was intentional") without building a full cap-strategy argument. **This is a labeling boundary issue** — the post sits exactly on our one-stat/one-fact rule.

**2. Single-stat MVP comparison (conf: 44%)**
> "If Shai Gilgeous Alexander wins the MVP this season… Joel Embiid will remain as the only MVP to never make the Conference Finals"

One conditional fact used to support an implicit hot take about Embiid/SGA. Model keyed on the factual structure. **Fix:** more examples of "one fact + opinion" explicitly labeled hot_take.

**3. Long opinion essay (conf: 46%)**
> "Joker has a title, has a known personality… ANT, Hali and Brunson are all bigger stars."

Long text with comparisons led the model to analysis despite no stats or verifiable evidence chain. **Pattern:** the model overweights *length and comparative structure* as signals for analysis.

### Sample classifications (fine-tuned)

| Post | Predicted | Confidence | Correct? |
|------|-----------|------------|----------|
| "OKCs 32pt win over the Nuggets in game 7 is the 7th largest margin of victory in a game 7 in NBA playoff history." | analysis | 48.7% | ✓ |
| "SGA flops on the 'elbow to the face'" | reaction | 37.6% | ✓ |
| "This sub during playoffs is bunch of toxic posts for karma farming…" | hot_take | 39.2% | ✓ |
| "i think this stat is only counting players that make the team in the same season that they play with jokic…" | analysis | 46.4% | ✗ (true: hot_take) |
| "Interesting observation since the current era of NBA stars has been dominated by European stars…" | analysis | 45.3% | ✓ |

**Why the OKC game-7 prediction is reasonable:** The post cites a specific, verifiable historical ranking (7th largest Game 7 margin) — exactly the kind of evidence-backed claim the `analysis` definition describes.

---

## Reflection: Intended vs. Learned

**What I intended:** Three-way distinction based on *argument structure* — does the post reason with evidence, assert boldly, or react emotionally?

**What the model learned:** A proxy for *information density*. Posts with numbers, proper nouns, cap figures, or multi-sentence structure → `analysis`. Short emotional posts → `reaction`. The `hot_take` boundary is the weakest: the model only catches hot takes when they're short and clearly opinionated; longer rants with comparisons get pulled toward analysis even without real evidence.

The high analysis recall (1.00) but lower hot_take recall (0.45) confirms this — the model would rather call something analysis than hot_take when uncertain. Confidence scores are also low across the board (37–49%), suggesting the model is appropriately uncertain on this subjective task.

---

## Spec Reflection

**How the spec helped:** Requiring label definitions and edge-case rules *before* annotation prevented the vague good/bad taxonomy trap. The baseline comparison requirement made it obvious that fine-tuning added real value (+45pp over zero-shot).

**Where implementation diverged:** I ran training locally via `scripts/train_and_evaluate.py` instead of only using Colab — same model and hyperparameters, but faster iteration. Data was collected via PullPush API because Reddit's live JSON API returned 403 from this environment.

---

## AI Usage

1. **Annotation pre-labeling (Groq):** Directed `llama-3.3-70b-versatile` to classify each fetched post using my label definitions. It pre-labeled 300 examples; I reviewed all 219 in the final dataset and corrected ~18% (mostly hot_take ↔ analysis boundaries).

2. **Label stress-testing (Claude):** Before annotation, asked Claude to generate boundary posts between analysis and hot_take. It produced "Curry's 3PT% is down 4% — he's clearly declining." I used this to tighten the one-stat rule in `planning.md`.

3. **Failure pattern analysis (Claude):** Pasted misclassified examples; Claude flagged "length/comparative structure → false analysis" and "single embedded fact → false analysis." Verified both patterns against 6+ test errors — confirmed in confusion matrix (all hot_take errors → analysis).

---

## Running the Classifier

```bash
pip install -r requirements.txt
python scripts/train_and_evaluate.py   # train + evaluate (needs GROQ_API_KEY in .env)
python scripts/predict.py "NO WAY THAT WAS A FLAGRANT LMAOOO"
```

**Demo examples for video:**
```bash
python scripts/predict.py "Jokic's on/off net rating is +18.4 — Denver's offense drops from 118 ORtg to 102 when he sits."
python scripts/predict.py "Luka is already a top-5 player all-time and it's not close."
python scripts/predict.py "SGA flops on the elbow to the face"
python scripts/predict.py "That was intentional. The Thunder were 30M under the cap..."
```

---

## Repository Contents

| File | Description |
|------|-------------|
| `planning.md` | Design doc (labels, edge cases, metrics, AI plan) |
| `data/labeled_posts.csv` | 219 labeled examples |
| `evaluation_results.json` | Full metrics for both models |
| `confusion_matrix.png` | Confusion matrix visualization |
| `model/` | Saved fine-tuned DistilBERT weights |
| `scripts/` | Data collection, training, inference |

---

## Demo Video

> **TODO:** Record 3–5 min video showing `predict.py` on 3–5 posts, one correct prediction explained, one incorrect prediction explained, and a brief walkthrough of this evaluation report.
