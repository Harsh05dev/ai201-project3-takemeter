# TakeMeter — r/nba Discourse Classifier

**TakeMeter** is a fine-tuned text classifier that labels r/nba posts and comments by discourse type: structured **analysis**, bold **hot_take**, or in-the-moment **reaction**. Built for CodePath AI201 Project 3.

**Repo:** https://github.com/Harsh05dev/ai201-project3-takemeter

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
4. Balanced to 73 examples per class (219 total). Split 70% / 15% / 15% train/val/test (stratified).

### Label distribution

| Label | Count | % |
|-------|------:|--:|
| analysis | 73 | 33.3% |
| hot_take | 73 | 33.3% |
| reaction | 73 | 33.3% |
| **Total** | **219** | 100% |

No single label exceeds 70%.

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
| Training platform | **Google Colab (T4 GPU)** + local verification (`scripts/train_and_evaluate.py`) |
| Framework | `transformers` Trainer + `datasets` |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max sequence length | 256 tokens |
| Split | 70% train / 15% val / 15% test (stratified) |

**Hyperparameter decision:** Kept **learning rate at 2e-5** (not 5e-5). With only ~153 training examples, 5e-5 caused validation loss to spike after epoch 1. At 2e-5, validation accuracy improved across epochs (51% → 64% → 61%). Also used `load_best_model_at_end=True` on Colab to keep the best checkpoint by validation accuracy.

Colab notebook: [TakeMeter starter](https://colab.research.google.com/drive/1On-MpFpQCQ3UU0NJ-zYKZF-X7zh1VVcd)

---

## Baseline

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot, no task-specific training)  
**Collection:** Same locked test set (33 examples), one API call per example, `temperature=0`, 0.1s delay between requests.

**Full prompt used:**

```
You are classifying posts and comments from r/nba.
Assign each post to exactly one of the following categories.

analysis: A structured argument backed by specific, verifiable evidence — stats, film
observations, historical comparisons, or tactical breakdowns. The post reasons toward a conclusion.
Example: "OKC's 32pt win over the Nuggets in game 7 is the 7th largest margin of victory
in a game 7 in NBA playoff history."

hot_take: A bold, confident opinion without genuine supporting evidence. May cite one
cherry-picked stat for effect but asserts rather than argues.
Example: "Luka is already a top-5 player all-time and it's not close."

reaction: An immediate emotional response to a specific in-game or news event. Little to no
argument — expressing a feeling in the moment.
Example: "SGA flops on the elbow to the face"

Edge rule: If a post cites only ONE stat to support a bold claim without building a
multi-point argument, label it hot_take.

Respond with ONLY the label name. Valid labels: analysis, hot_take, reaction
```

---

## Evaluation Report

### Overall accuracy (same test set, n=33)

| Model | Accuracy |
|-------|----------|
| Groq zero-shot baseline | **33.3%** |
| Fine-tuned DistilBERT | **78.8%** |

Fine-tuning beat baseline by **+45.5 percentage points**.

### Per-class metrics — fine-tuned

| Label | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| analysis | 0.65 | 1.00 | **0.79** |
| hot_take | 0.83 | 0.45 | **0.59** |
| reaction | 1.00 | 0.91 | **0.95** |
| **Macro avg** | 0.83 | 0.79 | **0.78** |

### Per-class metrics — baseline

| Label | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| analysis | 0.00 | 0.00 | 0.00 |
| hot_take | 0.00 | 0.00 | 0.00 |
| reaction | 0.33 | 1.00 | 0.50 |

Baseline predicted `reaction` for nearly every example.

### Confusion matrix (fine-tuned)

|  | analysis | hot_take | reaction |
|--|:--------:|:--------:|:--------:|
| **analysis** | 11 | 0 | 0 |
| **hot_take** | 6 | 5 | 0 |
| **reaction** | 0 | 1 | 10 |

See also: [`confusion_matrix.png`](confusion_matrix.png)

### Wrong predictions — analysis (3+)

**1. Cap-space claim → analysis (conf: 41%, true: hot_take)**  
> "That was intentional. The Thunder were $30M under the cap, spent it all on Hartenstein, then used Bird Rights to go over the cap re-signing Joe/Wiggins."

Model keyed on specific financial terms ($30M, Bird Rights). I labeled hot_take because the post asserts intent without a full cap-strategy argument. **Label boundary issue** at our one-fact rule.

**2. Single-stat MVP comparison → analysis (conf: 44%, true: hot_take)**  
> "If Shai Gilgeous Alexander wins the MVP… Joel Embiid will remain as the only MVP to never make the Conference Finals"

One conditional fact supporting an implicit take. Model treats factual structure as analysis.

**3. Long opinion essay → analysis (conf: 46%, true: hot_take)**  
> "Joker has a title, has a known personality… ANT, Hali and Brunson are all bigger stars."

Multi-paragraph comparison with no stats. Model overweights **length and comparative structure** as analysis signals.

**4. Short reaction → hot_take (conf: 34%, true: reaction)**  
> "SGA flops on the 'elbow to the face'"

Borderline: opinionated but in-the-moment. Model missed the live-game reaction context.

### Sample classifications (fine-tuned)

| Post | Predicted | Confidence | Correct? |
|------|-----------|------------|----------|
| "OKCs 32pt win… 7th largest margin of victory in a game 7 in NBA playoff history." | analysis | 48.7% | ✓ |
| "SGA flops on the 'elbow to the face'" | reaction | 37.6% | ✓ |
| "This sub during playoffs is bunch of toxic posts for karma farming…" | hot_take | 39.2% | ✓ |
| "i think this stat is only counting players… with jokic…" | analysis | 46.4% | ✗ |
| "Interesting observation… European stars like Luka, Jokic and Giannis…" | analysis | 45.3% | ✓ |

**Why OKC game-7 is reasonable:** cites a specific, verifiable historical ranking — exactly what `analysis` describes.

---

## Reflection: Intended vs. Learned

**Intended:** Three-way distinction by *argument structure* — evidence, assertion, or emotion.

**Learned:** A proxy for *information density*. Numbers, proper nouns, cap figures, multi-sentence structure → `analysis`. Short emotional posts → `reaction`. `hot_take` is weakest: longer rants with comparisons get pulled to analysis without real evidence.

Specific failure pattern: **`hot_take` → `analysis`** accounts for 6/7 test errors. The model never confuses `analysis` → `hot_take`. High analysis recall (1.00) + low hot_take recall (0.45) = model defaults to analysis when uncertain. Confidence scores cluster at 37–48%, so the model knows it's uncertain on this subjective task.

---

## Stretch Features

### Inter-Annotator Reliability (+1pt)

35 held-out examples independently labeled by two annotators (blind, same definitions):

| Metric | Value |
|--------|-------|
| Percent agreement | **82.9%** (29/35) |
| Cohen's κ | **0.73** (substantial agreement) |

**Disagreement pairs:** hot_take vs reaction (3), hot_take vs analysis (2), analysis vs hot_take (1).

**Where they disagreed:** Highlight/link posts ("Alex Caruso highlights from game 7") — one annotator saw title-style hot_take, the other saw reaction. Posts with embedded reporting language (Stein tweet) split hot_take vs analysis.

Data: [`data/inter_annotator_sample.csv`](data/inter_annotator_sample.csv) · Script: `scripts/inter_annotator.py`

### Confidence Calibration (+1pt)

| Confidence bucket | n | Accuracy | Avg confidence |
|-------------------|--:|---------:|---------------:|
| 0%–35% | 2 | 50.0% | 34.3% |
| 35%–40% | 10 | 100.0% | 38.1% |
| 40%–45% | 9 | 55.6% | 41.9% |
| 45%–50% | 11 | 81.8% | 47.0% |
| 50%+ | 1 | 100.0% | 50.0% |

**Verdict:** Weakly calibrated but directionally correct — accuracy rises from 50% (lowest bucket) to 82–100% (45%+ confidence). Most predictions sit at 37–48% confidence, reflecting appropriate uncertainty on a subjective 3-class task.

Chart: [`calibration_chart.png`](calibration_chart.png) · Data: [`data/calibration_results.json`](data/calibration_results.json)

### Error Pattern Analysis (+1pt)

**Systematic pattern: `hot_take` → `analysis` on "fact-decorated opinions."**

| Evidence | Count |
|----------|------:|
| Test errors with this pattern | 6/7 |
| Avg post length | 142 chars |
| Posts containing numbers | 5/6 |
| Posts with proper nouns (players/teams) | 6/6 |

The model treats **presence of facts** as **presence of argument**. It does not distinguish one embedded stat used for effect vs. a multi-point analytical chain. Short reactions with opinion framing ("SGA flops…") are a secondary failure mode (1/7 errors).

**Fix:** Add 30+ training examples of single-stat hot_takes and short opinionated reactions explicitly labeled.

### Deployed Interface (+1pt)

Gradio web UI accepts a new post and shows predicted label + per-class confidence.

```bash
pip install -r requirements.txt
python app.py
```

Opens a browser at `http://127.0.0.1:7860`. CLI alternative: `python scripts/predict.py "your post here"`.

---

## Spec Reflection

**How the spec helped:** Requiring label definitions and edge-case rules *before* annotation prevented a vague good/bad taxonomy. The baseline comparison requirement proved fine-tuning added +45pp over zero-shot.

**Where implementation diverged:** Trained on **Google Colab (T4)** per spec, but also ran `scripts/train_and_evaluate.py` locally for faster iteration. Data collected via PullPush API because Reddit's live JSON API returned 403.

---

## AI Usage

1. **Annotation pre-labeling (Groq):** Pre-labeled 300 posts with `llama-3.3-70b-versatile`; manually reviewed all 219 and corrected ~18% (mostly hot_take ↔ analysis).

2. **Label stress-testing (Claude):** Generated boundary posts before annotation. Produced "Curry's 3PT% is down 4% — he's clearly declining." → tightened one-stat rule in `planning.md`.

3. **Failure pattern analysis (Claude):** Pasted misclassified examples; flagged length/comparative-structure → false analysis. Verified against all 6 hot_take→analysis errors in confusion matrix.

4. **Code scaffolding (Cursor):** Generated `collect_data.py`, `train_and_evaluate.py`, and `app.py`; I overrode label definitions, balancing logic, and hyperparameter choices.

---

## Demo Video

Record 3–5 minutes showing:

1. **Interface:** `python app.py` → classify 3–5 posts with label + confidence visible
2. **Correct prediction:** Jokic on/off stats → `analysis` (~49%) — "has verifiable evidence"
3. **Incorrect prediction:** Thunder cap post → `analysis` but labeled `hot_take` — "one fact, assertive framing"
4. **Evaluation walkthrough:** README metrics table + confusion matrix

**CLI demo commands:**
```bash
cd ai201-project3-takemeter
python app.py
# or:
python scripts/predict.py "Jokic's on/off net rating is +18.4 — Denver's offense drops from 118 ORtg to 102 when he sits."
python scripts/predict.py "Luka is already a top-5 player all-time and it's not close."
python scripts/predict.py "SGA flops on the elbow to the face"
```

---

## Repository Contents

| File | Description |
|------|-------------|
| `planning.md` | Design doc |
| `data/labeled_posts.csv` | 219 labeled examples |
| `data/inter_annotator_sample.csv` | 35 dual-annotated examples |
| `evaluation_results.json` | Full metrics |
| `confusion_matrix.png` | Confusion matrix |
| `calibration_chart.png` | Calibration analysis |
| `app.py` | Gradio web interface |
| `scripts/` | Collection, training, inference, stretch analyses |
