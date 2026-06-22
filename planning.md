# TakeMeter — planning.md

> Written before data collection. Updated after annotation and before any stretch features.

---

## Community

**r/nba** — the largest NBA discussion community on Reddit (~8M members). Game threads, trade rumors, player debates, and post-game reactions all happen in the same feed, but regulars clearly distinguish between posts that *argue with evidence*, posts that *assert bold opinions*, and posts that *react in the moment*. That distinction matters here: analysis posts get upvoted in serious threads, hot takes get called out (or celebrated), and reaction posts dominate live game threads. The discourse is text-heavy, publicly accessible, and varied enough that a 3-class taxonomy applies cleanly to most comments.

---

## Labels

Three mutually exclusive labels grounded in how r/nba users actually talk about post quality:

### `analysis`
A structured argument backed by specific, verifiable evidence — stats, film observations, historical comparisons, or tactical breakdowns. The post reasons toward a conclusion; removing the opinion framing would still leave substantive evidence.

**Examples:**
1. "Jokic's on/off net rating this season is +18.4. Denver's offense drops from 118 ORtg to 102 when he sits. That's not a MVP debate — that's a system dependency problem."
2. "They switched everything onto Tatum in the fourth and he went 1/7. Boston's PnR coverage has been ICE all series; Miami's counters (short rolls, ghost screens) exploit the drop coverage gap."

**Uncertain example:** "LeBron's playoff win rate against top seeds is below .500" — one stat, accusatory framing. **Decision:** label `hot_take` unless multiple stats build an argument.

### `hot_take`
A bold, confident opinion stated without genuine supporting evidence. May cite a single cherry-picked stat or anecdote for effect, but the post asserts rather than argues.

**Examples:**
1. "Luka is already a top-5 player all-time and it's not close. People who disagree just hate iso-heavy guards."
2. "The Warriors dynasty was overrated — they only won because everyone was injured."

**Uncertain example:** "Trade AD now before his value craters" with no supporting reasoning. **Decision:** `hot_take` — confident claim, no evidence chain.

### `reaction`
An immediate emotional response to a specific in-game or news event. Little to no argument — expressing a feeling, meme energy, or one-line reaction.

**Examples:**
1. "NO WAY THAT WAS A FLAGRANT LMAOOO"
2. "I'm actually crying right now that was the worst ref call I've ever seen"

**Uncertain example:** "Fire the refs" after a bad call. **Decision:** `reaction` — pure emotional venting, no analytical content.

---

## Hard Edge Cases

**Borderline case:** Posts that cite one stat to support a bold claim (e.g., "Giannis is washed — he's shooting 52% at the rim this playoffs").

**Decision rule:** If removing the opinion sentence leaves only a single decorative stat that doesn't build toward a broader argument, label `hot_take`. If the post compares multiple data points, explains context (league average, prior seasons, matchup), or walks through cause-and-effect, label `analysis`. When genuinely stuck after applying the rule, default to `hot_take` and note in the `notes` column.

**Annotation difficulties encountered (3 examples):**

| Post | Could be | Decided | Why |
|------|----------|---------|-----|
| "SGA's midrange game is unstoppable — 58% on pull-ups this series" | analysis / hot_take | hot_take | One stat cited to back a superlative ("unstoppable"), not a multi-point argument |
| "This team has no heart" after a blowout loss | reaction / hot_take | reaction | Emotional judgment about a specific game, no claim meant to persuade |
| "The Nuggets need to trade Murray before his contract becomes an albatross" | hot_take / analysis | hot_take | Policy opinion with no supporting cap table or performance data |

---

## Data Collection Plan

**Source:** Public Reddit JSON API — r/nba hot, new, and top posts from the past month, plus top-level comments from high-activity game threads and discussion posts.

**Target:** 200+ labeled examples, aiming for ~70 `analysis`, ~70 `hot_take`, ~70 `reaction` (no label above 40%).

**Process:**
1. Fetch 300+ raw comments/posts via `scripts/collect_data.py` (User-Agent: TakeMeter/1.0 student project).
2. Pre-label with Groq (`llama-3.3-70b-versatile`) using definitions above; manually review every label.
3. If any label falls below 50 examples after 200 total, fetch more from threads skewed toward that type (game threads for reactions, offseason debate posts for hot takes, film-breakdown threads for analysis).

**Imbalance contingency:** If one label exceeds 70% after initial pass, oversample underrepresented categories from targeted subreddit searches (`flair:Game Thread`, `flair:Highlight`, long-form text posts).

---

## Evaluation Metrics

**Primary:** Per-class **F1** — the task is subjective and classes are imbalanced; accuracy alone hides a model that predicts only `reaction`. F1 balances precision and recall per label.

**Secondary:** **Overall accuracy** — easy to interpret for baseline comparison.

**Diagnostic:** **Confusion matrix** — reveals directional errors (e.g., `analysis` → `hot_take`) which map directly to label-boundary problems.

**Baseline comparison:** Same test set for fine-tuned DistilBERT vs. zero-shot Groq — the gap tells us whether task-specific training helped.

**Why not accuracy alone:** On a 3-class task with ~35% reactions, a model predicting `reaction` every time gets ~35% accuracy while being useless for analysis and hot_take detection.

---

## Definition of Success

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Fine-tuned beats baseline | ≥5 percentage points higher accuracy on test set | Fine-tuning must earn its cost |
| All-class F1 | ≥0.55 per class | Usable for triage, not perfect moderation |
| No single-class collapse | No class with recall < 0.40 | Model must detect all three types |
| Directional errors explainable | Top confusion pair identifiable in report | Shows we understand failure modes |

"Good enough for deployment" = correctly routing obvious reactions away from analysis threads in a mod queue preview — not replacing human judgment on borderline posts.

---

## AI Tool Plan

### Label stress-testing (done before annotation)
Asked Claude to generate boundary posts between `analysis` and `hot_take`. Produced: "Curry's 3PT% is down 4% — he's clearly declining" (one stat + conclusion). Confirmed our one-stat rule handles it → tightened definition to mention "multi-point argument" explicitly.

### Annotation assistance
Used Groq `llama-3.3-70b-versatile` to pre-label all 220 fetched examples with label definitions in the prompt. **Every label manually reviewed** in `scripts/collect_data.py` review pass; ~18% of pre-labels were corrected. Pre-labeled rows tracked via `notes` column (`pre-labeled, reviewed`).

### Failure analysis (completed — Milestone 6)
Pasted misclassified test examples into Claude; identified systematic pattern: **hot_take → analysis** on posts with embedded facts or length. Verified against all 7 test errors. Documented in README Stretch: Error Pattern Analysis.

---

## Stretch Features (added before implementation)

### Inter-Annotator Reliability
35 held-out examples independently re-labeled by a secondary reviewer (blind to primary labels, same `planning.md` definitions). Agreement measured with Cohen's kappa. See `data/inter_annotator_sample.csv`.

### Confidence Calibration
Bin test-set predictions by confidence bucket; verify higher confidence correlates with higher accuracy. See `scripts/calibration_analysis.py`.

### Deployed Interface
Gradio web UI (`app.py`) — accepts new post, returns label + confidence scores.

---

## Spec Reflection (working notes)

- **Helped:** Forcing label definitions before code prevented collecting 200 vague "good/bad" examples.
- **Diverged:** Ran training locally with a Python script instead of only Colab — same hyperparameters, faster iteration. Baseline prompt format matched Colab notebook spec.
