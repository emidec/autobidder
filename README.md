# autobidder — reviewer bidding toolkit

A small, self-service toolkit for HotCRP-run reviewing rounds (any conference). **Two commands**
turn your reviewing interests into bids:

1. `make_topic_interests.py` — list the conference's topics for you to rate.
2. `score_bids.py` — build your profile from those ratings + your papers, and fill every submission's
   bid (HotCRP uses roughly **−20…20**).

Runs locally with `scikit-learn` + `pypdf` (no network); the optional `--method specter2` adds a
one-time model download.

**Status:** beta — `v0.1.0-beta`.

---

## Pipeline

```
   revprefs.csv
        │
   make_topic_interests.py
        │
        ▼
   topic_interests.csv   ← you set each topic's interest [-2..2]        papers_pdf/  (your PDFs, ≥5)
        │                                                                      │
        └──────────────────────────────┬───────────────────────────────────────┘
                                       ▼
   revprefs.csv  ─────────────────►  score_bids.py  ─────────►  filled `preference` column
                                        ▲
                                   config.yaml
```

`score_bids.py` builds the profile on the fly (the topic interests differ every conference, so there's
nothing reusable to build separately) and saves it to `reviewer-expertise-profile.json` for inspection.

---

## Getting the CSV in and out of HotCRP

**Download (with abstracts).** Once submissions are visible, HotCRP shows a **Review preferences** link
on the home page; it lists every submission with a preference box. To export a CSV, select all papers
and use the page's **Download** action → *Review preferences*. The default columns are
`paper, title, preference`; to also include the **abstract** and **topics** columns this toolkit scores
on, add those fields to the view first (the **Show** menu / column options on the search bar → enable
*Abstract* and *Topics*), then download. Save it as `revprefs.csv`.

> ⚠️ **Heads-up:** this `revprefs.csv` contains the submissions' **abstracts**, and `score_bids.py`
> **deletes it by default** after scoring — the uploadable output it writes has the abstracts stripped.
> Pass `--keep-original` if you want to retain the abstract-containing CSV.

**Upload (your bids).** After scoring, upload the filled CSV back on the same **Review preferences**
page via its **Upload** action (or **Assignments → Upload**). HotCRP matches each row by its `paper` id
and reads the `preference` column; the other columns are ignored.

> The exact labels may vary a little between HotCRP versions — if your site's controls differ, the rule is:
> get a CSV that includes `paper, title, preference, abstract, topics`, and upload one with `paper` +
> `preference`.

---

## The two steps

### 1. Rate the conference topics

```bash
python3 make_topic_interests.py revprefs.csv        # writes topic_interests.csv: every topic at 0
```

Open `topic_interests.csv` and set each topic's `interest`. **Scale: an integer from -2 to +2** —
`2` = very high, `1` = high, `0` = neutral, `-1` = low, `-2` = very low. (The scale is also printed at
the top of the file.) Leave anything you don't care about at `0`.

### 2. Build the profile + fill the bids (one command)

```bash
python3 score_bids.py revprefs.csv                                # default: ~10% positive -> revprefs.scored.csv
python3 score_bids.py revprefs.csv --keep-original                # also keep the original revprefs.csv
python3 score_bids.py revprefs.csv -o filled.csv --report report.txt
python3 score_bids.py revprefs.csv --positive-frac 0.3           # bid positively on ~30% of papers instead
```

This uses `topic_interests.csv` (override with `--topic-interests`) and your papers in `papers_pdf/`
(**at least 5 unique PDFs, or it stops**). It scores each submission by **TF-IDF cosine similarity to
your most-similar paper**, blends that with your topic interests, writes
`reviewer-expertise-profile.json`, and fills the `preference` column — targeting a positive bid on ~10%
of papers (`--positive-frac`, default 0.1). **By default it deletes the original input and keeps only the
scored output, with the `abstract` column stripped** (`paper, title, preference, topics`) — pass
`--keep-original` to keep the input. It prints a histogram report; `--report` saves it.

---

## Files

| File | Role |
|---|---|
| `make_topic_interests.py` | Creates a blank `topic_interests.csv` (every topic at 0) from the preferences CSV. Stdlib. |
| `score_bids.py` | Scores submissions by similarity to your papers (+ interests) and fills the bids. Needs `scikit-learn` + `pypdf` (`PyYAML` optional). |
| `config.yaml` | Scoring parameters (`bid_max`, `interest_weight`, `sem_gain`). Edit to taste. |
| `topic_interests.csv` | **you edit** — `topic,interest` on a **-2..2** scale. Made by `make_topic_interests.py`. |
| `papers_pdf/` | your papers as PDFs (**≥5 unique**) — matched semantically against each submission. |
| `reviewer-expertise-profile.json` | *(generated — don't hand-edit)* your topic interests + top TF-IDF terms of your papers, for inspection. |
| `revprefs.csv` | the round's submissions (HotCRP export: `paper, title, preference[, abstract, topics]`). Deleted once scored, unless `--keep-original`. |
| `revprefs.scored.csv` | the scored output — **abstracts removed** — this is what you upload back to HotCRP. |

(CSV files and `papers_pdf/` are git-ignored — they hold conference-confidential data.)

---

## Requirements

- Python 3.7+
- `scikit-learn` — for the TF-IDF similarity: `pip install scikit-learn`
- `pypdf` — to read your PDFs: `pip install pypdf`
- `PyYAML` — optional; `score_bids.py` reads `config.yaml` with a built-in fallback parser if it isn't installed.
- `torch` + `transformers` + `adapters` — **only** if you use `--method specter2` (neural embeddings): `pip install torch transformers adapters`.

---

## How it works

### Building your profile

Your PDFs in `papers_pdf/` are read (first ~3 pages each — title, abstract, and intro, where the
topical vocabulary lives) and lightly cleaned (lower-cased, words split across line breaks rejoined).
The vocabulary and weighting (**TF-IDF** — term frequency–inverse document frequency) are learned from
the conference's **submissions**, and your papers are projected into that same space:

- **unigrams + bigrams**, so phrases like *differential privacy* or *membership inference* count as units;
- English stop-words plus PDF boilerplate (*et al*, *figure*, *arxiv*, …) are dropped;
- a term must appear in **≥2 submissions** but **< 30%** of them — dropping both rare noise and generic
  words — with term frequency scaled sub-linearly (a word used 10× isn't 10× as important).

Fitting the vocabulary on the submission pool means anything specific to *your* papers but absent from the
conference — your name, affiliation, venue boilerplate — simply never enters. So your "profile" isn't a
hand-written keyword list; it's *your actual papers as vectors*, in the conference's own vocabulary.
`reviewer-expertise-profile.json` saves the top-weighted terms so you can see what it picked up — this
summary is always TF-IDF-based, even when matching with `--method specter2` (below).

### Scoring a submission

Parameters live in `config.yaml`.

1. **Semantic similarity.** The submission's title+abstract is vectorized in the same space, and we take
   the **cosine similarity to your single most-similar paper** (the *max* over your papers — so a
   submission that strongly matches *any one* of your sub-areas scores high, even if it's unrelated to
   the rest of your work).
2. **Normalize.** Raw cosine similarities are small and bunched together (~0.05–0.2), so they're
   **z-scored** across all submissions — "how far above/below your typical match is this one" — and
   scaled by `sem_gain` onto a ±`ref_max` range.
3. **Blend with interests.** `(1 − interest_weight)·similarity + interest_weight·topic`, where `topic` is
   `0.6·max + 0.4·mean` of your −2..2 interests (×10) for the submission's topic tags. Default
   `interest_weight` 0.35 — similarity leads, your topic ratings steer.
4. **Map to bids.** Threshold at the `--positive-frac` quantile and rescale each side to
   `[-bid_max, bid_max]`, so ~that fraction end up positive **and** your strongest matches still reach
   ±`bid_max`.

**Method choice (step 1).** By default the similarity is **TF-IDF** cosine (above). Pass
`--method specter2` to instead use **AllenAI SPECTER2** neural embeddings — a model trained for
paper-to-paper similarity that matches on *meaning*, so it catches related work phrased differently
(needs `torch`/`transformers`/`adapters` and a one-time model download; slower on CPU). Only step 1
changes — the normalize/blend/map steps and the TF-IDF top-terms summary are identical either way.

The only judgment input is `topic_interests.csv`; everything else is mechanical and in `config.yaml`.

---

## Customizing

- **Re-rate a topic** → edit its `interest` in `topic_interests.csv`, then re-run `score_bids.py`.
- **Target how many papers you bid positively on** → `--positive-frac F` (0–1). **Defaults to `0.1`** (~10%); set e.g. `--positive-frac 0.3` for ~30%. It thresholds at the target quantile and rescales each side to the full range, so ~F end up positive **and** your strongest papers still reach ±`bid_max` (the run reports the achieved fraction, within ±10 points).
- **Balance similarity vs. your interests** → `interest_weight` in `config.yaml` (0 = pure paper-similarity, 1 = pure topic interests; default 0.35). `sem_gain` sharpens how much similarity differences matter.
- **Use neural embeddings instead of TF-IDF** → `--method specter2` (AllenAI SPECTER2). Catches related work phrased differently (semantic, not just shared wording), but needs `pip install torch transformers adapters` and a one-time model download. Default `tfidf` is light and offline; the rest of the pipeline (interest blend, `--positive-frac`) is identical either way. Embeddings are cached in `.specter2_cache.npz` (keyed by abstract text), so re-runs only embed new/changed papers — a fully-cached re-run doesn't even load the model (override the path with `--emb-cache`).
- **Change the output range** → set `bid_max` in `config.yaml` (default 20, max 100). The scorer runs on a fixed ±`ref_max` (20) reference span and *linearly rescales* the final bid to ±`bid_max`. (`bid_max` is an integer in `[1, bid_limit]`; `bid_limit` defaults to HotCRP's 100.)

## Reproducibility

Fully deterministic: same papers + same `topic_interests.csv` + same CSV → identical bids.

## Acknowledgments

Built with the assistance of [Claude](https://www.anthropic.com/claude) (Anthropic).
