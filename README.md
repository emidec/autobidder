# autobidder — reviewer bidding toolkit

Fill in your HotCRP reviewer bids automatically. Rate the conference's topics once, point the tool at
your own papers, and it scores every submission by how well it matches your work. Everything runs
locally — no submission data leaves your machine.

**Version:** `v0.5.0`.

---

## Contents

- [How to use autobidder](#how-to-use-autobidder)
- [The HotCRP round trip](#the-hotcrp-round-trip)
- [Matching methods](#matching-methods)
  - [Sizing the shortlist](#sizing-the-shortlist)
- [Requirements](#requirements)
  - [Installing a neural method](#installing-a-neural-method)
  - [Fetching the models](#fetching-the-models)
  - [Running on a GPU](#running-on-a-gpu)
- [Options](#options)
  - [config.yaml](#configyaml)
  - [Flags](#flags)
- [Files](#files)
- [How it works](#how-it-works)
  - [Building your profile](#building-your-profile)
  - [Checking your topic interests](#checking-your-topic-interests)
  - [Scoring a submission](#scoring-a-submission)
- [Performance](#performance)
  - [Benchmarks](#benchmarks)
  - [Accuracy](#accuracy)
  - [Reproducibility](#reproducibility)
- [Development](#development)
- [Acknowledgments](#acknowledgments)

---

## How to use autobidder

```bash
pip install -r requirements.txt            # scikit-learn + pypdf
```

That is all the default [matching method](#matching-methods), `tfidf`, needs. Each of the others
needs its own requirements file (`requirements-specter2.txt` or `requirements-rerank.txt`) and a
one-time model download: set those up from [Requirements](#requirements) before your first run. On
**Linux**, read it before installing either file — pip otherwise pulls 2–3 GB of CUDA you may not
need.

Then:

**1.** Put at least 5 of your own papers in `papers_pdf/`, as text-based PDFs — scans don't work.

**2.** Export the submissions from HotCRP as `revprefs.csv`, including the `abstract` and `topics`
columns ([click-path](#the-hotcrp-round-trip)).

**3.** Generate the topic list:

```bash
python3 make_topic_interests.py revprefs.csv
```

Then set each topic's `interest` in the `topic_interests.csv` it wrote — `+2` want, `+1` happy to,
`0` neutral, `-1` rather not, `-2` avoid. This is the only judgment call in the tool.

**4.** Score every submission and fill in your bids. `--method` chooses how submissions are matched
against your papers — only that step differs, so bids stay comparable across methods:

```bash
python3 score_bids.py revprefs.csv                           # tfidf (default): shared wording, nothing more to install
python3 score_bids.py revprefs.csv --method specter2         # meaning, not just wording  (+ requirements-specter2.txt)
python3 score_bids.py revprefs.csv --method specter2+scincl  # specter2 averaged with a second model  (same file)
python3 score_bids.py revprefs.csv --method rerank           # most precise, reads pairs  (+ requirements-rerank.txt, beta)
```

[Matching methods](#matching-methods) says what each one does.

**5.** Upload the `revprefs.scored.<timestamp>.csv` it wrote back to HotCRP. Done.

> ⚠️ Step 4 **deletes `revprefs.csv`** — it holds every submission's abstract, and the scored output
> has them stripped so the confidential copy doesn't linger. Pass `--keep-original` to keep it.

By default this bids positively on ~10% of papers; change that with `--positive-frac`. It and the
rest of the flags are in [Options](#options).

---

## The HotCRP round trip

**Export.** HotCRP generally shows a **Review preferences** link on the home page, listing every
submission with a preference box. Select all papers and use the page's **Download** action →
*Preference file with abstracts*. It has to be that one: the plain *Preference file* beside it has
only `paper, title, preference`, whatever columns the page is showing. Save the download as
`revprefs.csv` — or keep the name HotCRP gives it, `<conference>-revprefs.csv`, and pass that to the
commands instead.

`score_bids.py` checks the columns before it does anything else:

| Column | |
|---|---|
| `paper` | **required** — HotCRP matches rows by paper id on upload, so an export without it can't be uploaded at all |
| `preference` | **required** — this is the column that gets filled in |
| `abstract` | **required**, unless you pass `--title-only` — without it, matching falls back to titles alone |
| `topics` | strongly wanted — without it your topic interests can't be applied, so the `interest_weight` share of every score (35% by default) is a constant 0 |

A missing `abstract` column **stops the run**: it means you downloaded the plain *Preference file*
rather than *Preference file with abstracts*. Pass `--title-only` to score on titles anyway — possible,
just much weaker. A missing `topics` column only warns, and `topic_interests.csv` isn't needed then.

**Upload.** Upload the scored CSV on the same **Review preferences** page via its **Upload** action (or
**Assignments → Upload**). HotCRP matches each row by its `paper` id and reads the `preference` column;
the other columns are ignored.

> Labels vary a little between HotCRP versions. If your site's controls differ, the rule is: get a CSV
> that includes `paper, title, preference, abstract, topics`, and upload one with `paper` +
> `preference`.

---

## Matching methods

They all do the same thing — score each submission by similarity to your papers — and differ only in
how that similarity is computed. Everything downstream (the interest blend, `--positive-frac`, the
report) is identical, so you can switch freely between rounds or re-run with `--keep-original` and
compare.

| `--method` | Needs | Speed | Good for |
|---|---|---|---|
| **`tfidf`** *(default)* | nothing extra | fast | Shared vocabulary. Light, fully offline, no model download. |
| **`specter2`** | `requirements-specter2.txt` | slower | Meaning rather than wording — catches related work phrased differently. |
| **`specter2+scincl`** | same as `specter2` | slower (×2) | SPECTER2 and SciNCL similarities averaged 50/50. Resembles OpenReview's ensemble — see below. |
| **`rerank`** *(beta)* | `requirements-rerank.txt` *(pulls torch)* | slowest | The most precise. TF-IDF shortlists, then a cross-encoder rescores the shortlist. |

Every method but `tfidf` is neural: it loads a model, which takes a one-time setup first
([Requirements](#requirements)). [Performance](#performance) has what each one costs to run.

**`tfidf`** compares wording. The vocabulary and its weighting are learned from the conference's
submissions, and your papers are projected into that same space;
[Building your profile](#building-your-profile) has the details. There is no model, so it runs in
seconds.

**`specter2`** uses AllenAI SPECTER2, trained specifically for paper-to-paper similarity. It embeds
both sides as `title [SEP] abstract`, the form the model was trained on — your papers are parsed
down to their own title+abstract, falling back to the raw page text only when no abstract parses,
rather than being fed the three-page read. Embeddings are cached in `.specter2_cache.npz` keyed by
the text they came from, so re-runs only embed what's new — a fully-cached re-run doesn't even load
the model. Override the path with `--emb-cache`.

**`specter2+scincl`** adds [SciNCL](https://huggingface.co/malteos/scincl), a SciBERT model trained
on citation-graph neighbours, with the same input and pooling as SPECTER2 and no adapter (its
embeddings are cached separately, in `.scincl_cache.npz`). It runs both and averages each
submission's cosine similarity to each of your papers with equal weight, before the usual top-3
mean.

This resembles the SPECTER2+SciNCL model of
[openreview-expertise](https://github.com/openreview/openreview-expertise), OpenReview's
reviewer-expertise system, whose README says the two "are ensembled at the paper-paper similarity
level with equal weights of 0.5 on each model". It is a reimplementation of that idea, not
OpenReview's code, and it is not guaranteed to reproduce OpenReview's scores. Known differences:
OpenReview loads the Aug-2023 refresh of SPECTER2, `allenai/specter2_aug2023refresh`, where
autobidder uses `allenai/specter2`; and autobidder aggregates over your papers with its own top-3
mean.

In practice the ensemble's positive bids overlap more with SciNCL alone than with `specter2` alone:
SciNCL's similarities spread wider, so they dominate a plain average.
OpenReview averages the same way.

**`rerank`** *(beta)* is a two-stage pipeline. TF-IDF ranks the whole pool cheaply; the top slice
of that ranking becomes a **shortlist**, and a local cross-encoder rescores only the shortlist. The
cross-encoder reads each *(your paper, submission)* pair **together** and judges relevance directly,
rather than embedding each text alone and comparing vectors — more accurate, but far too slow to run
on every submission, which is the whole reason for the shortlist. Each shortlisted submission is
scored against your papers and aggregated with a top-3 mean.

TF-IDF cosines and cross-encoder logits aren't comparable, so they're never compared: each group is
rank-transformed within itself, and the reranked submissions are placed as a band strictly above the
rest, which keep their TF-IDF order. The usual normalize/blend/map steps then run unchanged.

### Sizing the shortlist

**`--rerank-frac`** — the fraction of submissions that get cross-encoded. Defaults to
**`--positive-frac` + 0.10**, so `0.2` at the default target and `0.35` at `--positive-frac 0.25`.
The default stops at `0.5`: a higher target gets a shortlist exactly its own size.

```bash
python3 score_bids.py revprefs.csv --method rerank --rerank-frac 0.3   # rescore the top 30%
```

- **Never below `--positive-frac`** — reranked submissions sort above the rest, so positive bids
  come from the reranked slice, and a smaller shortlist would leave some of them on papers the
  cross-encoder never saw. The run stops and tells you the minimum.
- **Above ~0.5, use `specter2` instead** — you'd be rescoring most of the pool, so the two-stage
  saving is gone. The run says so.

The `0.10` of headroom lets the cross-encoder promote submissions TF-IDF ranked too low. It's a
reasoned starting value, not a measured optimum.

---

## Requirements

**Python 3.7+** for the scripts themselves; in practice your `scikit-learn` build sets the floor
(recent versions need 3.10+).

| Package | When | Install |
|---|---|---|
| `scikit-learn` | always | `pip install -r requirements.txt` |
| `pypdf` | always | *(same command)* |
| `PyYAML` | optional — `config.yaml` has a built-in fallback parser without it | `pip install PyYAML` |
| `sentence-transformers` | `--method rerank` | `pip install -r requirements-rerank.txt` |
| `torch` | every method but `tfidf` | pulled in by either per-method file — see the Linux warning below |
| `transformers` | every method but `tfidf` | pulled in by `sentence-transformers`; explicit in `requirements-specter2.txt` |
| `adapters` | `--method specter2` and `specter2+scincl` | `pip install -r requirements-specter2.txt` |

`sentence-transformers` pulls `torch` and `transformers` even though its file doesn't name them,
which is how the CUDA download below catches people out.

### Installing a neural method

`tfidf` needs only the base packages. Each other method has its own requirements file, which
includes them.

> ⚠️ **Linux: `torch` defaults to the CUDA build.** pip serves it whether or not you own an NVIDIA
> GPU, dragging in a dozen `nvidia-*` packages totaling **2–3 GB**. To get the CPU build instead, run
> this *before* either per-method command:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```
>
> With an NVIDIA card, take a CUDA build instead — see [Running on a GPU](#running-on-a-gpu). macOS
> is unaffected — there is only one wheel for Darwin (CPU + MPS).

Then install the file for the method you want:

```bash
pip install -r requirements-rerank.txt     # for --method rerank    (sentence-transformers)
pip install -r requirements-specter2.txt   # for --method specter2 and specter2+scincl  (torch transformers adapters)
```

> ⚠️ **Both methods in one environment: install them together.** `adapters` pins `transformers` to a
> 4.x release, while the newest `sentence-transformers` needs 5.x (as of `adapters` 1.3.0 and
> `sentence-transformers` 6.3.0). Installed one after the other, the second changes `transformers`
> underneath the first and breaks it. In a single command, pip picks versions that suit both:
>
> ```bash
> pip install -r requirements-rerank.txt -r requirements-specter2.txt
> ```
>
> Or keep one virtual environment per method.

### Fetching the models

Every method but `tfidf` downloads its model once (two models for `specter2+scincl`), then runs
offline and deterministically. Fetch it **before** your first run — otherwise it downloads
mid-scoring, which looks like a hang:

```bash
python3 fetch_models.py rerank          # ~2.3 GB
python3 fetch_models.py specter2        # ~0.5 GB, plus its adapter
python3 fetch_models.py specter2+scincl # the above plus SciNCL, ~0.4 GB more
python3 fetch_models.py all --check     # what's already cached? downloads nothing
```

Models go to the shared HuggingFace cache (`~/.cache/huggingface`, or `HF_HOME`), which is where
`score_bids.py` looks. `--check` exits non-zero if anything is missing, so it works in a pre-flight
script.

Offline is enforced rather than assumed: once the models a run needs are in the cache, the run switches
the HuggingFace libraries to offline mode (`HF_HUB_OFFLINE=1`), so they can't contact the Hub even to
check for updates. If a cached model then fails to load — an interrupted download, usually — the run
stops and tells you to re-run with `HF_HUB_OFFLINE=0`, which lets it fetch what's missing.

### Running on a GPU

Every method but `tfidf` runs on a GPU whenever the installed `torch` can use one — that's what the
default `--device auto` does, trying CUDA, then Apple MPS, then the CPU. `tfidf` has no model and
always runs on the CPU in seconds anyway. The run says where it went:

```
Loading reranker (BAAI/bge-reranker-v2-m3) on cuda (NVIDIA TITAN X (Pascal))...
```

On Linux with an NVIDIA card, this means skipping the CPU-only install above and taking a CUDA build
of `torch` instead. Pick the wheel matching your card from
[pytorch.org](https://pytorch.org/get-started/locally/). Newer CUDA builds drop support for older
cards, so if `auto` reports it can't use yours, step back to an older build — `cu126` runs Pascal
cards (GTX 10-series, TITAN X):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-rerank.txt -r requirements-specter2.txt
```

[Benchmarks](#benchmarks) has measured GPU and CPU times.

`auto` falls back to the CPU, with a note, if the GPU it finds can't actually run (a card too old for
the installed `torch` build, typically). Naming a device — `--device cuda`, `--device cuda:1` to pick
the second card, `--device mps` — stops the run instead, so you find out rather than silently waiting
on the CPU. `--device cpu` keeps everything off the GPU.

---

## Options

### config.yaml

| Key | Default | What it does |
|---|---|---|
| `interest_weight` | `0.35` | How much your topic ratings steer the bid vs. similarity to your papers. `0` = pure paper-similarity, `1` = pure topic interests. |
| `sem_spread` | `1.0` | Shapes the rank curve. `1` is linear; `2` gives more separation between near-miss and on-target papers, `0.5` flattens them toward neutral. Changes how *strong* bids are, not how many are positive — that's `--positive-frac`. Range `0.1`–`10`. |
| `bid_max` | `20` | The output range: bids are written in `[-bid_max, bid_max]`. An integer in `[1, bid_limit]`. |
| `bid_limit` | `100` | Validation ceiling for `bid_max` — HotCRP's own maximum. |
| `ref_max` | `20` | Internal reference span the score is computed on. **Has no effect on your bids** — both halves of the blend scale with it and the final mapping depends only on ordering. Set `bid_max` instead. |

### Flags

All of these belong to `score_bids.py`.

| Flag | Default | What it does |
|---|---|---|
| `--positive-frac F` | `0.1` | Fraction of papers to bid positively on (0–1). See below. |
| `--keep-original` | off | Don't delete the input CSV after scoring. |
| `--title-only` | off | Accept an export with no `abstract` column and match on titles alone. Without it, such an export stops the run. |
| `--method M` | `tfidf` | Similarity method: `tfidf`, `specter2`, `specter2+scincl`, or `rerank`. See [Matching methods](#matching-methods). |
| `-o`, `--output PATH` | timestamped | Output CSV path. |
| `--report PATH` | — | Also write the histogram report to a file. |
| `--quiet` | off | Don't print the report. |
| `--topic-interests PATH` | `topic_interests.csv` | Where your topic ratings live. |
| `--pdfs DIR` | `papers_pdf` | Folder of your paper PDFs. |
| `--config PATH` | `config.yaml` | Scoring parameters file (defaults to the one beside the script). |
| `--profile-out PATH` | timestamped | Where to save the profile summary JSON. |
| `--rerank-frac F` | `--positive-frac` + 0.10, capped at 0.5 | `rerank` only: fraction of submissions the cross-encoder rescores. See [Sizing the shortlist](#sizing-the-shortlist). |
| `--rerank-model ID` | `BAAI/bge-reranker-v2-m3` | `rerank` only: cross-encoder model id. |
| `--rerank-max-length N` | `1024` | `rerank` only: joint token budget per (paper, submission) pair. |
| `--emb-cache PATH` | `.specter2_cache.npz` | Embedding methods: SPECTER2's cache file. SciNCL's goes beside it (`.scincl_cache.npz`). |
| `--device D` | `auto` | Every method but `tfidf`: where the model runs — `auto` (a CUDA GPU, then Apple MPS, then the CPU), `cpu`, `cuda`, `cuda:N`, or `mps`. See [Running on a GPU](#running-on-a-gpu). |

**Note on `--positive-frac F`:** this is the fraction of papers that should end up with a positive
bid. The threshold goes just below that many papers
([Scoring a submission](#scoring-a-submission), step 4), and the run reports the fraction it
achieved; ties at the threshold are the one thing that can hold it below target, and it warns if the
miss exceeds 10 points.

---

## Files

| File | Role |
|---|---|
| `make_topic_interests.py` | Creates a blank `topic_interests.csv` (every topic at 0) from the preferences CSV. Stdlib only. |
| `score_bids.py` | Scores submissions by similarity to your papers (+ interests) and fills the bids. |
| `fetch_models.py` | Pre-downloads the models for `specter2`, `specter2+scincl` and `rerank` so a scoring run never stalls on it. |
| `config.yaml` | Scoring parameters. Edit to taste. |
| `requirements*.txt` | What to `pip install`: the base packages, plus one file each for `rerank` and the embedding methods (`requirements-specter2.txt`). |
| `topic_interests.csv` | **you edit** — `topic,interest` on a **-2..2** scale. Made by `make_topic_interests.py`; each run reports unrated tags and unparsable values. |
| `papers_pdf/` | your papers as PDFs (**≥5 unique, with a text layer**) — matched semantically against each submission. |
| `revprefs.csv` | the round's submissions (HotCRP export). Deleted once scored, unless `--keep-original`. |
| `revprefs.scored.<ts>.csv` | the scored output — **abstracts removed** — this is what you upload back to HotCRP. |
| `revprefs.scored.<ts>.changes.txt` | *(generated)* what changed vs. the previous run's bids (movers, added/dropped submissions). |
| `reviewer-expertise-profile.<ts>.json` | *(generated — don't hand-edit)* the run's settings (method, targets, `bid_max`, which papers were used) plus your topic interests and the top TF-IDF terms of your papers. |

Every run stamps its outputs with a timestamp, so repeated runs never overwrite earlier results. Pass
an explicit `-o` / `--profile-out` to choose fixed names instead.

The `…changes.txt` diffs the new bids against the previous run: how many changed, moved into or out of
positive, the biggest movers, and any submissions added or dropped. It names the profile JSON for its
own run, so when bids move you can compare the two runs' settings to see why. The baseline is the
newest run stamped earlier than this one — or, with a fixed `-o`, the file at that path that this
run replaces, read before it's overwritten. The first run just notes there's nothing to compare
against.

(CSV files and `papers_pdf/` are git-ignored — they hold conference-confidential data.)

---

## How it works

```
   revprefs.csv
        │
   make_topic_interests.py
        │
        ▼
   topic_interests.csv   ← you set them [-2..2]     papers_pdf/  (your PDFs, ≥5)
        │                                                    │
        └──────────────────────────────┬─────────────────────┘
                                       ▼
   revprefs.csv  ─────────────────►  score_bids.py  ─────────►  revprefs.scored.<ts>.csv
   (HotCRP export)                      ▲                        (upload this back to HotCRP)
                                   config.yaml
```

The output is the same table as the export, with the `preference` column filled in and the `abstract`
column stripped. `score_bids.py` builds your profile on the fly — the topic interests differ every
conference, so there's nothing reusable to build separately — and saves a summary to
`reviewer-expertise-profile.<ts>.json` for inspection.

### Building your profile

Your PDFs in `papers_pdf/` are read (first ~3 pages each — title, abstract, and intro, where the
topical vocabulary lives) and lightly cleaned (lower-cased, words split across line breaks rejoined).
The vocabulary and weighting (**TF-IDF** — term frequency–inverse document frequency) are learned from
the conference's **submissions**, and your papers are projected into that same space:

- **unigrams + bigrams**, so phrases like *side channel* or *access control* count as
  units;
- English stop-words plus PDF boilerplate (*et al*, *figure*, *arxiv*, …) are dropped;
- a term must appear in **≥2 submissions** but **< 30%** of them — dropping both rare noise and generic
  words — with term frequency scaled sub-linearly (a word used 10× isn't 10× as important).

Those two bounds can only both hold once the pool has **at least 7 submissions**, so smaller rounds stop
with an explanation rather than a traceback — as does a pool whose abstracts are so alike that every
term lands above the 30% ceiling, which usually means the `abstract` column is empty.

A PDF only contributes if text can be extracted from it. Scanned, image-only, and encrypted files
yield nothing, so each one is **named on stderr and skipped**, and the run **stops** if fewer than 5
papers with usable text remain — without them the bids would be scored on your topic interests alone.
OCR such a paper, or re-export it with a text layer, to include it.

Fitting the vocabulary on the submission pool means anything specific to *your* papers but absent from
the conference — your name, affiliation, venue boilerplate — simply never enters. So your "profile"
isn't a hand-written keyword list; it's *your actual papers as vectors*, in the conference's own
vocabulary. The profile JSON saves the top-weighted terms so you can see what it picked up — this
summary is always TF-IDF-based, whichever `--method` you matched with.

### Checking your topic interests

Topics are matched by exact name, so `score_bids.py` checks `topic_interests.csv` every run and reports
on stderr:

- any `interest` that isn't a plain integer in −2..2, with the value it was read as (a word reads as
  `0`, an out-of-range number is clamped);
- topic tags on the submissions with **no row** in your file — they score neutral, which is what you'd
  see if the conference added or renamed a topic after you built it, or a name got mistyped;
- rows in your file that match **no submission** this round, e.g. left over from a previous conference.

None of these stop the run — a conference can legitimately carry topics you never rated — but a tag you
meant to rate showing up as unrated is worth knowing before you upload.

### Scoring a submission

Parameters live in `config.yaml`.

1. **Semantic similarity.** The submission's title+abstract is vectorized in the same space, and we take
   the **mean cosine similarity to your top-3 most-similar papers** — so a submission that strongly
   matches *any one* of your sub-areas still scores high, but a single fluke neighbor (or one shared rare
   bigram) can't spike it on its own. This is the only step `--method` changes — see
   [Matching methods](#matching-methods).
2. **Normalize.** Cosine similarities are small, bunched, and skewed (TF-IDF piles near zero; SPECTER2
   sits high even for unrelated papers), so each submission is **rank/quantile-transformed** across the
   pool onto a ±`ref_max` range — "where does this rank among your matches this year." `sem_spread`
   shapes the curve (1 = linear; higher pushes mid-rank papers toward the extremes). Unlike a
   z-score, this doesn't depend on the pool's spread, so the blend in step 3 behaves the same across
   venues.
3. **Blend with interests.** `(1 − interest_weight)·similarity + interest_weight·topic`, where `topic` is
   `0.6·max + 0.4·mean` of your −2..2 interests (×10) for the submission's topic tags. Default
   `interest_weight` 0.35 — similarity leads, your topic ratings steer.
4. **Map to bids.** Put the threshold just below the `--positive-frac` count and rescale each side to
   `[-bid_max, bid_max]`, so that many end up positive **and** your strongest matches still reach
   ±`bid_max`.

The only judgment input is `topic_interests.csv`; everything else is mechanical and in `config.yaml`.

---

## Performance

For `n` = #submissions, `P` = #papers in `papers_pdf/`, and `R` = `--rerank-frac` (the fraction of
submissions that get cross-encoded), the number of neural forward passes is:

```
tfidf              0             no model at all
specter2           n + P         each text embedded once, independently
specter2+scincl    2 · (n + P)   each text embedded once by each of the two models
rerank             R · n · P     every shortlisted submission against every paper
```

Notes:

- **Adding a PDF is free for `specter2` and `specter2+scincl`, and linear for `rerank`.** One more
  paper is one more embedding (two for the ensemble) versus one more pass over the entire shortlist.
  Going from 10 papers to 20 leaves `specter2` essentially unchanged and doubles `rerank`.
- **`--positive-frac` is also a cost knob, though for `rerank` only**, because `R` defaults to
  tracking it. Raising your bid target raises the shortlist, and the shortlist multiplies by `P`.
- **Re-runs are nearly free for `specter2` and `specter2+scincl`** — embeddings are cached by text,
  per model, so a second run with different `--positive-frac` re-embeds nothing. The ensemble costs
  both models' time, minus whatever either cache already holds. `rerank` has no such cache and
  repeats every pair.

The reranker is a ~568M-parameter model reading pairs up to `--rerank-max-length` tokens, while
SPECTER2 is a ~110M-parameter model reading one text at 512. Attention is quadratic in length, so a
rerank pass costs an order of magnitude more than a SPECTER2 one on top of there being more of them.

### Benchmarks

Measured on a real round: **~2,000 submissions × 12 papers**, at the default `--positive-frac 0.1`
(so `R = 0.2`). Every machine used the same inputs. Each run timed one method on one device, so no
two runs shared the hardware.

| | forward passes | TITAN X (Pascal) | GTX 1080 Ti | RTX PRO 6000 Blackwell | M5 GPU (MPS) | M5 CPU | Linux server CPU |
|---|---|---|---|---|---|---|---|
| `tfidf` | 0 | | | | | 1 s | 5 s |
| `specter2` | `2,000 + 12` ≈ 2,000 | 56 s | 55 s | 18 s | 1 min 14 s | 2 min 30 s | 4 min 40 s |
| `specter2+scincl` | `2 × (2,000 + 12)` ≈ 4,000 | 1 min 45 s | 1 min 41 s | 38 s | 2 min 21 s | 5 min 00 s | 8 min 25 s |
| `rerank` | `0.2 × 2,000 × 12` ≈ 4,800 | 10 min 19 s | 9 min 59 s | 1 min 25 s | 14 min 55 s | _pending_ | _pending_ |

The machines:

- **TITAN X and GTX 1080 Ti:** two 2016-era cards (12 / 11 GB) in a Linux desktop with an i7-6800K
  (6 cores). torch 2.14, CUDA 12.6.
- **RTX PRO 6000 Blackwell:** Server Edition, 96 GB, on a Slurm node with 8 cores of an AMD EPYC 9354
  allocated. torch 2.11, CUDA 12.8.
- **M5:** a MacBook Pro, 4 performance + 6 efficiency cores, a 10-core GPU and 32 GB, on AC power.
  torch 2.15.
- **Linux server:** a Dell PowerEdge R630 with two Xeon E5-2630 v4 (20 cores / 40 threads, 2016) and
  128 GB, no GPU. torch 2.15, CPU build, 20 threads.

`tfidf` has no model, so it always runs on the CPU and takes a few seconds anywhere. The embedding
methods' times are for a first run; re-runs read the cache. `rerank` time scales with the pair
count, so fewer papers or a lower `--positive-frac` shorten it in proportion.

The current datacenter card runs `rerank` about 7× faster than the 2016 cards and `specter2` about
3× faster. The embedding methods gain less because their runs are short, so loading the model and
reading the PDFs take up more of the total.

Core count is not the whole story on a CPU. The M5's 4 performance cores embed nearly twice as fast
as the server's 20 Xeon cores from 2016. The M5's GPU, in turn, is only about 2× faster than its own
CPU.

Memory is not a concern anywhere. `rerank` peaked at 4.5–5.1 GB of GPU memory and under 4 GB of RAM.
Embedding on the M5's CPU peaked at 4.5 GB of RAM.

### Accuracy

There are no true bids to grade against, so the accuracy test uses the papers whose answer is
known: your own. The test takes each of the 12 papers in turn and runs one round:

1. It holds that paper out of the profile.
2. It plants that paper's title and abstract among the ~2,000 submissions.
3. It scores the pool against the other 11 papers.
4. It records where the planted paper ranks.

A good method puts it near the top. These are the results on the round above, using the semantic
score alone (the planted paper has no topics):

| | `tfidf` | `specter2` | `specter2+scincl` | `rerank` | random |
|---|---|---|---|---|---|
| median rank (of ~2,000) | **12** | 32 | 32 | 70 | ~1,000 |
| mean percentile | **5.3%** | 12.6% | 14.8% | 9.4% | 50% |
| mean reciprocal rank | **0.31** | 0.12 | 0.21 | 0.12 | ~0 |
| in the top 1% | **7/12** | 6/12 | 6/12 | 4/12 | ~0 |
| would get a positive bid (top 10%) | **10/12** | 7/12 | 7/12 | 7/12 | ~1 |

Read this with three caveats:

- **It flatters `tfidf`.** A held-out paper shares its author's own vocabulary with the rest of the
  profile: system and dataset names, coined terms. `tfidf` matches those words exactly. Bidding needs
  to find *other people's* papers in your area, and this test doesn't measure that.
- **Off-profile papers rank badly everywhere.** A paper unlike the rest of its author's profile
  ranks low under every method. That is a fact about the profile, not about the methods.
- **It is a small sample.** With 12 papers, a difference of one or two in the counts is noise.

The test is deterministic and gives the same ranks on any device. It took about 20 minutes on one
2016 GPU, mostly `rerank`.

### Reproducibility

Fully deterministic: the same papers, `topic_interests.csv` and CSV give identical bids.

**Across devices on one machine**, the bids were identical in every measurement above. Every method
gave the same bid for every one of the ~2,000 submissions on both NVIDIA cards, and on the M5's GPU
and CPU. A GPU rounds floating-point math differently from a CPU, so scores differ in their last
digits. Bids are integers set by rank, so a difference only shows if it swaps two submissions that
are tied to that precision.

**Across machines**, the two NVIDIA machines and the CPU-only Linux server all matched each other
exactly, for every method. That held despite different GPU generations (Pascal and Blackwell), GPU
against CPU, and different torch and CUDA versions.

The Mac did not match at first, because it had an older **pypdf** (7.0.1 against 7.1.1). The two
versions pull slightly different text out of the same PDFs, so the profile changes. That shifted some
bids by a few points, while the set of positive bids stayed almost the same. With the same
pypdf version the bids were identical again. To get the same bids on several machines, install the
same pypdf version on all of them.

If you need a guarantee rather than a measurement, pass `--device cpu` on every machine. The
embedding caches don't record the device, so a cache built on one device is reused as-is on another.

---

## Development

```bash
python3 -m unittest discover -s tests
```

Stdlib `unittest`, so there's nothing extra to install. The suite covers the scoring contract
(`--positive-frac` hitting its target, bids spanning the full range, tie handling), the input
validation (required columns, unparsable interests, unrated topics, PDFs without text), the
vocabulary guards, and the change-summary baseline logic.

---

## Acknowledgments

Built with the assistance of [Claude](https://www.anthropic.com/claude) (Anthropic).
