# autobidder — reviewer bidding toolkit

Fill in your HotCRP reviewer bids automatically. Rate the conference's topics once, point the tool at
your own papers, and it scores every submission by how well it matches your work. Everything runs
locally — no submission data leaves your machine.

**Version:** `v0.5.0`.

---

## Quickstart

```bash
pip install scikit-learn pypdf
```

Then, once per reviewing round:

**1.** Put at least 5 of your own papers in `papers_pdf/`, as text-based PDFs — scans don't work.

**2.** Export the submissions from HotCRP as `revprefs.csv`, including the `abstract` and `topics`
columns ([click-path](#the-hotcrp-round-trip)).

**3.** Generate the topic list, then set each topic's `interest` in `topic_interests.csv` —
`+2` want, `+1` happy to, `0` neutral, `-1` rather not, `-2` avoid. The only judgment call in the tool:

```bash
python3 make_topic_interests.py revprefs.csv
```

**4.** Score every submission and fill in your bids. `--method` chooses how submissions are matched
against your papers — only that step differs, so bids stay comparable across methods:

```bash
python3 score_bids.py revprefs.csv                    # tfidf (default): shared wording, nothing to install
python3 score_bids.py revprefs.csv --method specter2  # meaning, not just wording  (+ torch transformers adapters)
python3 score_bids.py revprefs.csv --method rerank    # most precise, reads pairs   (+ sentence-transformers, beta)
```

`specter2` and `rerank` download their model once, then run offline — see
[Matching methods](#matching-methods) for what each one does.

**5.** Upload the `revprefs.scored.<timestamp>.csv` it wrote back to HotCRP. Done.

> ⚠️ Step 4 **deletes `revprefs.csv`** — it holds every submission's abstract, and the scored output
> has them stripped so the confidential copy doesn't linger. Pass `--keep-original` to keep it.

By default this bids positively on ~10% of papers; `--positive-frac` and the rest of the flags are in
[Options](#options).

---

## The HotCRP round trip

**Export.** Once submissions are visible, HotCRP shows a **Review preferences** link on the home page,
listing every submission with a preference box. Select all papers and use the page's **Download**
action → *Review preferences*. The default columns are `paper, title, preference`; add the **abstract**
and **topics** columns to the view first (the **Show** menu / column options on the search bar → enable
*Abstract* and *Topics*), then download. Save it as `revprefs.csv`.

`score_bids.py` checks the columns before it does anything else:

| Column | |
|---|---|
| `paper` | **required** — HotCRP matches rows by paper id on upload, so an export without it can't be uploaded at all |
| `preference` | **required** — this is the column that gets filled in |
| `abstract` | strongly wanted — without it, matching falls back to titles alone |
| `topics` | strongly wanted — without it your interests can't be applied, leaving `interest_weight` of every score a constant 0 |

The run warns when `abstract` or `topics` is missing rather than failing, so a title-only round is
possible — just much weaker.

**Upload.** Upload the scored CSV on the same **Review preferences** page via its **Upload** action (or
**Assignments → Upload**). HotCRP matches each row by its `paper` id and reads the `preference` column;
the other columns are ignored.

> Labels vary a little between HotCRP versions. If your site's controls differ, the rule is: get a CSV
> that includes `paper, title, preference, abstract, topics`, and upload one with `paper` +
> `preference`.

---

## Options

### Flags

| Flag | Default | What it does |
|---|---|---|
| `--positive-frac F` | `0.1` | Fraction of papers to bid positively on (0–1). See below. |
| `--zero-below N` | — | Floor every bid below `N` to `0` (neutral). See below. |
| `--keep-original` | off | Don't delete the input CSV after scoring. |
| `--method M` | `tfidf` | Similarity method: `tfidf`, `specter2`, or `rerank`. See [Matching methods](#matching-methods). |
| `-o`, `--output PATH` | timestamped | Output CSV path. |
| `--report PATH` | — | Also write the histogram report to a file. |
| `--quiet` | off | Don't print the report. |
| `--topic-interests PATH` | `topic_interests.csv` | Where your topic ratings live. |
| `--pdfs DIR` | `papers_pdf` | Folder of your paper PDFs. |
| `--config PATH` | `config.yaml` | Scoring parameters file (defaults to the one beside the script). |
| `--profile-out PATH` | timestamped | Where to save the profile summary JSON. |
| `--rerank-frac F` | `--positive-frac` + 0.10 | `rerank` only: fraction of submissions the cross-encoder rescores. |
| `--rerank-model ID` | `BAAI/bge-reranker-v2-m3` | `rerank` only: cross-encoder model id. |
| `--rerank-max-length N` | `1024` | `rerank` only: joint token budget per (paper, submission) pair. |
| `--emb-cache PATH` | `.specter2_cache.npz` | `specter2` only: embedding cache file. |

**`--positive-frac F`** — how many papers you want to end up wanting. It puts the threshold just below
the target count and rescales each side to the full range, so that many end up positive **and** your
strongest papers still reach ±`bid_max`. The run reports the fraction it achieved; ties at the
threshold are the one thing that can hold it below target, and it warns if the miss exceeds 10 points.

**`--zero-below N`** — sets every bid strictly below `N` to `0`, negatives included. `--zero-below 5`
keeps only bids of +5 or better, so you surface just the papers you actively want. Applied after
`--positive-frac` and reported separately: the run states what `--positive-frac` achieved on its own
terms, then how many bids were zeroed and how many positives remain. `N` may not exceed `bid_max`,
since nothing would survive it.

**Re-rating a topic** — edit its `interest` in `topic_interests.csv` and re-run `score_bids.py`.
Nothing else needs rebuilding.

### config.yaml

| Key | Default | What it does |
|---|---|---|
| `interest_weight` | `0.35` | How much your topic ratings steer the bid vs. similarity to your papers. `0` = pure paper-similarity, `1` = pure topic interests. |
| `sem_gain` | `9.0` | Shapes the rank curve. `9` is linear; higher pushes mid-rank papers toward the extremes, i.e. more separation between near-miss and on-target. |
| `bid_max` | `20` | The output range: bids are written in `[-bid_max, bid_max]`. An integer in `[1, bid_limit]`. |
| `bid_limit` | `100` | Validation ceiling for `bid_max` — HotCRP's own maximum. |
| `ref_max` | `20` | Internal reference span the score is computed on. **Has no effect on your bids** — both halves of the blend scale with it and the final mapping depends only on ordering. Set `bid_max` instead. |

---

## Matching methods

All three do the same thing — score each submission by similarity to your papers — and differ only in
how that similarity is computed. Everything downstream (the interest blend, `--positive-frac`, the
report) is identical, so you can switch freely between rounds or re-run with `--keep-original` and
compare.

| `--method` | Needs | Speed | Good for |
|---|---|---|---|
| **`tfidf`** *(default)* | nothing extra | fast | Shared vocabulary. Light, fully offline, no model download. |
| **`specter2`** | `pip install torch transformers adapters` | slower on CPU | Meaning rather than wording — catches related work phrased differently. |
| **`rerank`** *(beta)* | `pip install sentence-transformers` | slowest on CPU | The most precise. TF-IDF shortlists, then a cross-encoder rescores the shortlist. |

`specter2` and `rerank` each download their model once, then run offline and deterministically. That
first download happens *inside* a scoring run, which can leave it apparently doing nothing for several
minutes — so fetch it separately instead:

```bash
python3 fetch_models.py rerank          # ~2.3 GB
python3 fetch_models.py specter2        # ~0.5 GB, plus its adapter
python3 fetch_models.py all --check     # what's already cached? downloads nothing
```

Models go to the shared HuggingFace cache (`~/.cache/huggingface`, or `HF_HOME`), which is where
`score_bids.py` looks. `--check` exits non-zero if anything is missing, so it works in a pre-flight
script.

**`specter2`** uses AllenAI SPECTER2, trained specifically for paper-to-paper similarity. Embeddings
are cached in `.specter2_cache.npz` keyed by the text they came from, so re-runs only embed what's new
— a fully-cached re-run doesn't even load the model. Override the path with `--emb-cache`.

**`rerank`** *(beta)* is a two-stage pipeline. TF-IDF ranks the whole pool cheaply, the top slice
of that ranking becomes a **shortlist**, and a local cross-encoder rescores only the shortlist. The
cross-encoder reads each *(your paper, submission)* pair **together** and judges relevance directly,
rather than embedding each text alone and comparing vectors — more accurate, but far too slow to run on
every submission, which is the whole reason for the shortlist. Reranked submissions are then placed
above the rest, which keep their TF-IDF order, and the usual normalize/blend/map steps run unchanged.

### Sizing the shortlist

One setting controls this: **`--rerank-frac`**, the fraction of submissions that get cross-encoded.

```bash
python3 score_bids.py revprefs.csv --method rerank --rerank-frac 0.2   # rescore the top 20%
```

It defaults to **`--positive-frac` + 0.10** — ten points of headroom above whatever you're bidding on.
So the default at `--positive-frac 0.1` is `0.2`, and at `0.25` it's `0.35`.

Two things follow from how the pipeline works:

- **It can't go below `--positive-frac`.** Positive bids come from the reranked slice, so a shortlist
  smaller than your positive target would leave some positive bids on submissions the cross-encoder
  never scored. Set it lower and the run stops and tells you the minimum.
- **Above ~0.5 it stops being worth it.** You're rescoring most of the pool, so the two-stage saving
  has largely gone — `--method specter2` is the better tool at that point, and the run says so.

The headroom exists so the cross-encoder can promote submissions TF-IDF ranked too low. It's ten points
*of the pool* rather than a multiple of your target, because what it absorbs is disagreement between the
two rankings — a property of the pool, not of how many papers you want. A multiple would misbehave at
both ends: almost no slack at `--positive-frac 0.01`, far more than any reordering needs at `0.5`. The
`0.10` itself is a starting value chosen on those grounds, not a measured optimum.

### What a run costs

Every shortlisted submission is scored against **every** one of your papers, so the work is
`shortlist × papers` pairs. The run prints that number before it loads the model:

```
rerank: 505 candidate(s) of 1442 submissions x 13 paper(s) = 6565 cross-encoder pair(s), max_length 1024
```

If that's more than you want to wait for, stop and lower `--rerank-frac`. Each pair gets a joint budget
of `--rerank-max-length` tokens (default **1024**, roughly 300 words per side, enough for any realistic
abstract); the model accepts up to 8192, but cost grows faster than length.

Fetch the model first with `python3 fetch_models.py rerank`, or the first run spends its opening minutes
downloading 2.3 GB with nothing to show.

---

## Files

| File | Role |
|---|---|
| `make_topic_interests.py` | Creates a blank `topic_interests.csv` (every topic at 0) from the preferences CSV. Stdlib only. |
| `score_bids.py` | Scores submissions by similarity to your papers (+ interests) and fills the bids. |
| `fetch_models.py` | Pre-downloads the `specter2` / `rerank` models so a scoring run never stalls on it. |
| `config.yaml` | Scoring parameters. Edit to taste. |
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
own run, so when bids move you can compare the two runs' settings to see why. The baseline is the newest run
stamped earlier than this one — or, with a fixed `-o`, the file at that path that this run replaces,
read before it's overwritten. The first run just notes there's nothing to compare against.

(CSV files and `papers_pdf/` are git-ignored — they hold conference-confidential data.)

---

## Requirements

- **Python 3.7+** for the scripts themselves; in practice your `scikit-learn` build sets the floor
  (recent versions need 3.10+).
- `scikit-learn` and `pypdf` — required. `pip install scikit-learn pypdf`
- `PyYAML` — optional; `score_bids.py` reads `config.yaml` with a built-in fallback parser without it.
- Per-method extras are listed under [Matching methods](#matching-methods).

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
   revprefs.csv  ─────────────────►  score_bids.py  ─────────►  filled `preference` column
                                        ▲
                                   config.yaml
```

`score_bids.py` builds your profile on the fly — the topic interests differ every conference, so
there's nothing reusable to build separately — and saves a summary to
`reviewer-expertise-profile.<ts>.json` for inspection.

### Building your profile

Your PDFs in `papers_pdf/` are read (first ~3 pages each — title, abstract, and intro, where the
topical vocabulary lives) and lightly cleaned (lower-cased, words split across line breaks rejoined).
The vocabulary and weighting (**TF-IDF** — term frequency–inverse document frequency) are learned from
the conference's **submissions**, and your papers are projected into that same space:

- **unigrams + bigrams**, so phrases like *differential privacy* or *membership inference* count as units;
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
   bigram) can't spike it on its own. This is the only step `--method` changes.
2. **Normalize.** Cosine similarities are small, bunched, and skewed (TF-IDF piles near zero; SPECTER2
   sits high even for unrelated papers), so each submission is **rank/quantile-transformed** across the
   pool onto a ±`ref_max` range — "where does this rank among your matches this year." `sem_gain` shapes
   the curve (9 = linear; higher pushes mid-rank papers toward the extremes). Unlike a z-score, this
   doesn't depend on the pool's spread, so the blend in step 3 behaves the same across venues.
3. **Blend with interests.** `(1 − interest_weight)·similarity + interest_weight·topic`, where `topic` is
   `0.6·max + 0.4·mean` of your −2..2 interests (×10) for the submission's topic tags. Default
   `interest_weight` 0.35 — similarity leads, your topic ratings steer.
4. **Map to bids.** Put the threshold just below the `--positive-frac` count and rescale each side to
   `[-bid_max, bid_max]`, so that many end up positive **and** your strongest matches still reach
   ±`bid_max`.

The only judgment input is `topic_interests.csv`; everything else is mechanical and in `config.yaml`.

### Inside the matching methods

**`specter2`** embeds both sides as `title [SEP] abstract`, the form the model was trained on — your
papers are parsed down to their own title+abstract, falling back to the raw page text only when no
abstract parses, rather than being fed the three-page read.

**`rerank`** shortlists the top-N submissions by TF-IDF, then scores each candidate against your papers
with the cross-encoder and aggregates with a top-3 mean. The shortlist size auto-scales as
`max(150, 3 × --positive-frac × pool size)`, so it always covers the positive bid band — reranked
candidates sort above the rest, so a positive bid could otherwise land on a paper the cross-encoder
never saw. TF-IDF cosines and cross-encoder logits aren't comparable, so they're never compared: each
group is rank-transformed within itself and the reranked candidates are placed as a band strictly above
the non-candidates, which keep their TF-IDF order. The normalize/blend/map steps then run unchanged.

---

## Reproducibility

Fully deterministic: same papers + same `topic_interests.csv` + same CSV → identical bids.

## Development

```bash
python3 -m unittest discover -s tests
```

Stdlib `unittest`, so there's nothing extra to install. The suite covers the scoring contract
(`--positive-frac` hitting its target, bids spanning the full range, tie handling), the input
validation (required columns, unparsable interests, unrated topics, PDFs without text), the
vocabulary guards, and the change-summary baseline logic.

## Acknowledgments

Built with the assistance of [Claude](https://www.anthropic.com/claude) (Anthropic).
