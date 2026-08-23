# autobidder — reviewer bidding toolkit

Fill in your HotCRP reviewer bids automatically. Rate the conference's topics once, point the tool at
your own papers, and it scores every submission by how well it matches your work. Everything runs
locally — no submission data leaves your machine.

**Status:** beta — `v0.3.0-beta`.

---

## Quickstart

```bash
pip install scikit-learn pypdf
```

Then, once per reviewing round:

**1. Put your papers in `papers_pdf/`.** At least 5 PDFs, each with a real text layer — scans and
image-only files are skipped, and the run stops if fewer than 5 usable ones remain. Submissions get
matched against these.

**2. Export the submissions from HotCRP** as `revprefs.csv`, with the `abstract` and `topics` columns
included. See [The HotCRP round trip](#the-hotcrp-round-trip) for the click-path.

**3. Rate the conference's topics:**

```bash
python3 make_topic_interests.py revprefs.csv
```

Open `topic_interests.csv` and set each topic's `interest` to an integer from **−2 to +2** — `2` = very
high, `1` = high, `0` = neutral, `-1` = low, `-2` = very low. Leave anything you don't care about at
`0`. This is the only judgment call in the whole tool.

**4. Score the submissions and fill your bids:**

```bash
python3 score_bids.py revprefs.csv
```

This writes `revprefs.scored.<timestamp>.csv` with the `preference` column filled in, and prints a
histogram of the bids it chose.

**5. Upload that scored CSV back to HotCRP.** Done.

> ⚠️ **`score_bids.py` deletes `revprefs.csv` by default.** The export contains every submission's
> abstract; the scored output has them stripped, so the confidential copy doesn't linger on disk. Pass
> `--keep-original` to keep it — useful if you want to re-run with different settings.

#### Pick a matching method (optional)

Step 4 used `tfidf`, the default, which needs nothing beyond the two packages you already
installed. Two more accurate options are one flag away:

| | Matches on | Cost |
|---|---|---|
| `--method tfidf` *(default)* | shared wording | none — already installed |
| `--method specter2` | **meaning**, so it catches related work phrased differently | `pip install torch transformers adapters` + one-time download |
| `--method rerank` | **meaning, read pairwise** — the most precise | `pip install sentence-transformers` + one-time download |

Only the similarity step changes, so you can switch between rounds and the bids stay comparable.
Details in [Matching methods](#matching-methods).

Variations you're most likely to want:

```bash
python3 score_bids.py revprefs.csv --keep-original       # keep the abstract-laden input
python3 score_bids.py revprefs.csv --positive-frac 0.3   # bid positively on ~30%, not ~10%
python3 score_bids.py revprefs.csv --zero-below 5        # only surface bids of +5 or better
```

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
| `-o PATH` | timestamped | Output CSV path. |
| `--report PATH` | — | Also write the histogram report to a file. |
| `--quiet` | off | Don't print the report. |
| `--topic-interests PATH` | `topic_interests.csv` | Where your topic ratings live. |
| `--pdfs DIR` | `papers_pdf` | Folder of your paper PDFs. |
| `--profile-out PATH` | timestamped | Where to save the profile summary JSON. |
| `--rerank-topn N` | auto | `rerank` only: how many TF-IDF candidates the cross-encoder rescores. |
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
| **`rerank`** | `pip install sentence-transformers` | slowest on CPU | The most precise. TF-IDF shortlists, then a cross-encoder rescores the shortlist. |

`specter2` and `rerank` each download their model once, then run offline and deterministically.

**`specter2`** uses AllenAI SPECTER2, trained specifically for paper-to-paper similarity. Embeddings
are cached in `.specter2_cache.npz` keyed by the text they came from, so re-runs only embed what's new
— a fully-cached re-run doesn't even load the model. Override the path with `--emb-cache`.

**`rerank`** uses TF-IDF to shortlist the top-N submissions, then has a local cross-encoder rescore
only those. A cross-encoder reads each *(your paper, submission)* pair **together** and judges their
relevance directly, rather than embedding each text alone and comparing vectors — more precise, but too
slow for a whole pool, hence the shortlist. `--rerank-topn` defaults to auto-scaling with the pool and
your target so the shortlist always covers the positive bid band; override the model with
`--rerank-model`.

Each candidate is scored against **every** one of your papers, so the real work is
`candidates × papers` pairs — the run prints that number before loading the model, since
`--rerank-topn` scales with `--positive-frac` and the cost can grow via a flag you changed for another
reason. Each pair gets a joint budget of `--rerank-max-length` tokens (default **1024**, roughly 300
words per side, enough for any realistic abstract). The model accepts up to 8192, but cost grows faster
than length — 2048 runs several times slower for headroom no abstract uses.

---

## Files

| File | Role |
|---|---|
| `make_topic_interests.py` | Creates a blank `topic_interests.csv` (every topic at 0) from the preferences CSV. Stdlib only. |
| `score_bids.py` | Scores submissions by similarity to your papers (+ interests) and fills the bids. |
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

## Acknowledgments

Built with the assistance of [Claude](https://www.anthropic.com/claude) (Anthropic).
