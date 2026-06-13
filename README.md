# hotcrp-bidding — reviewer bidding toolkit

A small, self-service toolkit for HotCRP-run reviewing rounds (any conference). **Two commands**
turn your reviewing interests into bids:

1. `make_topic_interests.py` — list the conference's topics for you to rate.
2. `score_bids.py` — build your profile from those ratings + your papers, and fill every submission's
   bid (HotCRP uses roughly **−20…20**).

Runs locally on the Python standard library (+ `pypdf`). No model, no network.

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

This uses `topic_interests.csv` (override with `--topic-interests`), mines positive keywords from
`papers_pdf/` (**at least 5 unique PDFs are required, or it stops**), writes
`reviewer-expertise-profile.json`, then fills the `preference` column — targeting a positive bid on
~10% of papers (`--positive-frac`, default 0.1). **By default it deletes the
original input and keeps only the scored output, which has the `abstract` column stripped**
(`paper, title, preference, topics`) — pass `--keep-original` to keep the input. It auto-detects columns
(full-text scoring with `abstract`/`topics`, else noisier title-only) and prints a histogram report;
`--report` saves it.

---

## Files

| File | Role |
|---|---|
| `make_topic_interests.py` | Creates a blank `topic_interests.csv` (every topic at 0) from the preferences CSV. Stdlib. |
| `score_bids.py` | Builds the profile from your interests + papers and fills the bids. Needs `pypdf` (`PyYAML` optional). |
| `config.yaml` | Scoring parameters (`bid_max`, caps, curve, guardrails). Edit to taste. |
| `topic_interests.csv` | **you edit** — `topic,interest` on a **-2..2** scale. Made by `make_topic_interests.py`. |
| `papers_pdf/` | your papers as PDFs (**≥5 unique**), used for keyword mining. |
| `reviewer-expertise-profile.json` | *(generated — don't hand-edit)* the profile, saved for inspection. |
| `revprefs.csv` | the round's submissions (HotCRP export: `paper, title, preference[, abstract, topics]`). Deleted once scored, unless `--keep-original`. |
| `revprefs.scored.csv` | the scored output — **abstracts removed** — this is what you upload back to HotCRP. |

(CSV files and `papers_pdf/` are git-ignored — they hold conference-confidential data.)

---

## Requirements

- Python 3.7+
- `pypdf` — to read your PDFs: `pip install pypdf`
- `PyYAML` — optional; `score_bids.py` reads `config.yaml` with a built-in fallback parser if it isn't installed.

---

## How a bid is computed

For each paper (parameters in `config.yaml`):

1. **Topic base** — `0.6·max + 0.4·mean` of your topic interests for the paper's tags (each interest is mapped ×10 onto the scoring scale), then scaled.
2. **Keyword adjustment** — sum of mined positive-keyword weights found in the title (counted twice) and abstract, capped.
3. **Combine + compress** the top tail so the best-fit papers separate instead of all pinning at the ceiling.
4. **Guardrails** — out-of-scope / strong in-scope nudges.
5. **Rescale, round, clamp** to the output range — **[-20, 20]** by default (`bid_max` in `config.yaml`).

The only judgment input is `topic_interests.csv`; everything else is mechanical and in `config.yaml`.

---

## Customizing

- **Re-rate a topic** → edit its `interest` in `topic_interests.csv`, then re-run `score_bids.py`.
- **Target how many papers you bid positively on** → `--positive-frac F` (0–1). **Defaults to `0.1`** (~10%); set e.g. `--positive-frac 0.3` for ~30%. It thresholds at the target quantile and rescales each side to the full range, so ~F end up positive **and** your strongest papers still reach ±`bid_max` (the run reports the achieved fraction, within ±10 points).
- **Change the curve** (caps, compression, guardrails) → edit `config.yaml`.
- **Change the output range** → set `bid_max` in `config.yaml` (default 20, max 100). The scorer runs on a fixed ±`ref_max` (20) reference span and *linearly rescales* the final bid to ±`bid_max`. (`bid_max` is an integer in `[1, bid_limit]`; `bid_limit` defaults to HotCRP's 100.)

## Reproducibility

Fully deterministic: same papers + same `topic_interests.csv` + same CSV → identical bids.
