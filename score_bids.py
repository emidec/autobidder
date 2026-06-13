#!/usr/bin/env python3
"""
score_bids.py - score conference submissions by similarity to your work, and fill bids.

Matching is SEMANTIC, not keyword-based. Your papers (papers_pdf/) and each submission's
title+abstract are turned into TF-IDF vectors; every submission is scored by cosine
similarity to your most-similar paper. Your -2..2 topic interests then steer that score
(a blend). Finally the bids are written in [-bid_max, bid_max], targeting a chosen
fraction of positive bids without squashing your top papers.

PIPELINE
    python3 make_topic_interests.py revprefs.csv       # 1. template (topics at 0)
    #   ... set your interest per topic in topic_interests.csv (-2..2) ...
    python3 score_bids.py revprefs.csv                 # 2. score + fill bids (uses topic_interests.csv)

HOW A BID IS COMPUTED (parameters in config.yaml)
    sem        = max cosine TF-IDF similarity of the submission to any of your papers,
                 z-scored across all submissions and scaled by sem_gain -> [-ref_max, ref_max]
    topic_base = 0.6*max + 0.4*mean of your interests (x10) for the submission's topic tags
    value      = (1 - interest_weight)*sem + interest_weight*topic_base       (in [-ref_max, ref_max])
    bid        = value mapped to [-bid_max, bid_max] so ~--positive-frac of papers are positive
                 (threshold at that quantile, each side rescaled to the full range)

By default the abstract-laden input is deleted and an abstract-free scored CSV is written;
--keep-original keeps it. At least 5 unique PDFs in papers_pdf/ are required.

Dependencies: scikit-learn (pip install scikit-learn) and pypdf (pip install pypdf).
PyYAML is optional (config.yaml has a built-in fallback parser).

--method specter2 swaps TF-IDF for AllenAI SPECTER2 neural embeddings (semantic matching
beyond shared wording). It needs extra packages and a one-time model download:
    pip install torch transformers adapters
"""

import argparse
import collections
import csv
import datetime
import glob
import hashlib
import json
import math
import os
import re
import sys

try:
    import yaml  # optional
except ImportError:
    yaml = None

CONFIG_KEYS = ("bid_max", "bid_limit", "ref_max", "interest_weight", "sem_gain")
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
DEFAULT_PDF_DIR = "papers_pdf"
MIN_UNIQUE_PDFS = 5
DEFAULT_PROFILE = "reviewer-expertise-profile.json"
INTEREST_TO_AFFINITY = 10   # map a -2..2 interest onto the -20..20 reference scale

csv.field_size_limit(10 ** 7)


# ------------------------------- config --------------------------------------
def _parse_simple_yaml(text):
    out = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = (s.strip() for s in line.split(":", 1))
        if not val:
            continue
        try:
            out[key] = int(val)
        except ValueError:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val.strip("\"'")
    return out


def load_config(path):
    if not path or not os.path.exists(path):
        sys.exit("config: file not found: %s" % path)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh.read()) if yaml else _parse_simple_yaml(fh.read())
    if not isinstance(cfg, dict) or not cfg:
        sys.exit("config: %s is empty or not a key/value mapping" % path)
    missing = set(CONFIG_KEYS) - set(cfg)
    unknown = set(cfg) - set(CONFIG_KEYS)
    if missing:
        sys.exit("config: %s is missing key(s): %s" % (path, ", ".join(sorted(missing))))
    if unknown:
        sys.exit("config: unknown key(s) in %s: %s" % (path, ", ".join(sorted(unknown))))

    g = globals()
    g["BID_MAX"]         = cfg["bid_max"]
    g["BID_LIMIT"]       = cfg["bid_limit"]
    g["_REF_MAX"]        = cfg["ref_max"]
    g["INTEREST_WEIGHT"] = cfg["interest_weight"]
    g["SEM_GAIN"]        = cfg["sem_gain"]

    if not (isinstance(BID_MAX, int) and 1 <= BID_MAX <= BID_LIMIT):
        sys.exit("config: bid_max must be an int in [1, %d], got %r" % (BID_LIMIT, BID_MAX))
    if not (0.0 <= INTEREST_WEIGHT <= 1.0):
        sys.exit("config: interest_weight must be in [0, 1], got %r" % INTEREST_WEIGHT)
    g["BID_MIN"]  = -BID_MAX
    g["_REF_MIN"] = -_REF_MAX
    g["SCALE"]    = BID_MAX / float(_REF_MAX)


# ------------------------------- helpers -------------------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rnd(x):
    """Round half away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


# ------------------------------- PDF folder ----------------------------------
def find_unique_pdfs(folder):
    if not os.path.isdir(folder):
        return None
    cand = set()
    for pat in ("*.pdf", "*.PDF"):
        cand.update(glob.glob(os.path.join(folder, pat)))
    seen = collections.OrderedDict()
    for fn in sorted(cand):
        try:
            with open(fn, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        if not data.startswith(b"%PDF"):
            continue
        seen.setdefault(hashlib.sha256(data).hexdigest(), fn)
    return list(seen.values())


def require_pdfs(folder, minimum):
    pdfs = find_unique_pdfs(folder)
    if pdfs is None:
        sys.exit("ERROR: PDF folder not found: '%s' (put at least %d of your papers in it)."
                 % (folder, minimum))
    if len(pdfs) < minimum:
        sys.exit("ERROR: '%s' has %d unique PDF(s); at least %d required "
                 "(dupes/non-PDFs don't count)." % (folder, len(pdfs), minimum))
    return pdfs


def read_pdf_texts(pdf_paths, pages=3):
    try:
        from pypdf import PdfReader
    except Exception:
        sys.exit("pypdf is required to read your PDFs: pip install pypdf")
    import warnings
    warnings.filterwarnings("ignore")
    out = []
    for fn in pdf_paths:
        try:
            out.append("".join((p.extract_text() or "") for p in PdfReader(fn).pages[:pages]))
        except Exception:
            out.append("")
    return out


# ------------------------------ topic interests ------------------------------
def read_topic_interests(path):
    """topic,interest CSV; interest is an integer in [-2, 2] (out-of-range clamped)."""
    if not os.path.exists(path):
        sys.exit("ERROR: topic interests file not found: '%s'\n"
                 "Create one with:  python3 make_topic_interests.py revprefs.csv" % path)
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    rows = list(csv.DictReader(lines))
    if not rows or "topic" not in rows[0] or "interest" not in rows[0]:
        sys.exit("ERROR: '%s' must have columns: topic,interest "
                 "(make one with: python3 make_topic_interests.py revprefs.csv)" % path)
    out = collections.OrderedDict()
    for r in rows:
        t = (r.get("topic") or "").strip()
        if not t:
            continue
        v = (r.get("interest") or "").strip()
        try:
            iv = int(round(float(v))) if v else 0
        except ValueError:
            iv = 0
        out[t] = max(-2, min(2, iv))
    if not out:
        sys.exit("ERROR: no topics found in %s." % path)
    return out


# --------------------------- semantic similarity -----------------------------
# boilerplate / PDF-extraction noise to drop on top of sklearn's English stop words
_EXTRA_STOP = ("et al fig figs figure figures table tables section sec eq ref refs arxiv doi http https "
               "www com org edu university email abstract introduction conclusion ing like also however "
               "thus paper papers results result show shows shown using used use based proposed approach "
               "method methods work works can may given ieee acm usenix proceedings conference workshop "
               "preprint appendix springer lncs cite vol pp "
               # generic filler that isn't part of any meaningful phrase
               "different example number overall fact possible previous related follow focus particular "
               "various general specific order find found consider considered note specifically "
               "way similar instance main generally typically essentially known popular uk us").split()


def _clean(text):
    """Lowercase + join words split across line breaks (so 'learn-\\ning' -> 'learning')."""
    return text.replace("-\n", "").replace("\n", " ").lower()


SPECTER2_MODEL = "allenai/specter2_base"
SPECTER2_ADAPTER = "allenai/specter2"


def _tfidf_fit(paper_texts, sub_texts):
    """Fit TF-IDF on the CONFERENCE submissions and project your papers into that space.

    Returns (S, P, vectorizer). Fitting on the submission pool means terms specific to your
    papers but absent from it (your name, affiliation, venue boilerplate) never enter the
    vocabulary, and domain-generic words get low IDF weight.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    except Exception:
        sys.exit("scikit-learn is required: pip install scikit-learn")
    import warnings
    warnings.filterwarnings("ignore")
    stop = list(ENGLISH_STOP_WORDS.union(_EXTRA_STOP))
    vec = TfidfVectorizer(preprocessor=_clean, stop_words=stop,
                          token_pattern=r"(?u)\b[a-z][a-z]+\b",   # alphabetic tokens, 2+ chars
                          ngram_range=(1, 2), min_df=2, max_df=0.3, sublinear_tf=True)
    S = vec.fit_transform(sub_texts)
    P = vec.transform(paper_texts)
    return S, P, vec


def _tfidf_scores(paper_texts, sub_texts):
    """Default: TF-IDF cosine similarity. Returns (best_per_submission, top_terms)."""
    from sklearn.metrics.pairwise import cosine_similarity
    S, P, vec = _tfidf_fit(paper_texts, sub_texts)
    best = cosine_similarity(S, P).max(axis=1)   # similarity to the closest of your papers
    return best, corpus_top_terms(vec, P)


def _specter2_scores(paper_texts, sub_texts):
    """--method specter2: neural scientific-paper embeddings (AllenAI SPECTER2).

    The embeddings drive the *matching*; the profile's top-terms summary is still derived from
    TF-IDF (cheap, always available) so the saved JSON stays readable either way.
    """
    try:
        import torch
        from transformers import AutoTokenizer
        try:
            from adapters import AutoAdapterModel                 # adapters >= 0.x
        except Exception:
            from transformers.adapters import AutoAdapterModel    # older transformers
    except Exception:
        sys.exit("SPECTER2 needs extra packages:\n"
                 "    pip install torch transformers adapters\n"
                 "It downloads the model once on first run (needs network); afterwards it runs offline.")
    import warnings
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    warnings.filterwarnings("ignore")
    tok = AutoTokenizer.from_pretrained(SPECTER2_MODEL)
    model = AutoAdapterModel.from_pretrained(SPECTER2_MODEL)
    model.load_adapter(SPECTER2_ADAPTER, source="hf", load_as="proximity", set_active=True)
    model.eval()

    def embed(texts, batch=16):
        out = []
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            with torch.no_grad():
                rep = model(**enc).last_hidden_state[:, 0, :]     # CLS-token embedding
            out.append(rep.cpu().numpy())
        return np.vstack(out)

    best = cosine_similarity(embed(sub_texts), embed(paper_texts)).max(axis=1)
    _, P, vec = _tfidf_fit(paper_texts, sub_texts)    # readable top-terms summary for the profile
    return best, corpus_top_terms(vec, P)


def semantic_scores(paper_texts, sub_texts, method="tfidf"):
    """Per-submission similarity to your most-similar paper, by the chosen method."""
    if method == "specter2":
        return _specter2_scores(paper_texts, sub_texts)
    return _tfidf_scores(paper_texts, sub_texts)


def corpus_top_terms(vec, P, topn=40):
    """Highest mean TF-IDF terms across your papers - a readable summary of your profile."""
    mean = P.mean(axis=0)
    arr = mean.A1 if hasattr(mean, "A1") else (mean.toarray().ravel() if hasattr(mean, "toarray") else mean.ravel())
    terms = vec.get_feature_names_out()
    idx = arr.argsort()[::-1][:topn]
    return [str(terms[i]) for i in idx]


# ------------------------- target positive fraction --------------------------
def bids_for_positive_fraction(values, target):
    """Map continuous reference scores to final bids so ~`target` of papers are positive,
    putting the threshold at the target quantile and rescaling EACH SIDE to the full output
    range, so the strongest paper reaches +bid_max and the weakest -bid_max."""
    n = len(values)
    if n == 0:
        return []
    sv = sorted(values)
    idx = min(max(int(round((1.0 - target) * n)), 0), n - 1)
    tau = sv[idx]
    hi = (sv[-1] - tau) or 1.0
    lo = (tau - sv[0]) or 1.0
    out = []
    for v in values:
        if v > tau:
            out.append(int(clamp(1 + rnd((v - tau) / hi * (BID_MAX - 1)), 1, BID_MAX)))
        else:
            out.append(int(clamp(rnd((v - tau) / lo * BID_MAX), BID_MIN, 0)))
    return out


# ------------------------------- reporting -----------------------------------
def make_report(rows, mode):
    vals = [int(r["preference"]) for r in rows]
    dist = collections.Counter(vals)
    lines = ["Reviewer bidding report", "=" * 60]
    lines.append("mode            : %s" % mode)
    lines.append("papers scored   : %d" % len(rows))
    lines.append("mean bid        : %+.2f" % (sum(vals) / len(vals)))
    lines.append("want (>0)       : %d" % sum(v > 0 for v in vals))
    lines.append("neutral (=0)    : %d" % sum(v == 0 for v in vals))
    lines.append("avoid (<0)      : %d" % sum(v < 0 for v in vals))
    strong = int(round(BID_MAX * 0.75))
    lines.append("strong + (>=%+d) : %d" % (strong, sum(v >= strong for v in vals)))
    lines.append("strong - (<=%+d) : %d" % (-strong, sum(v <= -strong for v in vals)))
    lines.append("")
    lines.append("distribution (bid: count)")
    mx = max(dist.values()) if dist else 1
    for b in range(BID_MAX, BID_MIN - 1, -1):
        c = dist.get(b, 0)
        if c:
            lines.append("  %+3d | %-50s %d" % (b, "#" * int(50 * c / mx), c))
    srt = sorted(rows, key=lambda r: -int(r["preference"]))
    lines += ["", "top 10 (highest bids)"]
    for r in srt[:10]:
        lines.append("  %+3s  %s" % (r["preference"], (r.get("title") or "")[:78]))
    lines += ["", "bottom 10 (lowest bids)"]
    for r in srt[-10:]:
        lines.append("  %+3s  %s" % (r["preference"], (r.get("title") or "")[:78]))
    return "\n".join(lines)


# --------------------------------- main --------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Score submissions by TF-IDF similarity to your papers (+ topic interests), and fill bids.")
    ap.add_argument("input", help="preferences CSV (paper,title,preference[,abstract,topics])")
    ap.add_argument("--topic-interests", dest="topic_interests", default="topic_interests.csv",
                    help="your topic,interest CSV in [-2,2] (default: topic_interests.csv)")
    ap.add_argument("--pdfs", default=DEFAULT_PDF_DIR,
                    help="folder of your paper PDFs (default: %s; at least %d unique required)"
                         % (DEFAULT_PDF_DIR, MIN_UNIQUE_PDFS))
    ap.add_argument("--method", choices=["tfidf", "specter2"], default="tfidf",
                    help="similarity method: tfidf (default; light, offline) or specter2 (neural "
                         "scientific-paper embeddings; needs torch+transformers+adapters and a one-time "
                         "model download)")
    ap.add_argument("--profile-out", dest="profile_out", default=DEFAULT_PROFILE,
                    help="where to save the profile summary JSON (default: %s)" % DEFAULT_PROFILE)
    ap.add_argument("--positive-frac", dest="positive_frac", type=float, default=0.1, metavar="F",
                    help="target fraction (0..1) of papers to bid positively on, within +/-10 points "
                         "(default: 0.1)")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="scoring parameters YAML (default: config.yaml beside this script)")
    ap.add_argument("-o", "--output", help="output CSV path (default: <input>.scored.csv; abstracts removed)")
    ap.add_argument("--keep-original", dest="keep_original", action="store_true",
                    help="keep the input CSV (by default it is deleted once the scored output is written)")
    ap.add_argument("--report", help="also write the text report to this path")
    ap.add_argument("--quiet", action="store_true", help="don't print the report")
    args = ap.parse_args(argv)

    if not os.path.exists(args.input):
        ap.error("input not found: %s" % args.input)
    if not (0.0 <= args.positive_frac <= 1.0):
        ap.error("--positive-frac must be a float between 0 and 1")
    load_config(args.config)

    # read submissions
    with open(args.input, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "preference" not in fields:
        ap.error("input must have a 'preference' column")
    if not rows:
        ap.error("input has no data rows")
    has_topics = "topics" in fields
    has_abstract = "abstract" in fields
    mode = "abstract+topics" if (has_abstract and has_topics) else \
           ("abstract" if has_abstract else ("topics" if has_topics else "title-only"))

    interests = read_topic_interests(args.topic_interests)          # required input
    pdfs = require_pdfs(args.pdfs, MIN_UNIQUE_PDFS)                  # >=5 enforced

    # ---- semantic similarity: your papers vs each submission ----
    paper_texts = read_pdf_texts(pdfs)
    sub_texts = [((r.get("title") or "") + ". " + (r.get("abstract") or "")) for r in rows]
    sem, top_terms = semantic_scores(paper_texts, sub_texts, args.method)
    mu = sum(sem) / len(sem)
    sd = (sum((s - mu) ** 2 for s in sem) / len(sem)) ** 0.5 or 1.0
    sem_signal = [clamp((s - mu) / sd * SEM_GAIN, _REF_MIN, _REF_MAX) for s in sem]

    # ---- topic-interest base per submission ----
    aff = {t: i * INTEREST_TO_AFFINITY for t, i in interests.items()}
    topic_base = []
    for r in rows:
        a = [aff.get(t.strip(), 0) for t in (r.get("topics") or "").split(";") if t.strip()]
        topic_base.append(0.6 * max(a) + 0.4 * (sum(a) / len(a)) if a else 0.0)

    # ---- blend, then map to bids targeting ~positive_frac positive ----
    iw = INTEREST_WEIGHT
    values = [clamp((1 - iw) * sem_signal[i] + iw * topic_base[i], _REF_MIN, _REF_MAX)
              for i in range(len(rows))]
    prefs = bids_for_positive_fraction(values, args.positive_frac)
    for r, p in zip(rows, prefs):
        r["preference"] = str(p)
    achieved = sum(1 for p in prefs if p > 0) / len(prefs)
    target_note = "target positive: %.0f%%  ->  achieved: %.0f%%" % (
        100 * args.positive_frac, 100 * achieved)
    if abs(achieved - args.positive_frac) > 0.10:
        sys.stderr.write("WARNING: " + target_note + "  - more than 10 points off "
                         "(distribution too lumpy).\n")

    # ---- save the profile summary (for inspection; never hand-edited) ----
    profile = collections.OrderedDict()
    profile["meta"] = {
        "built": datetime.date.today().isoformat(),
        "matching": ("%s similarity to your papers, blended with topic interests"
                     % ("SPECTER2 neural-embedding" if args.method == "specter2" else "TF-IDF cosine")),
        "interest_weight": INTEREST_WEIGHT, "sem_gain": SEM_GAIN,
        "unique_pdfs": len(pdfs), "submissions_scored": len(rows),
        "note": "Generated by score_bids.py - do not hand-edit. Topic interests come from %s; "
                "semantics from papers_pdf/." % os.path.basename(args.topic_interests),
    }
    profile["topic_affinity"] = collections.OrderedDict(
        (t, {"affinity": aff[t], "interest": interests[t]}) for t in interests)
    profile["corpus_top_terms"] = top_terms
    with open(args.profile_out, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, ensure_ascii=False)

    # ---- output path ----
    if args.output:
        out_path = args.output
    else:
        stem, ext = os.path.splitext(args.input)
        out_path = stem + ".scored" + (ext or ".csv")

    # write WITHOUT the abstract column
    out_fields = [f for f in fields if f != "abstract"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # delete the original (abstract-laden) input unless asked to keep it
    removed_original = False
    if not args.keep_original and os.path.abspath(args.input) != os.path.abspath(out_path):
        try:
            os.remove(args.input)
            removed_original = True
        except OSError as e:
            sys.stderr.write("WARNING: could not delete original %s: %s\n" % (args.input, e))

    report = make_report(rows, mode)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
    if not args.quiet:
        print(report)
        print(target_note)
        tail = "\nwrote %d bids -> %s  (profile: %s)" % (len(rows), out_path, args.profile_out)
        if removed_original:
            tail += "\ndeleted original %s; use --keep-original to keep it" % args.input
        print(tail)


if __name__ == "__main__":
    main()
