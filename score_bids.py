#!/usr/bin/env python3
"""
score_bids.py - score conference submissions by similarity to your work, and fill bids.

Matching is SEMANTIC, not keyword-based. Your papers (papers_pdf/) and each submission's
title+abstract are compared, and every submission is scored by its mean similarity to your
three most-similar papers -- so it has to match a sub-area of your work rather than one
fluke neighbor. Your -2..2 topic interests then steer that score (a blend). Finally the
bids are written in [-bid_max, bid_max], targeting a chosen fraction of positive bids
without squashing your top papers.

--method picks how that similarity is computed -- tfidf (default), specter2, or rerank (see
the bottom of this docstring). Only that step differs; everything after it is identical.

PIPELINE
    python3 make_topic_interests.py revprefs.csv       # 1. template (topics at 0)
    #   ... set your interest per topic in topic_interests.csv (-2..2) ...
    python3 score_bids.py revprefs.csv                 # 2. score + fill bids (uses topic_interests.csv)

HOW A BID IS COMPUTED (parameters in config.yaml)
    sem        = mean of the submission's top-3 similarities to your papers (by --method:
                 TF-IDF cosine, SPECTER2 embeddings, or a cross-encoder rerank), then
                 rank/quantile-transformed across submissions and shaped by sem_gain
                 -> [-ref_max, ref_max]
    topic_base = 0.6*max + 0.4*mean of your interests (scaled to ref_max) for the
                 submission's topic tags
    value      = (1 - interest_weight)*sem + interest_weight*topic_base       (in [-ref_max, ref_max])
    bid        = value mapped to [-bid_max, bid_max] so ~--positive-frac of papers are positive
                 (threshold just below that many papers, each side rescaled to the full range)

By default the abstract-laden input is deleted and an abstract-free scored CSV is written;
--keep-original keeps it. At least 5 unique PDFs with extractable text in papers_pdf/ are required.

Dependencies: scikit-learn (pip install scikit-learn) and pypdf (pip install pypdf).
PyYAML is optional (config.yaml has a built-in fallback parser).

--method specter2 swaps TF-IDF for AllenAI SPECTER2 neural embeddings (semantic matching
beyond shared wording). It needs extra packages and a one-time model download:
    pip install torch transformers adapters

--method rerank keeps the TF-IDF pass to shortlist the top-N candidates, then rescores only
those with a local cross-encoder (retrieve-then-rerank). It also needs a one-time download:
    pip install sentence-transformers
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
MIN_PDF_CHARS = 200         # a ~3-page extract shorter than this yielded no usable text
DEFAULT_PROFILE = "reviewer-expertise-profile.json"
DEFAULT_EMB_CACHE = ".specter2_cache.npz"   # on-disk cache for SPECTER2 embeddings

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

    if not (isinstance(BID_LIMIT, int) and BID_LIMIT >= 1):   # checked first: bounds bid_max
        sys.exit("config: bid_limit must be an int >= 1, got %r" % BID_LIMIT)
    if not (isinstance(BID_MAX, int) and 1 <= BID_MAX <= BID_LIMIT):
        sys.exit("config: bid_max must be an int in [1, %d], got %r" % (BID_LIMIT, BID_MAX))
    if not (0.0 <= INTEREST_WEIGHT <= 1.0):
        sys.exit("config: interest_weight must be in [0, 1], got %r" % INTEREST_WEIGHT)
    if not (isinstance(_REF_MAX, (int, float)) and _REF_MAX > 0):
        sys.exit("config: ref_max must be a positive number, got %r" % _REF_MAX)
    g["BID_MIN"]  = -BID_MAX
    g["_REF_MIN"] = -_REF_MAX
    # A -2..2 interest maps onto the same reference span the similarity signal uses, so the two
    # sides of the blend stay in proportion whatever ref_max is. Hardcoding this let an edited
    # ref_max move one side and not the other, silently reweighting papers against interests.
    g["INTEREST_TO_AFFINITY"] = _REF_MAX / 2.0


# ------------------------------- helpers -------------------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rnd(x):
    """Round half away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _signpow(x, e):
    """Signed power: keep x's sign, raise its magnitude to e."""
    return math.copysign(abs(x) ** e, x)


def _quantile_ranks(values):
    """Average-rank quantiles in (0, 1); tied values share the mean rank (so the many
    exactly-equal TF-IDF similarities -- e.g. the pile at zero -- aren't spread across
    the scale by tie order)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    q = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        qv = ((i + j) / 2.0 + 0.5) / n          # mean rank of the tie block -> quantile
        for t in range(i, j + 1):
            q[order[t]] = qv
        i = j + 1
    return q


def _quiet_library_warnings(warnings):
    """Silence sklearn/torch/transformers chatter without blanket-ignoring everything.

    A bare filterwarnings("ignore") also hides RuntimeWarning, which is how numpy reports
    divide-by-zero and invalid values -- i.e. exactly the warnings that would mean the scoring
    itself has gone wrong. These three categories cover the library noise and nothing numeric.
    """
    for cat in (UserWarning, FutureWarning, DeprecationWarning):
        warnings.filterwarnings("ignore", category=cat)


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
    out, noisy = [], []
    for fn in pdf_paths:
        # Scoped and RECORDED, not discarded: pypdf's complaints are the first sign of a damaged
        # file, and the old process-wide ignore also muted every warning raised after this point.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                out.append("".join((p.extract_text() or "") for p in PdfReader(fn).pages[:pages]))
            except Exception as exc:
                out.append("")
                noisy.append((fn, "unreadable (%s)" % exc))
                continue
        if caught:
            noisy.append((fn, "%d parser warning(s): %s"
                          % (len(caught), str(caught[0].message)[:60])))
    if noisy:
        sys.stderr.write("NOTE: pypdf reported problems with %d of %d PDF(s):\n"
                         % (len(noisy), len(pdf_paths)))
        for fn, why in noisy:
            sys.stderr.write("    %-34s %s\n" % (os.path.basename(fn)[:34], why))
    return out


def usable_paper_texts(pdf_paths, folder, minimum):
    """Read the PDFs and keep only the ones that actually yielded text.

    pypdf extracts nothing from scanned, image-only, or encrypted files, and read_pdf_texts
    turns any failure into "". Left in the corpus those become empty documents: every cosine
    similarity against them is exactly 0, which ties the pool in the rank transform and
    flattens the semantic half of the score to nothing -- the bids would come entirely from
    topic_interests.csv, and the profile's top terms would be an argsort of an all-zero
    vector (plausible-looking, but meaningless). So name the unreadable ones, drop them, and
    hold their contents to the same minimum the files themselves must meet.
    """
    texts = read_pdf_texts(pdf_paths)
    good, bad = [], []
    for fn, txt in zip(pdf_paths, texts):
        (good if len(txt.strip()) >= MIN_PDF_CHARS else bad).append((fn, txt))
    if bad:
        sys.stderr.write("WARNING: %d of %d PDF(s) yielded no usable text (scanned, image-only "
                         "or encrypted?) and were skipped:\n" % (len(bad), len(pdf_paths)))
        for fn, txt in bad:
            sys.stderr.write("    %-50s %d chars\n" % (os.path.basename(fn)[:50], len(txt.strip())))
        sys.stderr.write("  OCR them or re-export them with a text layer to include them.\n")
    if len(good) < minimum:
        sys.exit("ERROR: only %d of %d PDF(s) in '%s' contain extractable text; at least %d "
                 "required.\nWithout them the bids would be scored on your topic interests "
                 "alone." % (len(good), len(pdf_paths), folder, minimum))
    return [fn for fn, _ in good], [txt for _, txt in good]


# ------------------------------ topic interests ------------------------------
def read_topic_interests(path):
    """topic,interest CSV; interest is an integer in [-2, 2].

    This file is the only judgment input in the tool, so anything that isn't a plain integer in
    range is reported rather than quietly coerced: a word reads as 0 and an out-of-range number
    is clamped, either of which would silently discard a rating you meant to give.
    """
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
    bad = []
    for r in rows:
        t = (r.get("topic") or "").strip()
        if not t:
            continue
        v = (r.get("interest") or "").strip()
        try:
            raw, note = (float(v) if v else 0.0), None
        except ValueError:
            raw, note = 0.0, "not a number"
        iv = max(-2, min(2, int(round(raw))))
        if note is None and v and raw != iv:
            note = "not an integer in -2..2"
        if note:
            bad.append((t, v, note, iv))
        out[t] = iv
    if bad:
        sys.stderr.write("WARNING: %d row(s) in %s don't hold an integer in -2..2:\n"
                         % (len(bad), path))
        for t, v, note, iv in bad:
            sys.stderr.write("    %-45s %-10s %s; read as %+d\n"
                             % (t[:45], repr(v), note, iv))
        sys.stderr.write("  Fix them and re-run, or accept the values shown.\n")
    if not out:
        sys.exit("ERROR: no topics found in %s." % path)
    return out


def report_topic_coverage(rows, interests):
    """Report topic tags the interests file doesn't cover, and rows that cover nothing.

    Tags are matched by exact string, so a topic the conference added or renamed after you built
    the file scores neutral with nothing to show it -- as does a hand-edited typo. Both directions
    are worth seeing, and neither is fatal: a conference may legitimately carry a topic you never
    rated. Silent when the export has no topics column at all.
    """
    tags = collections.Counter()
    for r in rows:
        for t in (r.get("topics") or "").split(";"):
            t = t.strip()
            if t:
                tags[t] += 1
    if not tags:
        return
    unrated = sorted(t for t in tags if t not in interests)
    unused = sorted(t for t in interests if t not in tags)
    if unrated:
        sys.stderr.write("WARNING: %d topic tag(s) on the submissions have no row in your "
                         "interests file and score neutral:\n" % len(unrated))
        for t in unrated[:20]:
            sys.stderr.write("    %-58s (%d submission(s))\n" % (t[:58], tags[t]))
        if len(unrated) > 20:
            sys.stderr.write("    ... and %d more\n" % (len(unrated) - 20))
        sys.stderr.write("  Re-run make_topic_interests.py to pick up new or renamed topics.\n")
    if unused:
        shown = ", ".join(unused[:5]) + (", ..." if len(unused) > 5 else "")
        sys.stderr.write("NOTE: %d topic(s) in your interests file match no submission this "
                         "round: %s\n" % (len(unused), shown))


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


_ABSTRACT_RE = re.compile(r"\babstract\b", re.I)
_ABSTRACT_END_RE = re.compile(
    r"\b(\d+\s*[.\)]?\s*introduction\b|introduction\b|keywords\b|index terms\b|"
    r"categories and subject\b|ccs concepts\b|general terms\b)", re.I)


def _split_title_abstract(raw):
    """Best-effort (title, abstract) from a paper's leading PDF text.

    Returns (title, abstract), or (None, None) if no plausible abstract parses so the
    caller can fall back to the raw page text. Used to feed SPECTER2 the in-distribution
    'title + abstract' it was trained on instead of a truncated three-page dump.
    """
    txt = raw.replace("-\n", "")
    m = _ABSTRACT_RE.search(txt)
    if not m:
        return None, None
    head = [ln.strip() for ln in txt[:m.start()].splitlines() if ln.strip()]
    head = [ln for ln in head if not re.match(r"(?i)^arxiv[:\s]", ln)]
    title = head[0] if head else ""
    rest = txt[m.end():]
    em = _ABSTRACT_END_RE.search(rest)
    abstract = " ".join((rest[:em.start()] if em else rest[:2500]).split())
    if len(abstract) < 40:                       # too short to be a real abstract
        return None, None
    return title, abstract


SPECTER2_MODEL = "allenai/specter2_base"
SPECTER2_ADAPTER = "allenai/specter2"
SPECTER2_SEP = "[SEP]"   # BERT-based SPECTER2's sep_token; the model is trained on title+SEP+abstract

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"   # multilingual cross-encoder; handles long abstracts
RERANK_AGG_TOPK = 3              # a candidate's score is the mean of its top-k paper scores
RERANK_MARGIN_FRAC = 0.10        # default shortlist headroom above --positive-frac
RERANK_MAX_FRAC = 0.5            # past this the shortlist isn't one; specter2 fits better
# Joint token budget for a (your paper, submission) pair. The model accepts 8192, so this is a
# speed choice: at ~1.7 tokens per word, 1024 leaves ~300 words per side, which clears any
# realistic abstract, while cost grows faster than length (1024 costs ~3x 512; 2048 ~20x).
RERANK_MAX_LENGTH = 1024


def paper_pair_texts(paper_texts):
    """Your papers as the cross-encoder sees them: title+abstract where one parses, else the
    raw leading page text. Shared so a calibration run feeds the model exactly what scoring does.
    """
    out = []
    for raw in paper_texts:
        pt, pa = _split_title_abstract(raw)
        out.append((pt + ". " + pa) if pa else " ".join(raw.split()))
    return out


def _topk_mean(sim, k=3):
    """Per-submission similarity as the mean of its top-k paper cosines, not the single
    max: a submission must match a sub-area of your work rather than one fluke neighbor
    or one shared rare bigram. Falls back to fewer columns when you have <k papers."""
    import numpy as np
    k = min(k, sim.shape[1])
    if k <= 1:
        return sim.max(axis=1)
    return np.partition(sim, -k, axis=1)[:, -k:].mean(axis=1)


TFIDF_MIN_DF = 2       # a term must appear in at least this many submissions ...
TFIDF_MAX_DF = 0.3     # ... and in fewer than this fraction of them
# Both conditions can only hold together once the pool is big enough for MAX_DF to admit
# MIN_DF documents; below that, no term can qualify and no vocabulary exists at all.
TFIDF_MIN_SUBMISSIONS = int(math.ceil(TFIDF_MIN_DF / TFIDF_MAX_DF))


def _tfidf_fit(paper_texts, sub_texts, required=True):
    """Fit TF-IDF on the CONFERENCE submissions and project your papers into that space.

    Returns (S, P, vectorizer). Fitting on the submission pool means terms specific to your
    papers but absent from it (your name, affiliation, venue boilerplate) never enter the
    vocabulary, and domain-generic words get low IDF weight.

    The two document-frequency bounds can leave nothing behind -- for a pool too small to
    satisfy them at all, or one whose texts are so alike that every term lands above MAX_DF.
    sklearn signals both with a bare ValueError, so they are reported as what the user can act
    on instead. required=False downgrades that to a warning and returns (None, None, None), for
    callers that only wanted the cosmetic top-terms summary and shouldn't lose a finished run
    over it.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    except Exception:
        sys.exit("scikit-learn is required: pip install scikit-learn")
    import warnings
    _quiet_library_warnings(warnings)

    def fail(msg):
        if required:
            sys.exit("ERROR: " + msg)
        sys.stderr.write("WARNING: " + msg + "\n  The profile's top-terms summary will be "
                         "empty; the bids themselves are unaffected.\n")

    n = len(sub_texts)
    if n < TFIDF_MIN_SUBMISSIONS:
        fail("%d submission(s) is too few to build a vocabulary from: a term has to appear in at "
             "least %d of them but in fewer than %.0f%%, and no term can do both below %d "
             "submissions.\nThis tool is built for a full reviewing round -- with a pool this "
             "small, bid by hand."
             % (n, TFIDF_MIN_DF, 100 * TFIDF_MAX_DF, TFIDF_MIN_SUBMISSIONS))
        return None, None, None

    stop = list(ENGLISH_STOP_WORDS.union(_EXTRA_STOP))
    vec = TfidfVectorizer(preprocessor=_clean, stop_words=stop,
                          token_pattern=r"(?u)\b[a-z][a-z]+\b",   # alphabetic tokens, 2+ chars
                          ngram_range=(1, 2), min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF,
                          sublinear_tf=True)
    try:
        S = vec.fit_transform(sub_texts)
    except ValueError:
        fail("no usable vocabulary in these %d submissions: every candidate term was either too "
             "rare (in fewer than %d of them) or too common (in %.0f%% or more).\nThat happens "
             "when the abstracts are near-identical, or mostly empty -- check that the 'abstract' "
             "column is actually populated."
             % (n, TFIDF_MIN_DF, 100 * TFIDF_MAX_DF))
        return None, None, None
    P = vec.transform(paper_texts)
    return S, P, vec


def _tfidf_scores(paper_texts, sub_pairs):
    """Default: TF-IDF cosine similarity. Returns (best_per_submission, top_terms)."""
    from sklearn.metrics.pairwise import cosine_similarity
    sub_texts = [(t + ". " + a) for t, a in sub_pairs]
    S, P, vec = _tfidf_fit(paper_texts, sub_texts)
    best = _topk_mean(cosine_similarity(S, P), k=3)   # match a sub-area, not one fluke paper
    return best, corpus_top_terms(vec, P)


def _load_emb_cache(path, model_id):
    """Load {sha256(text): vector} from an .npz cache; ignore it if the model differs."""
    import numpy as np
    if not path or not os.path.exists(path):
        return {}
    try:
        d = np.load(path, allow_pickle=True)
        if "model" not in d.files or str(d["model"]) != model_id:
            return {}
        return {str(k): d["vecs"][i] for i, k in enumerate(d["keys"])}
    except Exception:
        return {}


def _save_emb_cache(path, model_id, cache):
    import numpy as np
    if not path:
        return
    try:
        ks = list(cache.keys())
        vecs = np.vstack([cache[k] for k in ks]) if ks else np.zeros((0, 0), dtype="float32")
        np.savez(path, model=model_id, keys=np.array(ks), vecs=vecs)
    except Exception:
        pass


def _specter2_scores(paper_texts, sub_pairs, cache_path=DEFAULT_EMB_CACHE):
    """--method specter2: neural scientific-paper embeddings (AllenAI SPECTER2).

    Both sides are embedded as 'title [SEP] abstract', the in-distribution form SPECTER2 was
    trained on -- your papers are parsed down to title+abstract (falling back to the raw page
    text only when no abstract parses) instead of being fed a truncated three-page dump.

    Embeddings drive the matching and are cached on disk keyed by sha256(text), so unchanged
    abstracts are embedded only once across runs (a fully-cached re-run never loads the model).
    The profile's top-terms summary is still derived from TF-IDF so the saved JSON stays readable.
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
    _quiet_library_warnings(warnings)

    # assemble 'title [SEP] abstract' for both sides; parse your papers, fall back to raw text
    pap_texts = []
    for raw in paper_texts:
        pt, pa = _split_title_abstract(raw)
        pap_texts.append((pt + SPECTER2_SEP + pa) if pa else " ".join(raw.split()))
    sub_join = [(t + SPECTER2_SEP + a) for t, a in sub_pairs]

    cache = _load_emb_cache(cache_path, SPECTER2_MODEL)
    n_cached = len(cache)
    tok = model = None   # loaded lazily -- only if something actually needs embedding

    def embed(texts, batch=16):
        nonlocal tok, model
        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in cache]
        if todo and model is None:
            sys.stderr.write("Loading SPECTER2 (%s)...\n" % SPECTER2_MODEL)
            sys.stderr.flush()
            tok = AutoTokenizer.from_pretrained(SPECTER2_MODEL)
            model = AutoAdapterModel.from_pretrained(SPECTER2_MODEL)
            name = model.load_adapter(SPECTER2_ADAPTER, source="hf", load_as="proximity", set_active=True)
            model.set_active_adapters(name or "proximity")   # ensure the adapter runs in forward()
            model.eval()
        done = 0
        for s in range(0, len(todo), batch):
            idx = todo[s:s + batch]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            with torch.no_grad():
                rep = model(**enc).last_hidden_state[:, 0, :].cpu().numpy()   # CLS-token embedding
            for j, i in enumerate(idx):
                cache[keys[i]] = rep[j]
            done += len(idx)
            if len(todo) > 200:                          # live progress for the slow CPU encode
                sys.stderr.write("\r  SPECTER2 embedding %d/%d" % (done, len(todo)))
                sys.stderr.flush()
        if len(todo) > 200:
            sys.stderr.write("\n")
        return np.vstack([cache[k] for k in keys])

    sub_emb = embed(sub_join)
    pap_emb = embed(pap_texts)
    if len(cache) != n_cached:                        # only rewrite the cache if it grew
        _save_emb_cache(cache_path, SPECTER2_MODEL, cache)
    best = _topk_mean(cosine_similarity(sub_emb, pap_emb), k=3)
    sub_tfidf = [(t + ". " + a) for t, a in sub_pairs]
    # only for the profile's readable summary, so a pool with no vocabulary must not throw away
    # an otherwise finished (and expensive) embedding run
    _, P, vec = _tfidf_fit(paper_texts, sub_tfidf, required=False)
    return best, (corpus_top_terms(vec, P) if vec is not None else [])


def _rerank_scores(paper_texts, sub_pairs, topn, model_id=DEFAULT_RERANK_MODEL,
                   max_length=RERANK_MAX_LENGTH):
    """--method rerank: TF-IDF retrieval + a local cross-encoder reranker (retrieve-then-rerank).

    The cheap TF-IDF pass ranks every submission; only the top-N candidates are then rescored by
    a cross-encoder, which reads each (your paper, submission) pair JOINTLY -- more precise than
    comparing independent vectors, but too slow to run on the whole pool. Each candidate is scored
    against your papers (title+abstract, parsed like specter2, falling back to the raw page text)
    and aggregated over them with a top-3 mean. Fully offline after the one-time model download,
    and deterministic: an eval-mode forward pass with no sampling, so identical inputs give
    identical scores.

    Reconciling the two scales: TF-IDF cosines (~0..1, bunched low) and cross-encoder logits
    (unbounded, different units) are not comparable, so we never compare them. Each group is
    rank-transformed within itself, and the reranked candidates are placed as a band strictly
    ABOVE the non-candidates, which keep their TF-IDF order. The downstream normalize step only
    cares about ordering, so this is all that's needed to keep the two scores from fighting.
    """
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        sys.exit("--method rerank needs sentence-transformers:\n"
                 "    pip install sentence-transformers\n"
                 "It downloads the reranker once on first run (needs network); afterwards it runs offline.")
    import warnings
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    _quiet_library_warnings(warnings)

    # ---- stage 1: TF-IDF retrieval over the whole pool (same scoring as --method tfidf) ----
    sub_texts = [(t + ". " + a) for t, a in sub_pairs]
    S, P, vec = _tfidf_fit(paper_texts, sub_texts)
    tfidf_best = _topk_mean(cosine_similarity(S, P), k=3)
    n = len(sub_pairs)
    k = max(1, min(topn, n))
    cand = sorted(range(n), key=lambda i: tfidf_best[i], reverse=True)[:k]   # top-N candidate indices
    cand_set = set(cand)
    # Announce the real work: every candidate is scored against every paper, so the job is
    # k x papers pairs, not k. And k auto-scales with --positive-frac, which is a flag people
    # change for unrelated reasons -- so the cost has to be visible before the model loads.
    sys.stderr.write("rerank: %d candidate(s) of %d submissions x %d paper(s) = %d "
                     "cross-encoder pair(s), max_length %d\n"
                     % (k, n, len(paper_texts), k * len(paper_texts), max_length))
    sys.stderr.flush()

    # ---- stage 2: cross-encoder rerank of the candidates only ----
    pap_texts = paper_pair_texts(paper_texts)
    sys.stderr.write("Loading reranker (%s)...\n" % model_id)
    sys.stderr.flush()
    ce = CrossEncoder(model_id, max_length=max_length)
    ce.model.eval()                                                         # deterministic forward pass
    pairs = [(pap_texts[p], sub_texts[i]) for i in cand for p in range(len(pap_texts))]
    logits = ce.predict(pairs, batch_size=32, convert_to_numpy=True,
                        show_progress_bar=(len(pairs) > 200))
    ce_mat = np.asarray(logits, dtype="float64").reshape(len(cand), len(pap_texts))
    ce_best = _topk_mean(ce_mat, k=RERANK_AGG_TOPK)                         # aggregate over your papers

    # ---- reconcile: reranked band [1, 2) sits above the non-candidate band [0, 1) ----
    scores = np.empty(n, dtype="float64")
    noncand = [i for i in range(n) if i not in cand_set]
    for i, q in zip(noncand, _quantile_ranks([tfidf_best[i] for i in noncand])):
        scores[i] = q
    for j, q in zip(cand, _quantile_ranks(list(ce_best))):
        scores[j] = 1.0 + q
    return scores, corpus_top_terms(vec, P)


def semantic_scores(paper_texts, sub_pairs, method="tfidf", cache_path=DEFAULT_EMB_CACHE,
                    rerank_topn=None, rerank_model=DEFAULT_RERANK_MODEL,
                    rerank_max_length=RERANK_MAX_LENGTH):
    """Per-submission similarity to your most-similar papers (top-k), by the chosen method.

    paper_texts: raw leading PDF text per paper. sub_pairs: (title, abstract) per submission.
    """
    if method == "specter2":
        return _specter2_scores(paper_texts, sub_pairs, cache_path=cache_path)
    if method == "rerank":
        return _rerank_scores(paper_texts, sub_pairs, topn=rerank_topn, model_id=rerank_model,
                              max_length=rerank_max_length)
    return _tfidf_scores(paper_texts, sub_pairs)


def corpus_top_terms(vec, P, topn=40):
    """Highest mean TF-IDF terms across your papers - a readable summary of your profile."""
    mean = P.mean(axis=0)
    arr = mean.A1 if hasattr(mean, "A1") else (mean.toarray().ravel() if hasattr(mean, "toarray") else mean.ravel())
    terms = vec.get_feature_names_out()
    idx = arr.argsort()[::-1][:topn]
    return [str(terms[i]) for i in idx]


# ------------------------- target positive fraction --------------------------
def rerank_shortlist_frac(positive_frac, rerank_frac=None):
    """Fraction of the pool the cross-encoder rescores.

    Reranked candidates sort above the rest, so positive bids come from the reranked set: the
    shortlist must cover positive_frac of the pool, or some positive bids fall to submissions the
    cross-encoder never scored. That floor is a correctness condition, not a preference.

    Anything above it is headroom for promoting submissions TF-IDF underranks. The default adds
    RERANK_MARGIN_FRAC of the POOL rather than a multiple of positive_frac, because what the
    headroom absorbs is rank displacement between the two scorers -- a property of the pool, not
    of the target. A multiple of the target misbehaves at both ends: at positive_frac 0.01 it
    leaves a handful of ranks of slack, and at 0.5 far more than any reordering needs.
    """
    if rerank_frac is not None:
        return rerank_frac
    return max(positive_frac, min(positive_frac + RERANK_MARGIN_FRAC, RERANK_MAX_FRAC))


def rerank_topn(n_submissions, frac):
    """The shortlist fraction as a candidate count, at least 1 and at most the whole pool."""
    return max(1, min(n_submissions, int(math.ceil(frac * n_submissions))))


def bids_for_positive_fraction(values, target):
    """Map continuous reference scores to final bids so ~`target` of papers are positive,
    putting the threshold just below the target count and rescaling EACH SIDE to the full
    output range, so the strongest paper reaches +bid_max and the weakest -bid_max.

    The threshold is derived from the number of papers that should end up positive rather than
    from its complement: a bid counts as positive only when it is STRICTLY above tau, so tau has
    to sit one slot below the target band or the paper at the cut is excluded and the run lands
    one short. (target=1 still tops out at (n-1)/n -- some paper has to be the lowest.)
    """
    n = len(values)
    if n == 0:
        return []
    sv = sorted(values)
    k = int(round(target * n))                        # papers that should end up positive
    idx = min(max(n - k - 1, 0), n - 1)               # tau sits just below them
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


# ---------------------------- change tracking --------------------------------
def _load_bids(path):
    """paper_id -> (bid, title) from a scored CSV (best-effort; skips unparsable rows)."""
    out = {}
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                key = (r.get("paper") or "").strip()
                if not key:
                    continue
                try:
                    out[key] = (int(r["preference"]), (r.get("title") or ""))
                except (KeyError, ValueError, TypeError):
                    continue
    except OSError:
        pass
    return out


_RUN_TS_RE = r"\d{8}-\d{6}"


def _find_previous_output(out_path, run_ts):
    """The most recent EARLIER scored CSV to diff the new bids against.

    Default output names carry this run's timestamp, so the series is every name matching it with
    some other stamp; we take the latest stamp strictly before run_ts. Ordering on the stamp is
    what makes it the previous RUN -- %Y%m%d-%H%M%S sorts correctly as text, whereas mtime is
    perturbed by any copy, sync, or editor save and would silently pick the wrong baseline.

    A custom -o carries no stamp and so has no series, but a file already sitting at that exact
    path is the run we are about to replace -- which makes it the right baseline, provided the
    caller reads it before writing.
    """
    base = os.path.basename(out_path)
    if run_ts not in base:
        return out_path if os.path.isfile(out_path) else None
    dirn = os.path.dirname(out_path) or "."
    head, _, tail = base.partition(run_ts)
    pat = re.compile("^%s(%s)%s$" % (re.escape(head), _RUN_TS_RE, re.escape(tail)))
    best = None
    try:
        names = os.listdir(dirn)
    except OSError:
        return None
    for name in names:
        m = pat.match(name)
        if not m or m.group(1) >= run_ts:                 # only runs before this one
            continue
        path = os.path.join(dirn, name)
        if os.path.isfile(path) and (best is None or m.group(1) > best[0]):
            best = (m.group(1), path)
    return best[1] if best else None


def make_change_summary(prev_path, prev, rows, profile_path=None):
    """Human-readable diff of the current bids vs the previous run's scored CSV, keyed on the
    `paper` id. Takes the previous bids already read, because with a fixed -o the baseline file
    is the one this run overwrites. Always returns text (notes 'first run' when there's nothing
    to compare)."""
    new = {}
    for r in rows:
        key = (r.get("paper") or "").strip()
        if not key:
            continue
        try:
            new[key] = (int(r["preference"]), (r.get("title") or ""))
        except (ValueError, TypeError):
            continue
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = ["Bid change summary", "=" * 60, "generated : %s" % now]
    if profile_path:                      # the run's settings live here, so the diff is explainable
        lines.append("settings  : %s" % os.path.basename(profile_path))
    if not prev_path:
        lines += ["previous  : (none -- first run in this series)",
                  "papers    : %d" % len(new), "",
                  "No previous output to compare against."]
        return "\n".join(lines)

    common = [k for k in new if k in prev]
    added = [k for k in new if k not in prev]
    removed = [k for k in prev if k not in new]
    changed = [(k, prev[k][0], new[k][0]) for k in common if prev[k][0] != new[k][0]]
    into_pos = sum(1 for _, o, n in changed if o <= 0 < n)
    out_of_pos = sum(1 for _, o, n in changed if n <= 0 < o)
    mad = (sum(abs(n - o) for _, o, n in changed) / len(changed)) if changed else 0.0

    lines += [
        "previous  : %s" % os.path.basename(prev_path),
        "papers    : %d previous, %d current" % (len(prev), len(new)), "",
        "unchanged : %d" % (len(common) - len(changed)),
        "changed   : %d   (mean |delta| %.2f)" % (len(changed), mad),
        "  into positive  (<=0 -> >0) : %d" % into_pos,
        "  out of positive (>0 -> <=0): %d" % out_of_pos,
        "added     : %d   (submissions new this round)" % len(added),
        "removed   : %d   (submissions gone this round)" % len(removed),
    ]
    if changed:
        movers = sorted(changed, key=lambda t: -abs(t[2] - t[1]))
        lines += ["", "biggest movers (up to 20):"]
        for k, o, n in movers[:20]:
            lines.append("  %+3d -> %+3d  (%+d)  %s" % (o, n, n - o, (new[k][1] or "")[:64]))
    if added:
        lines += ["", "new submissions (up to 20):"]
        for k in added[:20]:
            lines.append("  bid %+3d    %s" % (new[k][0], (new[k][1] or "")[:64]))
    if removed:
        lines += ["", "dropped submissions (up to 20):"]
        for k in removed[:20]:
            lines.append("  was %+3d    %s" % (prev[k][0], (prev[k][1] or "")[:64]))
    return "\n".join(lines)


# --------------------------------- main --------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Score submissions by similarity to your papers (+ topic interests), and fill bids.")
    ap.add_argument("input", help="preferences CSV (paper,title,preference[,abstract,topics])")
    ap.add_argument("--topic-interests", dest="topic_interests", default="topic_interests.csv",
                    help="your topic,interest CSV in [-2,2] (default: topic_interests.csv)")
    ap.add_argument("--pdfs", default=DEFAULT_PDF_DIR,
                    help="folder of your paper PDFs (default: %s; at least %d unique required)"
                         % (DEFAULT_PDF_DIR, MIN_UNIQUE_PDFS))
    ap.add_argument("--method", choices=["tfidf", "specter2", "rerank"], default="tfidf",
                    help="similarity method: tfidf (default; light, offline); specter2 (neural "
                         "scientific-paper embeddings; needs torch+transformers+adapters); or rerank "
                         "(TF-IDF retrieves the top-N candidates, a local cross-encoder reranks them; "
                         "needs sentence-transformers). specter2 and rerank do a one-time model download.")
    ap.add_argument("--emb-cache", dest="emb_cache", default=DEFAULT_EMB_CACHE,
                    help="cache file for --method specter2 embeddings, reused across runs "
                         "(default: %s)" % DEFAULT_EMB_CACHE)
    ap.add_argument("--rerank-frac", dest="rerank_frac", type=float, default=None, metavar="F",
                    help="--method rerank: fraction (0..1) of submissions the cross-encoder "
                         "rescores. Default: --positive-frac plus %g, capped at %g. Cannot be "
                         "below --positive-frac, since positive bids come from the reranked set."
                         % (RERANK_MARGIN_FRAC, RERANK_MAX_FRAC))
    ap.add_argument("--rerank-model", dest="rerank_model", default=DEFAULT_RERANK_MODEL,
                    help="--method rerank: cross-encoder model id (default: %s)" % DEFAULT_RERANK_MODEL)
    ap.add_argument("--rerank-max-length", dest="rerank_max_length", type=int,
                    default=RERANK_MAX_LENGTH, metavar="N",
                    help="--method rerank: joint token budget per (paper, submission) pair "
                         "(default: %d). Raising it keeps long abstracts intact but costs more "
                         "than proportionally more time." % RERANK_MAX_LENGTH)
    ap.add_argument("--profile-out", dest="profile_out", default=None,
                    help="where to save the profile summary JSON (default: %s with a run "
                         "timestamp inserted, so re-runs don't overwrite)" % DEFAULT_PROFILE)
    ap.add_argument("--positive-frac", dest="positive_frac", type=float, default=0.1, metavar="F",
                    help="target fraction (0..1) of papers to bid positively on, within +/-10 points "
                         "(default: 0.1)")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="scoring parameters YAML (default: config.yaml beside this script)")
    ap.add_argument("-o", "--output", help="output CSV path (default: <input>.scored.<timestamp>.csv, "
                    "abstracts removed; the timestamp keeps re-runs from overwriting)")
    ap.add_argument("--keep-original", dest="keep_original", action="store_true",
                    help="keep the input CSV (by default it is deleted once the scored output is written)")
    ap.add_argument("--report", help="also write the text report to this path")
    ap.add_argument("--quiet", action="store_true", help="don't print the report")
    args = ap.parse_args(argv)

    if not os.path.exists(args.input):
        ap.error("input not found: %s" % args.input)
    if not (0.0 <= args.positive_frac <= 1.0):
        ap.error("--positive-frac must be a float between 0 and 1")
    if args.rerank_frac is not None:
        if not (0.0 < args.rerank_frac <= 1.0):
            ap.error("--rerank-frac must be greater than 0 and at most 1")
        if args.rerank_frac < args.positive_frac:
            ap.error("--rerank-frac %.2f is below --positive-frac %.2f. Positive bids come from "
                     "the reranked set, so the shortlist has to cover them -- use at least %.2f."
                     % (args.rerank_frac, args.positive_frac, args.positive_frac))
    if args.rerank_max_length < 16:
        ap.error("--rerank-max-length must be at least 16 tokens")
    load_config(args.config)

    # one timestamp per run, stamped into every default output name so repeated runs
    # never clobber earlier results (and so the change summary can find the prior run).
    run_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # read submissions
    with open(args.input, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    # Validate the export before anything expensive or destructive happens: the input is
    # deleted by default, and --method rerank would otherwise pay for a whole cross-encoder
    # pass before finding out the columns it needs aren't there.
    missing = [c for c in ("paper", "preference") if c not in fields]
    if missing:
        ap.error("input must have a %s column (HotCRP matches rows by 'paper' on upload)"
                 % " and a ".join("'%s'" % c for c in missing))
    if not rows:
        ap.error("input has no data rows")
    has_topics = "topics" in fields
    has_abstract = "abstract" in fields
    mode = "abstract+topics" if (has_abstract and has_topics) else \
           ("abstract" if has_abstract else ("topics" if has_topics else "title-only"))
    if not has_abstract:
        sys.stderr.write("WARNING: no 'abstract' column - matching on titles alone, which is much "
                         "weaker. Re-export with the abstract column for a real match.\n")
    if not has_topics:
        sys.stderr.write("WARNING: no 'topics' column - your topic interests can't be applied, so "
                         "%.0f%% of every score is a constant 0. Re-export with the topics "
                         "column.\n" % (100 * INTEREST_WEIGHT))

    interests = read_topic_interests(args.topic_interests)          # required input
    report_topic_coverage(rows, interests)
    pdfs = require_pdfs(args.pdfs, MIN_UNIQUE_PDFS)                  # >=5 unique files

    # ---- semantic similarity: your papers vs each submission ----
    # narrows pdfs to the ones with real text, so meta.unique_pdfs counts what was used
    pdfs, paper_texts = usable_paper_texts(pdfs, args.pdfs, MIN_UNIQUE_PDFS)
    sub_pairs = [((r.get("title") or ""), (r.get("abstract") or "")) for r in rows]
    rr_frac = rerank_shortlist_frac(args.positive_frac, args.rerank_frac)
    rr_topn = rerank_topn(len(rows), rr_frac)
    if args.method == "rerank" and rr_frac > RERANK_MAX_FRAC:
        sys.stderr.write("NOTE: --rerank-frac %.2f rescores most of the pool, so retrieve-then-"
                         "rerank saves little here; --method specter2 likely fits better.\n" % rr_frac)
    sem, top_terms = semantic_scores(paper_texts, sub_pairs, args.method, cache_path=args.emb_cache,
                                     rerank_topn=rr_topn, rerank_model=args.rerank_model,
                                     rerank_max_length=args.rerank_max_length)
    # Rank/quantile-transform similarity onto the fixed reference scale: robust to TF-IDF's
    # right-skew and SPECTER2's anisotropy, and -- unlike a z-score -- independent of this
    # year's similarity spread, so the blend with topic_base stays stable across venues.
    # sem_gain shapes the curve (9 = linear; higher pushes mid-rank papers toward the rails).
    e = (9.0 / SEM_GAIN) if SEM_GAIN > 0 else 1.0
    sem_signal = [clamp(_signpow(2 * q - 1, e) * _REF_MAX, _REF_MIN, _REF_MAX)
                  for q in _quantile_ranks(list(sem))]

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

    achieved = sum(1 for p in prefs if p > 0) / len(prefs)
    target_note = "target positive: %.0f%%  ->  achieved: %.0f%%" % (
        100 * args.positive_frac, 100 * achieved)
    if abs(achieved - args.positive_frac) > 0.10:
        sys.stderr.write("WARNING: " + target_note + "  - more than 10 points off "
                         "(distribution too lumpy).\n")

    for r, p in zip(rows, prefs):
        r["preference"] = str(p)

    # ---- save the profile summary (for inspection; never hand-edited) ----
    profile = collections.OrderedDict()
    # Record what produced these bids, not just how they turned out: the change summary can show
    # that 109 bids moved between runs, but only these settings say why.
    matching = {"tfidf": "TF-IDF cosine", "specter2": "SPECTER2 neural-embedding",
                "rerank": "TF-IDF retrieval + %s cross-encoder rerank" % args.rerank_model}
    profile["meta"] = {
        "built": datetime.date.today().isoformat(),
        "matching": ("%s similarity to your papers, blended with topic interests"
                     % matching[args.method]),
        "method": args.method,
        "interest_weight": INTEREST_WEIGHT, "sem_gain": SEM_GAIN, "bid_max": BID_MAX,
        "positive_frac_target": args.positive_frac,
        "positive_frac_achieved": round(achieved, 4),
        "unique_pdfs": len(pdfs), "submissions_scored": len(rows),
        "papers": [os.path.basename(p) for p in pdfs],
        "note": "Generated by score_bids.py - do not hand-edit. Topic interests come from %s; "
                "semantics from papers_pdf/." % os.path.basename(args.topic_interests),
    }
    profile["topic_affinity"] = collections.OrderedDict(
        (t, {"affinity": aff[t], "interest": interests[t]}) for t in interests)
    profile["corpus_top_terms"] = top_terms
    if args.profile_out:
        profile_path = args.profile_out
    else:
        pstem, pext = os.path.splitext(DEFAULT_PROFILE)
        profile_path = "%s.%s%s" % (pstem, run_ts, pext)          # timestamped default
    with open(profile_path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, ensure_ascii=False)

    # ---- output path ----
    if args.output:
        out_path = args.output
    else:
        stem, ext = os.path.splitext(args.input)
        out_path = "%s.scored.%s%s" % (stem, run_ts, ext or ".csv")   # timestamped default

    # Find AND READ the previous run before writing: with a fixed -o the baseline is the very
    # file this run replaces, so reading it afterwards would diff the new bids against themselves.
    prev_output = _find_previous_output(out_path, run_ts)
    prev_bids = _load_bids(prev_output) if prev_output else {}
    if prev_output and not prev_bids:
        sys.stderr.write("WARNING: no bids could be read from %s; skipping the change "
                         "comparison.\n" % prev_output)
        prev_output = None

    # write WITHOUT the abstract column
    out_fields = [f for f in fields if f != "abstract"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- change summary vs the previous most-recent output ----
    changes_path = os.path.splitext(out_path)[0] + ".changes.txt"
    with open(changes_path, "w", encoding="utf-8") as fh:
        fh.write(make_change_summary(prev_output, prev_bids, rows, profile_path) + "\n")

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
        tail = "\nwrote %d bids -> %s  (profile: %s)" % (len(rows), out_path, profile_path)
        tail += "\nchange summary -> %s%s" % (
            changes_path, "" if prev_output else "  (first run in this series)")
        if removed_original:
            tail += "\ndeleted original %s; use --keep-original to keep it" % args.input
        print(tail)


if __name__ == "__main__":
    main()
