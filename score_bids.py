#!/usr/bin/env python3
"""
score_bids.py - build a reviewer profile from your topic scores + papers, and fill bids.

This is the one command you run each round. It (a) builds an expertise profile from your
edited topic_scores.csv plus positive keywords mined from papers_pdf/, saving it to
reviewer-expertise-profile.json, and (b) fills the `preference` column of the HotCRP
reviewer-preferences CSV with bids in [-bid_max, bid_max] (default +/-20).

PIPELINE
    python3 make_topic_interests.py revprefs.csv       # 1. template: every topic = 0 (neutral)
    #   ... set your interest per topic in topic_interests.csv (-2..2) ...
    python3 score_bids.py revprefs.csv                 # 2. build profile + fill bids (uses topic_interests.csv)

By default the abstract-laden input CSV is deleted and an abstract-free scored CSV is
written in its place; pass --keep-original to keep the input. At least 5 unique PDFs in
papers_pdf/ are required, or the tool stops.

(The profile is conference-specific - the topic scores differ each round - so it's built
on the fly here rather than as a separate step. The JSON is still written, for inspection.)

--------------------------------------------------------------------------------
HOW IT SCORES EACH PAPER
--------------------------------------------------------------------------------
1. TOPIC BASE  (only if the CSV has a `topics` column; ';'-separated topic categories)
       base        = 0.6 * max(aff) + 0.4 * mean(aff)    # aff = your topic interest (-2..2) x 10
       base_scaled = base * BASE_SCALE
2. KEYWORD ADJUSTMENT  (whole-word / phrase match, case-insensitive)
       kw = sum of mined positive-keyword weights in title (x2) + abstract, clamped to +/-KW_CAP
3. COMBINE + COMPRESS the top tail so best-fit papers separate from the merely in-scope.
4. GUARDRAILS for clearly out-of / in-scope terms.
5. RESCALE the +/-ref_max result to the configured output range [-bid_max, bid_max].

--positive-frac F (default 0.1) keeps ~F (0..1) of the papers positive by thresholding at the
F-quantile and rescaling each side to the full range, so the top papers still reach +/-bid_max.

If the CSV has neither `abstract` nor `topics`, it falls back to TITLE-ONLY mode.

All scoring parameters live in config.yaml (override with --config). PyYAML is used if
installed; otherwise a tiny built-in parser reads the flat key/value file. Reading PDFs
needs pypdf (pip install pypdf); nothing else is required.
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
    import yaml  # optional: nicer YAML support if PyYAML is installed
except ImportError:
    yaml = None

# config.yaml is the single source of truth for parameter values.
CONFIG_KEYS = (
    "bid_max", "bid_limit", "ref_max", "base_scale", "kw_cap",
    "compress_knee", "compress_factor", "strong_neg_threshold",
    "strong_neg_floor", "strong_pos_title_floor",
    "to_pos_cap", "to_neg_cap", "to_scale",
)
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

DEFAULT_PDF_DIR = "papers_pdf"
MIN_UNIQUE_PDFS = 5
DEFAULT_PROFILE = "reviewer-expertise-profile.json"

_STOP = set((
    "a an and or the of for to in on with by is are be as at from this that these those we our "
    "it its their than then them they you your not no").split())
_KW_STOP = _STOP | set((
    "been has have had having work works working prior more most can could would should may might "
    "via using used use show shows shown results result paper papers also such thus hence where while "
    "et al arxiv cs uk ac de university college london however based approach approaches propose "
    "proposes present presents number both two three first second new novel many several given due "
    "across toward towards within without other only over under between because each").split())

csv.field_size_limit(10 ** 7)  # abstracts can be long


# ------------------------------- config --------------------------------------
def _parse_simple_yaml(text):
    """Minimal flat 'key: value  # comment' scalar parser (PyYAML fallback)."""
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
    """Load every scoring parameter from the YAML config; validate; set globals."""
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
    g["BID_MAX"]                = cfg["bid_max"]
    g["BID_LIMIT"]              = cfg["bid_limit"]
    g["_REF_MAX"]               = cfg["ref_max"]
    g["BASE_SCALE"]             = cfg["base_scale"]
    g["KW_CAP"]                 = cfg["kw_cap"]
    g["COMPRESS_KNEE"]          = cfg["compress_knee"]
    g["COMPRESS_FACTOR"]        = cfg["compress_factor"]
    g["STRONG_NEG_THRESHOLD"]   = cfg["strong_neg_threshold"]
    g["STRONG_NEG_FLOOR"]       = cfg["strong_neg_floor"]
    g["STRONG_POS_TITLE_FLOOR"] = cfg["strong_pos_title_floor"]
    g["TO_POS_CAP"]             = cfg["to_pos_cap"]
    g["TO_NEG_CAP"]             = cfg["to_neg_cap"]
    g["TO_SCALE"]               = cfg["to_scale"]

    # validation + derived
    if not (isinstance(BID_MAX, int) and 1 <= BID_MAX <= BID_LIMIT):
        sys.exit("config: bid_max must be an int in [1, %d], got %r" % (BID_LIMIT, BID_MAX))
    g["BID_MIN"]  = -BID_MAX
    g["_REF_MIN"] = -_REF_MAX
    g["SCALE"]    = BID_MAX / _REF_MAX   # rescales a reference-span bid to the output range


# ------------------------- build the profile ---------------------------------
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


def _tokens(s):
    return [w for w in re.findall(r"[a-z][a-z\-]{1,}", s.lower()) if w not in _STOP]


def mine_positive_keywords(texts, topn=40):
    """Distinctive positive phrases (content-word bigrams) mined from your papers."""
    cnt = collections.Counter()
    for t in texts:
        tk = _tokens(t)
        for a, b in zip(tk, tk[1:]):
            if len(a) >= 3 and len(b) >= 3 and a not in _KW_STOP and b not in _KW_STOP:
                cnt[a + " " + b] += 1
    return [w for w, c in cnt.most_common(400) if c >= 3][:topn]


def read_topic_interests(path):
    """Read topic,interest CSV; interest is an integer in [-2, 2] (out-of-range values are clamped)."""
    if not os.path.exists(path):
        sys.exit("ERROR: topic interests file not found: '%s'\n"
                 "Create one with:  python3 make_topic_interests.py revprefs.csv" % path)
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]  # skip scale-comment lines
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


INTEREST_TO_AFFINITY = 10   # map user interest (-2..2) onto the internal scoring scale (-20..20)


def build_profile(interests_path, pdf_dir, min_pdfs, out_path, quiet=False):
    """Topic interests (your input, -2..2) + positive keywords mined from your PDFs -> profile JSON."""
    interests = read_topic_interests(interests_path)
    pdfs = require_pdfs(pdf_dir, min_pdfs)
    positives = mine_positive_keywords(read_pdf_texts(pdfs))
    profile = collections.OrderedDict()
    profile["meta"] = {
        "built": datetime.date.today().isoformat(),
        "topic_interests_file": interests_path,
        "topics": len(interests),
        "unique_pdfs": len(pdfs),
        "note": "Generated by score_bids.py - do not hand-edit. Topic interests (-2..2) come from %s "
                "and are mapped x%d onto the scoring scale; positive keywords are mined from %s/."
                % (os.path.basename(interests_path), INTEREST_TO_AFFINITY, pdf_dir),
    }
    profile["topic_affinity"] = collections.OrderedDict(
        (t, {"affinity": i * INTEREST_TO_AFFINITY, "interest": i}) for t, i in interests.items())
    profile["keyword_lexicon"] = {"strong_positive": {}, "positive": {w: 2 for w in positives}, "negative": {}}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, ensure_ascii=False)
    if not quiet:
        nz = sum(1 for i in interests.values() if i)
        print("Built %s: %d topics (%d with non-zero interest), %d mined keywords, %d PDFs."
              % (out_path, len(interests), nz, len(positives), len(pdfs)))
        if nz == 0:
            print("  NOTE: every topic interest is 0 - edit %s, then re-run." % interests_path)
    return out_path


# ------------------------------- helpers -------------------------------------
def clamp(x, lo=None, hi=None):
    lo = _REF_MIN if lo is None else lo
    hi = _REF_MAX if hi is None else hi
    return max(lo, min(hi, x))


def rnd(x):
    """Round half away from zero (so 17.5 -> 18, -17.5 -> -18)."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def load_profile(path):
    with open(path, encoding="utf-8") as fh:
        prof = json.load(fh)
    aff = {k: v["affinity"]
           for k, v in prof.get("topic_affinity", {}).items()
           if isinstance(v, dict) and "affinity" in v}
    lex = prof.get("keyword_lexicon", {})

    def bucket(name):  # drop "_comment" and similar meta keys
        return {k: w for k, w in lex.get(name, {}).items() if not k.startswith("_")}

    strong_pos = bucket("strong_positive")
    pos_all = {**strong_pos, **bucket("positive")}
    neg = bucket("negative")
    return aff, strong_pos, pos_all, neg


def compile_terms(term_weights):
    """Whole-word / phrase, case-insensitive matchers."""
    out = []
    for term, weight in term_weights.items():
        rx = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
        out.append((rx, weight))
    return out


# ------------------------------- scoring -------------------------------------
class Scorer:
    def __init__(self, profile_path):
        aff, strong_pos, pos_all, neg = load_profile(profile_path)
        self.aff = aff
        self.pos_m = compile_terms(pos_all)
        self.neg_m = compile_terms(neg)
        self.strong_m = [rx for rx, _ in compile_terms(strong_pos)]

    def topic_base(self, topics_field):
        topics = [t.strip() for t in (topics_field or "").split(";") if t.strip()]
        affs = [self.aff.get(t, 0) for t in topics]
        if not affs:
            return 0.0
        return 0.6 * max(affs) + 0.4 * (sum(affs) / len(affs))

    def keyword_sum(self, text):
        pos = sum(w for rx, w in self.pos_m if rx.search(text))
        neg = sum(w for rx, w in self.neg_m if rx.search(text))
        return pos + neg

    def value(self, row, has_topics, has_abstract):
        """Continuous reference-span score (everything score() does except the final rounding)."""
        title = row.get("title") or ""

        if not has_topics and not has_abstract:
            # ---------- TITLE-ONLY fallback ----------
            pos = min(TO_POS_CAP, sum(w for rx, w in self.pos_m if rx.search(title)))
            neg = min(TO_NEG_CAP, sum(-w for rx, w in self.neg_m if rx.search(title)))
            v = clamp((pos - neg) * TO_SCALE)
            if any(rx.search(title) for rx in self.strong_m):
                v = max(v, STRONG_POS_TITLE_FLOOR)
            return clamp(v)

        # ---------- ABSTRACT / TOPIC mode ----------
        base_s = self.topic_base(row.get("topics")) * BASE_SCALE
        text = (title + " ") * 2 + (row.get("abstract") or "")  # title double-weighted
        kw = clamp(self.keyword_sum(text), -KW_CAP, KW_CAP)
        raw = base_s + kw
        if raw > COMPRESS_KNEE:
            raw = COMPRESS_KNEE + (raw - COMPRESS_KNEE) * COMPRESS_FACTOR
        v = clamp(raw)

        # guardrails
        strong_neg = any(w <= STRONG_NEG_THRESHOLD and rx.search(text)
                         for rx, w in self.neg_m)
        if strong_neg and base_s <= 0:
            v = min(v, STRONG_NEG_FLOOR)
        if any(rx.search(title) for rx in self.strong_m):
            v = max(v, STRONG_POS_TITLE_FLOOR)
        return clamp(v)

    def score(self, row, has_topics, has_abstract):
        """Integer reference-span bid (value() rounded)."""
        return rnd(self.value(row, has_topics, has_abstract))


# ------------------------------- reporting -----------------------------------
def make_report(rows, mode):
    vals = [int(r["preference"]) for r in rows]
    dist = collections.Counter(vals)
    lines = []
    lines.append("Reviewer bidding report")
    lines.append("=" * 60)
    lines.append("mode            : %s" % mode)
    lines.append("papers scored   : %d" % len(rows))
    lines.append("mean bid        : %+.2f" % (sum(vals) / len(vals)))
    lines.append("want (>0)       : %d" % sum(v > 0 for v in vals))
    lines.append("neutral (=0)    : %d" % sum(v == 0 for v in vals))
    lines.append("avoid (<0)      : %d" % sum(v < 0 for v in vals))
    strong = int(round(BID_MAX * 0.75))  # "strong" cutoff scales with the output range
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
    lines.append("")
    lines.append("top 10 (highest bids)")
    for r in srt[:10]:
        lines.append("  %+3s  %s" % (r["preference"], (r.get("title") or "")[:78]))
    lines.append("")
    lines.append("bottom 10 (lowest bids)")
    for r in srt[-10:]:
        lines.append("  %+3s  %s" % (r["preference"], (r.get("title") or "")[:78]))
    return "\n".join(lines)


# ----------------------- target positive fraction ----------------------------
def bids_for_positive_fraction(values, target):
    """Map continuous reference scores to final bids so ~`target` of papers are positive.

    Rather than translating the whole distribution (which squashes the top toward 0), we put a
    threshold at the target quantile and rescale EACH SIDE to the full output range: the strongest
    paper reaches +bid_max, the weakest -bid_max, and the threshold lands at 0. So your top papers
    stay strongly positive even when only a small fraction are positive.
    """
    n = len(values)
    if n == 0:
        return []
    sv = sorted(values)
    idx = min(max(int(round((1.0 - target) * n)), 0), n - 1)
    tau = sv[idx]                              # ~`target` of papers have value > tau
    hi = (sv[-1] - tau) or 1.0
    lo = (tau - sv[0]) or 1.0
    out = []
    for v in values:
        if v > tau:                            # positive side -> [1, bid_max]
            out.append(int(clamp(1 + rnd((v - tau) / hi * (BID_MAX - 1)), 1, BID_MAX)))
        else:                                  # threshold and below -> [-bid_max, 0]
            out.append(int(clamp(rnd((v - tau) / lo * BID_MAX), BID_MIN, 0)))
    return out


# --------------------------------- main --------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build a profile from your topic interests + papers, and fill bids (see config.yaml).")
    ap.add_argument("input", help="preferences CSV (columns: paper,title,preference[,abstract,topics])")
    ap.add_argument("--topic-interests", dest="topic_interests", default="topic_interests.csv",
                    help="your topic,interest CSV in [-2,2] (default: topic_interests.csv; "
                         "make one with make_topic_interests.py)")
    ap.add_argument("--pdfs", default=DEFAULT_PDF_DIR,
                    help="folder of your paper PDFs, for keyword mining (default: %s; at least %d "
                         "unique PDFs are required)" % (DEFAULT_PDF_DIR, MIN_UNIQUE_PDFS))
    ap.add_argument("--profile-out", dest="profile_out", default=DEFAULT_PROFILE,
                    help="where to save the built profile JSON (default: %s)" % DEFAULT_PROFILE)
    ap.add_argument("--positive-frac", dest="positive_frac", type=float, default=0.1, metavar="F",
                    help="target fraction (0..1) of papers to bid positively on, kept within +/-10 "
                         "percentage points by shifting the whole distribution (default: 0.1)")
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
    if args.positive_frac is not None and not (0.0 <= args.positive_frac <= 1.0):
        ap.error("--positive-frac must be a float between 0 and 1")
    load_config(args.config)  # required; exits with a clear message if missing/invalid

    # ---- build the profile from your topic interests + papers (saves the JSON) ----
    build_profile(args.topic_interests, args.pdfs, MIN_UNIQUE_PDFS, args.profile_out, quiet=args.quiet)

    # read
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
    mode = "abstract/topics" if (has_topics or has_abstract) else "title-only"

    # score (algorithm runs on the +/-_REF_MAX span, then rescale to the output range)
    scorer = Scorer(args.profile_out)
    target_note = None
    if args.positive_frac is None:
        for r in rows:
            ref_bid = scorer.score(r, has_topics, has_abstract)
            r["preference"] = str(rnd(clamp(ref_bid * SCALE, BID_MIN, BID_MAX)))
    else:
        # keep ~positive_frac of papers positive WITHOUT squashing the top (threshold + rescale)
        vals = [scorer.value(r, has_topics, has_abstract) for r in rows]
        prefs = bids_for_positive_fraction(vals, args.positive_frac)
        for r, p in zip(rows, prefs):
            r["preference"] = str(p)
        achieved = sum(1 for p in prefs if p > 0) / len(prefs)
        target_note = ("target positive: %.0f%%  ->  achieved: %.0f%%"
                       % (100 * args.positive_frac, 100 * achieved))
        if abs(achieved - args.positive_frac) > 0.10:
            sys.stderr.write("WARNING: " + target_note +
                             "  - more than 10 percentage points off (score distribution too lumpy).\n")

    # decide output path
    if args.output:
        out_path = args.output
    else:
        stem, ext = os.path.splitext(args.input)
        out_path = stem + ".scored" + (ext or ".csv")

    # write the scored CSV WITHOUT the abstract column (keep the rest, original order)
    out_fields = [f for f in fields if f != "abstract"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # by default delete the original (abstract-laden) input; --keep-original keeps it
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
        if target_note:
            print(target_note)
        tail = "\nwrote %d bids -> %s" % (len(rows), out_path)
        if removed_original:
            tail += "  (deleted original %s; use --keep-original to keep it)" % args.input
        print(tail)


if __name__ == "__main__":
    main()
