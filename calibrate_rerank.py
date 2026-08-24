#!/usr/bin/env python3
"""
calibrate_rerank.py - measure what pruning the paper axis would cost you, before doing it.

--method rerank cross-encodes every shortlisted submission against EVERY paper in papers_pdf/,
then keeps only its best RERANK_AGG_TOPK. With 13 papers, 10 of every 13 passes are paid for and
discarded. They could be skipped by scoring each submission against only its most promising
papers, chosen by the TF-IDF similarities the run already computes -- but that can only ever
LOWER a score (a high scorer can be missed, never invented), so it needs evidence first.

This measures the evidence on YOUR data. It cross-encodes a sample of candidates against all
papers as ground truth, then reports what each pruning depth would have done to them:

    recall    - of a candidate's true best RERANK_AGG_TOPK papers, how many the TF-IDF top-m held
    score err - how far the aggregated score moved
    spearman  - whether the ORDERING survived, which is all that reaches your bids, since
                everything downstream of this is rank-based

Read-only: it never writes to or deletes your input CSV, and never writes bids.

    python3 calibrate_rerank.py revprefs.csv
    python3 calibrate_rerank.py revprefs.csv --sample 300 --report calib.txt

Needs sentence-transformers, like --method rerank itself.
"""

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_bids as sb   # noqa: E402


def _pearson(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))


def _spearman(a, b):
    """Pearson on tie-averaged ranks. Reuses the scorer's own rank transform, so ties are
    handled the same way here as in the pipeline."""
    return _pearson(sb._quantile_ranks(list(a)), sb._quantile_ranks(list(b)))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Measure what pruning rerank's paper axis would cost, on your own data.")
    ap.add_argument("input", help="preferences CSV (paper,title,preference,abstract[,topics])")
    ap.add_argument("--pdfs", default=sb.DEFAULT_PDF_DIR,
                    help="folder of your paper PDFs (default: %s)" % sb.DEFAULT_PDF_DIR)
    ap.add_argument("--config", default=sb.DEFAULT_CONFIG, help="scoring parameters YAML")
    ap.add_argument("--positive-frac", dest="positive_frac", type=float, default=0.1, metavar="F",
                    help="the target you'd score with; sets which submissions are shortlisted "
                         "(default: 0.1)")
    ap.add_argument("--rerank-frac", dest="rerank_frac", type=float, default=None, metavar="F",
                    help="shortlist fraction you'd score with (default: same rule as score_bids)")
    ap.add_argument("--sample", type=int, default=200, metavar="N",
                    help="how many shortlisted submissions to measure (default: 200). Cost is "
                         "N x your paper count, in cross-encoder pairs.")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed; fixed so the measurement is reproducible (default: 0)")
    ap.add_argument("--target-spearman", dest="target", type=float, default=0.99, metavar="R",
                    help="recommend the cheapest depth holding at least this rank correlation "
                         "(default: 0.99)")
    ap.add_argument("--rerank-model", dest="rerank_model", default=sb.DEFAULT_RERANK_MODEL,
                    help="cross-encoder model id (default: %s)" % sb.DEFAULT_RERANK_MODEL)
    ap.add_argument("--rerank-max-length", dest="max_length", type=int,
                    default=sb.RERANK_MAX_LENGTH, metavar="N",
                    help="joint token budget per pair (default: %d)" % sb.RERANK_MAX_LENGTH)
    ap.add_argument("--report", help="also write the table to this path")
    args = ap.parse_args(argv)

    if not os.path.exists(args.input):
        ap.error("input not found: %s" % args.input)
    if args.sample < 2:
        ap.error("--sample must be at least 2 (a correlation needs more than one point)")
    sb.load_config(args.config)

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        sys.exit("calibration needs sentence-transformers:\n    pip install sentence-transformers")
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import warnings
    sb._quiet_library_warnings(warnings)

    import csv
    with open(args.input, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        ap.error("input has no data rows")

    pdfs = sb.require_pdfs(args.pdfs, sb.MIN_UNIQUE_PDFS)
    pdfs, paper_texts = sb.usable_paper_texts(pdfs, args.pdfs, sb.MIN_UNIQUE_PDFS)
    n_papers = len(pdfs)
    agg_k = sb.RERANK_AGG_TOPK
    if n_papers <= agg_k + 1:
        sys.exit("Nothing to calibrate: %d usable paper(s) and the score already averages the top "
                 "%d, so there is no paper axis to prune." % (n_papers, agg_k))

    # ---- stage 1, exactly as a scoring run does it ----
    sub_pairs = [((r.get("title") or ""), (r.get("abstract") or "")) for r in rows]
    sub_texts = [(t + ". " + a) for t, a in sub_pairs]
    S, P, _vec = sb._tfidf_fit(paper_texts, sub_texts)
    sim = cosine_similarity(S, P)                      # submissions x papers -- normally discarded
    tfidf_best = sb._topk_mean(sim, k=agg_k)
    n = len(rows)
    frac = sb.rerank_shortlist_frac(args.positive_frac, args.rerank_frac)
    topn = sb.rerank_topn(n, frac)
    cand = sorted(range(n), key=lambda i: tfidf_best[i], reverse=True)[:topn]

    sample = min(args.sample, len(cand))
    picked = random.Random(args.seed).sample(cand, sample)
    pairs_needed = sample * n_papers
    print("paper-axis calibration")
    print("  submissions      : %d   shortlist %d (--rerank-frac %.2f)" % (n, topn, frac))
    print("  papers           : %d   aggregation keeps the top %d" % (n_papers, agg_k))
    print("  sampling         : %d shortlisted submission(s), seed %d" % (sample, args.seed))
    print("  ground-truth cost: %d x %d = %s cross-encoder pair(s), max_length %d\n"
          % (sample, n_papers, format(pairs_needed, ","), args.max_length))
    sys.stdout.flush()

    # ---- ground truth: every sampled candidate against every paper ----
    pap_texts = sb.paper_pair_texts(paper_texts)
    sys.stderr.write("Loading reranker (%s)...\n" % args.rerank_model)
    sys.stderr.flush()
    ce = CrossEncoder(args.rerank_model, max_length=args.max_length)
    ce.model.eval()
    flat = [(pap_texts[p], sub_texts[i]) for i in picked for p in range(n_papers)]
    logits = ce.predict(flat, batch_size=32, convert_to_numpy=True,
                        show_progress_bar=(len(flat) > 200))
    ce_mat = np.asarray(logits, dtype="float64").reshape(sample, n_papers)

    full = sb._topk_mean(ce_mat, k=agg_k)              # the score a full run would produce
    true_top = [set(np.argsort(-ce_mat[r])[:agg_k]) for r in range(sample)]

    lines = ["", "  papers scored    recall   mean |score err|    spearman   saving"]
    best_m = None
    for m in range(agg_k, n_papers + 1):
        pruned, hits = [], 0.0
        for r, i in enumerate(picked):
            sel = np.argsort(-sim[i])[:m]              # TF-IDF's top-m papers for this submission
            pruned.append(sb._topk_mean(ce_mat[r:r + 1, sel], k=agg_k)[0])
            hits += len(true_top[r] & set(sel)) / float(agg_k)
        recall = hits / sample
        err = float(np.mean(np.abs(np.asarray(pruned) - full)))
        rho = _spearman(pruned, full)
        lines.append("  %4d (%3.0f%%)      %5.2f          %9.4f      %8.5f   %4.1fx"
                     % (m, 100.0 * m / n_papers, recall, err, rho, float(n_papers) / m))
        if best_m is None and not math.isnan(rho) and rho >= args.target:
            best_m = m

    lines.append("")
    if best_m is None:
        lines.append("  No depth holds spearman >= %.3f -- pruning would reorder your bids. "
                     "Keep scoring against all papers." % args.target)
    elif best_m >= n_papers:
        lines.append("  Only the full set holds spearman >= %.3f. Nothing to save here."
                     % args.target)
    else:
        lines.append("  Cheapest depth holding spearman >= %.3f: %d of %d papers (%.1fx less "
                     "cross-encoder work)." % (args.target, best_m, n_papers,
                                               float(n_papers) / best_m))
        lines.append("  As a fraction to pass to a future --rerank-papers: %.2f"
                     % (math.ceil(100.0 * best_m / n_papers) / 100.0))
    lines.append("")
    lines.append("  Sampled from the shortlist, so this describes the submissions pruning would")
    lines.append("  actually be applied to. Re-run per venue: it depends on your papers and pool.")
    out = "\n".join(lines)
    print(out)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("paper-axis calibration\n" + out + "\n")
        print("\n  wrote %s" % args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
