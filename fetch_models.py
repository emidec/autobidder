#!/usr/bin/env python3
"""
fetch_models.py - download the models score_bids.py needs, before you need them.

--method tfidf needs no model. The other two each download one on first use, which means a
scoring run can sit there for minutes with nothing to show for it. Fetch them separately
instead, then every later run is offline and instant to start.

    python3 fetch_models.py rerank            # cross-encoder for --method rerank   (~2.3 GB)
    python3 fetch_models.py specter2          # embeddings for --method specter2    (~0.5 GB)
    python3 fetch_models.py all               # both
    python3 fetch_models.py rerank --check    # report what's cached, download nothing

Models land in the shared HuggingFace cache (~/.cache/huggingface, or HF_HOME if set), which
is where score_bids.py looks -- so this doesn't need to know anything about your project.

Downloading uses the same code path the scoring run uses, so what it caches is exactly what
the run needs -- including the SPECTER2 adapter, which lives in its own repository.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bids import (DEFAULT_RERANK_MODEL, RERANK_MAX_LENGTH,   # noqa: E402
                        SPECTER2_ADAPTER, SPECTER2_MODEL)

APPROX_SIZE = {"rerank": "~2.3 GB", "specter2": "~0.5 GB + adapter"}


def hf_cache_dir():
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def cached_repos():
    """Repo ids already in the HuggingFace cache, as {'org/name', ...}."""
    hub = hf_cache_dir()
    if not os.path.isdir(hub):
        return set()
    out = set()
    for name in os.listdir(hub):
        if name.startswith("models--"):
            out.add(name[len("models--"):].replace("--", "/"))
    return out


def report(repos, cached):
    for repo in repos:
        print("  %-34s %s" % (repo, "cached" if repo in cached else "NOT cached"))
    return all(r in cached for r in repos)


def fetch_rerank(model_id, check):
    repos = [model_id]
    cached = cached_repos()
    size = APPROX_SIZE["rerank"] if model_id == DEFAULT_RERANK_MODEL else "size unknown"
    print("rerank: cross-encoder for --method rerank  (%s)" % size)
    complete = report(repos, cached)
    if check or complete:
        print("  nothing to download." if complete else "  --check: not downloading.")
        return complete
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        sys.exit("  ERROR: needs sentence-transformers:  pip install sentence-transformers")
    print("  downloading via the same path the scoring run uses...")
    CrossEncoder(model_id, max_length=RERANK_MAX_LENGTH)
    return report(repos, cached_repos())


def fetch_specter2(check):
    repos = [SPECTER2_MODEL, SPECTER2_ADAPTER]
    cached = cached_repos()
    print("specter2: embeddings for --method specter2  (%s)" % APPROX_SIZE["specter2"])
    complete = report(repos, cached)
    if check or complete:
        print("  nothing to download." if complete else "  --check: not downloading.")
        return complete
    try:
        from transformers import AutoTokenizer
        try:
            from adapters import AutoAdapterModel
        except ImportError:
            from transformers.adapters import AutoAdapterModel
    except ImportError:
        sys.exit("  ERROR: needs torch, transformers and adapters:\n"
                 "         pip install torch transformers adapters")
    print("  downloading base model and proximity adapter...")
    AutoTokenizer.from_pretrained(SPECTER2_MODEL)
    model = AutoAdapterModel.from_pretrained(SPECTER2_MODEL)
    model.load_adapter(SPECTER2_ADAPTER, source="hf", load_as="proximity", set_active=True)
    return report(repos, cached_repos())


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Download the models score_bids.py needs for --method specter2 / rerank.")
    ap.add_argument("method", choices=["rerank", "specter2", "all"],
                    help="which model(s) to fetch (tfidf needs none)")
    ap.add_argument("--check", action="store_true",
                    help="report what's already cached and exit without downloading")
    ap.add_argument("--rerank-model", dest="rerank_model", default=DEFAULT_RERANK_MODEL,
                    help="cross-encoder to fetch instead of the default (%s); pass the same id "
                         "to score_bids.py --rerank-model" % DEFAULT_RERANK_MODEL)
    args = ap.parse_args(argv)

    print("HuggingFace cache: %s\n" % hf_cache_dir())
    ok = True
    if args.method in ("rerank", "all"):
        ok = fetch_rerank(args.rerank_model, args.check) and ok
        print("")
    if args.method in ("specter2", "all"):
        ok = fetch_specter2(args.check) and ok
        print("")
    if args.check:
        print("Everything needed is cached." if ok else
              "Missing model(s) above -- re-run without --check to download.")
        return 0 if ok else 1
    print("Ready. Later runs load from the cache and need no network."
          if ok else "Something is still missing -- see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
