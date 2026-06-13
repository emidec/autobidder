#!/usr/bin/env python3
"""
make_topic_interests.py - create a blank topic_interests.csv from a preferences CSV.

Reads the conference's topic list (the `topics` column - column E - of the HotCRP
reviewer-preferences export) and writes topic_interests.csv with every distinct topic
defaulted to 0 (neutral), for you to fill in.

Please indicate your interest in reviewing papers on these conference topics, on a
-2..2 scale:

    2  very high      1  high      0  neutral      -1  low      -2  very low

Edit the `interest` column, then build your profile + bids:

    python3 score_bids.py revprefs.csv --topic-interests topic_interests.csv

Usage:
    python3 make_topic_interests.py revprefs.csv
    python3 make_topic_interests.py revprefs.csv -o topic_interests.csv [--force]

Standard library only.
"""

import argparse
import csv
import os
import sys

# Written verbatim (as # comments, which score_bids.py ignores) at the top of the file,
# so the prompt and scale are documented right where you edit.
SCALE_HEADER = [
    "# topic_interests.csv",
    "# Please indicate your interest in reviewing papers on these conference topics.",
    "# INTEREST: an integer from -2 to 2.",
    "#   2 = very high     1 = high     0 = neutral     -1 = low     -2 = very low",
    "# Edit the 'interest' column (leave 0 for neutral), then run:",
    "#   python3 score_bids.py revprefs.csv --topic-interests topic_interests.csv",
]


def topics_from_csv(path):
    """Distinct, sorted topics from the 'topics' column (column E), ';'-separated."""
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            sys.exit("ERROR: %s is empty." % path)
        idx = next((i for i, h in enumerate(header) if h.strip().lower() == "topics"), None)
        if idx is None:
            if len(header) >= 5:
                idx = 4  # column E
            else:
                sys.exit("ERROR: no 'topics' column (column E) found in %s." % path)
        topics = set()
        for row in reader:
            if idx < len(row):
                for t in row[idx].split(";"):
                    t = t.strip()
                    if t:
                        topics.add(t)
    return sorted(topics)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Create a blank topic_interests.csv (every topic at 0) from a preferences CSV.")
    ap.add_argument("csv", help="preferences CSV (reads the 'topics' column / column E)")
    ap.add_argument("-o", "--output", default="topic_interests.csv",
                    help="output file (default: topic_interests.csv)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the output if it already exists")
    args = ap.parse_args(argv)

    if not os.path.exists(args.csv):
        ap.error("not found: %s" % args.csv)
    if os.path.exists(args.output) and not args.force:
        ap.error("%s already exists - use --force to overwrite (you'd lose any edits)." % args.output)

    topics = topics_from_csv(args.csv)
    if not topics:
        sys.exit("ERROR: no topics found in the 'topics' column of %s." % args.csv)

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        for line in SCALE_HEADER:
            fh.write(line + "\n")
        w = csv.writer(fh)
        w.writerow(["topic", "interest"])
        for t in topics:
            w.writerow([t, 0])

    print("Wrote %s - %d topics, all at 0 (neutral)." % (args.output, len(topics)))
    print("INTEREST: -2 very low / -1 low / 0 neutral / 1 high / 2 very high.")
    print("Edit the 'interest' column, then:  python3 score_bids.py revprefs.csv --topic-interests %s" % args.output)


if __name__ == "__main__":
    main()
