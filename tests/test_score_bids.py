#!/usr/bin/env python3
"""Regression tests for score_bids.py and make_topic_interests.py.

Stdlib unittest only -- no test dependency to install:

    python3 -m unittest discover -s tests

Most of the scoring is pure functions, which is what this leans on. load_config writes the
config into module globals, so setUpModule loads it once for the whole suite.
"""

import csv
import io
import math
import os
import subprocess
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import make_topic_interests as mti          # noqa: E402
import score_bids as sb                     # noqa: E402


def setUpModule():
    sb.load_config(os.path.join(ROOT, "config.yaml"))


def write(path, text):
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def write_csv(path, rows, fields=("paper", "title", "preference", "abstract", "topics")):
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


class BidMapping(unittest.TestCase):
    """bids_for_positive_fraction: the --positive-frac contract."""

    def test_target_is_hit_exactly(self):
        for n in (2, 10, 20, 137, 800):
            for target in (0.0, 0.05, 0.1, 0.3, 0.5):
                bids = sb.bids_for_positive_fraction([float(v) for v in range(n)], target)
                got = sum(1 for b in bids if b > 0)
                self.assertEqual(int(round(target * n)), got,
                                 "n=%d target=%s: wanted %d positive, got %d"
                                 % (n, target, int(round(target * n)), got))

    def test_stays_within_bid_max(self):
        bids = sb.bids_for_positive_fraction([float(v) for v in range(200)], 0.2)
        self.assertLessEqual(max(bids), sb.BID_MAX)
        self.assertGreaterEqual(min(bids), sb.BID_MIN)

    def test_both_rails_are_reached(self):
        bids = sb.bids_for_positive_fraction([float(v) for v in range(200)], 0.2)
        self.assertEqual(sb.BID_MAX, max(bids))
        self.assertEqual(sb.BID_MIN, min(bids))

    def test_target_one_tops_out_one_short(self):
        # the comparison is strict, so some paper must be the lowest -- documented behavior
        bids = sb.bids_for_positive_fraction([float(v) for v in range(20)], 1.0)
        self.assertEqual(19, sum(1 for b in bids if b > 0))

    def test_degenerate_inputs(self):
        self.assertEqual([], sb.bids_for_positive_fraction([], 0.1))
        self.assertEqual([0] * 6, sb.bids_for_positive_fraction([5.0] * 6, 0.1))
        self.assertEqual(1, len(sb.bids_for_positive_fraction([1.0], 0.1)))

    def test_scale_invariant(self):
        a = sb.bids_for_positive_fraction([float(v) for v in range(50)], 0.1)
        b = sb.bids_for_positive_fraction([float(v) * 1000 for v in range(50)], 0.1)
        self.assertEqual(a, b)


class Numerics(unittest.TestCase):
    def test_quantile_ties_share_a_rank(self):
        q = sb._quantile_ranks([1, 1, 1, 2, 2])
        self.assertEqual(q[0], q[1])
        self.assertEqual(q[1], q[2])
        self.assertEqual(q[3], q[4])
        self.assertLess(q[0], q[3])

    def test_all_tied_lands_mid_scale(self):
        # an all-zero similarity matrix must not be spread across the scale by tie order
        self.assertEqual([0.5] * 8, sb._quantile_ranks([0.0] * 8))

    def test_rnd_rounds_half_away_from_zero(self):
        self.assertEqual([-2, -1, 0, 1, 2], [sb.rnd(x) for x in (-1.5, -0.5, 0.0, 0.5, 1.5)])

    def test_signpow_keeps_sign(self):
        self.assertAlmostEqual(-0.25, sb._signpow(-0.5, 2))
        self.assertAlmostEqual(0.25, sb._signpow(0.5, 2))

    def test_clamp(self):
        self.assertEqual(5, sb.clamp(9, 0, 5))
        self.assertEqual(0, sb.clamp(-9, 0, 5))


class TitleAbstractParsing(unittest.TestCase):
    def test_parses_title_and_abstract(self):
        raw = "A Great Paper\nAbstract\n" + ("privacy " * 30) + "\n1 Introduction\nbody"
        title, abstract = sb._split_title_abstract(raw)
        self.assertEqual("A Great Paper", title)
        self.assertIn("privacy", abstract)
        self.assertNotIn("Introduction", abstract)

    def test_returns_none_when_no_abstract(self):
        self.assertEqual((None, None), sb._split_title_abstract("Just a title\nand some body text"))

    def test_returns_none_when_abstract_too_short(self):
        self.assertEqual((None, None), sb._split_title_abstract("T\nAbstract\ntiny\n1 Introduction"))


class TopicInterests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_reports_unparsable_and_out_of_range(self):
        p = write(os.path.join(self.dir, "ti.csv"),
                  "topic,interest\nA,5\nB,high\nC,1.5\nD,-99\nE,2\nF,\n")
        err = io.StringIO()
        old, sys.stderr = sys.stderr, err
        try:
            got = sb.read_topic_interests(p)
        finally:
            sys.stderr = old
        self.assertEqual({"A": 2, "B": 0, "C": 2, "D": -2, "E": 2, "F": 0}, dict(got))
        msg = err.getvalue()
        for topic in ("A", "B", "C", "D"):
            self.assertIn(topic, msg)
        self.assertNotIn("\nE ", msg)          # valid rows aren't reported

    def test_comments_are_skipped(self):
        p = write(os.path.join(self.dir, "ti.csv"), "# a comment\ntopic,interest\nA,1\n")
        self.assertEqual({"A": 1}, dict(sb.read_topic_interests(p)))

    def test_missing_file_exits(self):
        with self.assertRaises(SystemExit):
            sb.read_topic_interests(os.path.join(self.dir, "nope.csv"))


class TopicCoverage(unittest.TestCase):
    def _run(self, rows, interests):
        err = io.StringIO()
        old, sys.stderr = sys.stderr, err
        try:
            sb.report_topic_coverage(rows, interests)
        finally:
            sys.stderr = old
        return err.getvalue()

    def test_flags_tags_with_no_interest_row(self):
        out = self._run([{"topics": "Privacy;Networking"}], {"Privacy": 2})
        self.assertIn("Networking", out)

    def test_flags_interest_rows_matching_nothing(self):
        out = self._run([{"topics": "Privacy"}], {"Privacy": 2, "Stale Topic": 1})
        self.assertIn("Stale Topic", out)

    def test_silent_when_no_topics_column(self):
        self.assertEqual("", self._run([{"title": "x"}], {"Privacy": 2}))


class PdfTextRequirement(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_drops_textless_and_keeps_the_rest(self):
        paths = [write(os.path.join(self.dir, "p%d.pdf" % i), "x") for i in range(6)]
        texts = ["real extracted text " * 30] * 5 + [""]
        sb_read = sb.read_pdf_texts
        sb.read_pdf_texts = lambda p, pages=3: texts
        err = io.StringIO()
        old, sys.stderr = sys.stderr, err
        try:
            kept, kept_texts = sb.usable_paper_texts(paths, self.dir, 5)
        finally:
            sys.stderr, sb.read_pdf_texts = old, sb_read
        self.assertEqual(5, len(kept))
        self.assertEqual(5, len(kept_texts))
        self.assertIn("p5.pdf", err.getvalue())

    def test_exits_when_too_few_have_text(self):
        paths = [write(os.path.join(self.dir, "p%d.pdf" % i), "x") for i in range(6)]
        sb_read = sb.read_pdf_texts
        sb.read_pdf_texts = lambda p, pages=3: ["real text " * 30] * 2 + [""] * 4
        err, old = io.StringIO(), sys.stderr
        sys.stderr = err
        try:
            with self.assertRaises(SystemExit):
                sb.usable_paper_texts(paths, self.dir, 5)
        finally:
            sys.stderr, sb.read_pdf_texts = old, sb_read


class VocabularyGuards(unittest.TestCase):
    def _varied(self, n):
        words = ("privacy attack gradient federated encryption proof routing packet transformer "
                 "embedding kernel scheduler latency cache timing verification synthesis").split()
        return [" ".join(words[i % len(words):][:5] + words[:3]) + " unique%d" % i for i in range(n)]

    def test_floor_matches_the_thresholds(self):
        # the floor must be the smallest n where max_df admits min_df documents
        self.assertGreaterEqual(int(sb.TFIDF_MAX_DF * sb.TFIDF_MIN_SUBMISSIONS), sb.TFIDF_MIN_DF)
        self.assertLess(int(sb.TFIDF_MAX_DF * (sb.TFIDF_MIN_SUBMISSIONS - 1)), sb.TFIDF_MIN_DF)

    def test_small_pool_exits_cleanly(self):
        with self.assertRaises(SystemExit):
            sb._tfidf_fit(["paper text"], self._varied(sb.TFIDF_MIN_SUBMISSIONS - 1))

    def test_uniform_pool_exits_cleanly(self):
        with self.assertRaises(SystemExit):
            sb._tfidf_fit(["paper text"], ["identical wording here"] * 40)

    def test_not_required_degrades_instead_of_exiting(self):
        err = io.StringIO()
        old, sys.stderr = sys.stderr, err
        try:
            got = sb._tfidf_fit(["paper text"], ["identical wording here"] * 40, required=False)
        finally:
            sys.stderr = old
        self.assertEqual((None, None, None), got)
        self.assertIn("WARNING", err.getvalue())


class RerankShortlist(unittest.TestCase):
    """The shortlist must cover the positive band; headroom scales with the pool."""

    def test_default_adds_the_margin_to_the_target(self):
        self.assertAlmostEqual(0.20, sb.rerank_shortlist_frac(0.10))
        self.assertAlmostEqual(0.35, sb.rerank_shortlist_frac(0.25))

    def test_default_never_falls_below_the_target(self):
        # the correctness condition: positive bids come from the reranked set
        for i in range(1, 101):
            pf = i / 100.0
            self.assertGreaterEqual(sb.rerank_shortlist_frac(pf) + 1e-12, pf,
                                    "positive_frac=%.2f" % pf)

    def test_default_is_capped_until_the_target_exceeds_the_cap(self):
        self.assertAlmostEqual(sb.RERANK_MAX_FRAC, sb.rerank_shortlist_frac(0.45))
        self.assertAlmostEqual(0.60, sb.rerank_shortlist_frac(0.60))   # target wins over the cap

    def test_margin_is_additive_in_the_pool_not_multiplied_by_the_target(self):
        # a multiplicative cushion would give a margin proportional to the target; this must not
        margins = [sb.rerank_shortlist_frac(pf) - pf for pf in (0.02, 0.10, 0.30)]
        for m in margins:
            self.assertAlmostEqual(sb.RERANK_MARGIN_FRAC, m)

    def test_explicit_fraction_is_returned_verbatim(self):
        self.assertAlmostEqual(0.42, sb.rerank_shortlist_frac(0.1, 0.42))

    def test_topn_converts_fraction_to_count(self):
        self.assertEqual(289, sb.rerank_topn(1442, 0.20))
        self.assertEqual(1442, sb.rerank_topn(1442, 1.0))

    def test_topn_is_clamped_to_the_pool_and_at_least_one(self):
        self.assertEqual(10, sb.rerank_topn(10, 1.0))
        self.assertEqual(1, sb.rerank_topn(10, 0.0001))
        self.assertEqual(5, sb.rerank_topn(5, 2.0))


class PreviousOutputLookup(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_picks_latest_earlier_stamp_not_newest_mtime(self):
        for ts in ("20260101-000000", "20260102-000000", "20260103-000000"):
            write(os.path.join(self.dir, "r.scored.%s.csv" % ts), "paper,preference\n1,5\n")
        old = os.path.join(self.dir, "r.scored.20260101-000000.csv")
        os.utime(old, (2 ** 31 - 1, 2 ** 31 - 1))          # make the oldest run the newest file
        got = sb._find_previous_output(
            os.path.join(self.dir, "r.scored.20260104-000000.csv"), "20260104-000000")
        self.assertEqual("r.scored.20260103-000000.csv", os.path.basename(got))

    def test_ignores_later_stamps(self):
        write(os.path.join(self.dir, "r.scored.20260105-000000.csv"), "paper,preference\n1,5\n")
        got = sb._find_previous_output(
            os.path.join(self.dir, "r.scored.20260104-000000.csv"), "20260104-000000")
        self.assertIsNone(got)

    def test_fixed_output_uses_the_file_it_replaces(self):
        p = write(os.path.join(self.dir, "filled.csv"), "paper,preference\n1,5\n")
        self.assertEqual(p, sb._find_previous_output(p, "20260104-000000"))

    def test_fixed_output_first_run_has_no_baseline(self):
        self.assertIsNone(sb._find_previous_output(
            os.path.join(self.dir, "absent.csv"), "20260104-000000"))


class ChangeSummary(unittest.TestCase):
    def test_first_run_says_so(self):
        out = sb.make_change_summary(None, {}, [{"paper": "1", "preference": "5", "title": "T"}])
        self.assertIn("first run", out)

    def test_counts_changed_added_and_removed(self):
        prev = {"1": (5, "One"), "2": (-3, "Two"), "3": (0, "Three")}
        rows = [{"paper": "1", "preference": "5", "title": "One"},      # unchanged
                {"paper": "2", "preference": "9", "title": "Two"},      # changed, into positive
                {"paper": "4", "preference": "1", "title": "Four"}]     # added; 3 removed
        out = sb.make_change_summary("prev.csv", prev, rows)
        self.assertIn("unchanged : 1", out)
        self.assertIn("changed   : 1", out)
        self.assertIn("into positive  (<=0 -> >0) : 1", out)
        self.assertIn("added     : 1", out)
        self.assertIn("removed   : 1", out)

    def test_names_the_profile_when_given(self):
        out = sb.make_change_summary(None, {}, [{"paper": "1", "preference": "5"}],
                                     profile_path="/x/profile.20260101-000000.json")
        self.assertIn("profile.20260101-000000.json", out)


class WarningScoping(unittest.TestCase):
    def test_numeric_warnings_survive_library_silencing(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sb._quiet_library_warnings(warnings)
            warnings.warn("library chatter", UserWarning)
            warnings.warn("deprecated", FutureWarning)
            warnings.warn("numeric trouble", RuntimeWarning)
        self.assertEqual(["RuntimeWarning"], [c.category.__name__ for c in caught])


class MakeTopicInterests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_reads_the_topics_column(self):
        p = write_csv(os.path.join(self.dir, "r.csv"),
                      [{"paper": "1", "title": "T", "preference": "", "abstract": "a",
                        "topics": "Privacy;Systems"}])
        self.assertEqual(["Privacy", "Systems"], mti.topics_from_csv(p))

    def test_no_positional_fallback(self):
        p = write_csv(os.path.join(self.dir, "r.csv"),
                      [{"paper": "1", "title": "T", "preference": "", "abstract": "x; y; z",
                        "extra": "e"}],
                      fields=("paper", "title", "preference", "abstract", "extra"))
        with self.assertRaises(SystemExit):
            mti.topics_from_csv(p)


class ColumnValidation(unittest.TestCase):
    """End-to-end: the CLI must reject an export it can't upload."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        write(os.path.join(self.dir, "topic_interests.csv"), "topic,interest\nPrivacy,1\n")

    def _run(self, name, fields):
        rows = [{"paper": str(i), "title": "T%d" % i, "preference": "",
                 "abstract": "privacy attack %d gradient" % i, "topics": "Privacy"}
                for i in range(10)]
        path = write_csv(os.path.join(self.dir, name), rows, fields=fields)
        return subprocess.run([sys.executable, os.path.join(ROOT, "score_bids.py"), path,
                               "--keep-original", "--quiet"],
                              cwd=self.dir, capture_output=True, text=True)

    def test_missing_paper_column_is_an_error(self):
        r = self._run("no_paper.csv", ("title", "preference", "abstract", "topics"))
        self.assertNotEqual(0, r.returncode)
        self.assertIn("paper", r.stderr)

    def test_missing_preference_column_is_an_error(self):
        r = self._run("no_pref.csv", ("paper", "title", "abstract", "topics"))
        self.assertNotEqual(0, r.returncode)
        self.assertIn("preference", r.stderr)


if __name__ == "__main__":
    unittest.main()
