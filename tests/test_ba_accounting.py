"""Stage 01a: BA solve accounting (spec C4).

Drives pipeline.record_ba_outcome with a scripted sequence and asserts the identities
  attempted = skipped + completed + errors,   completed = accepted + rejected,
the ba_success / ba_fail aliases, and the exact summary-line formats the verifier parses.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import ba_summary_lines, record_ba_outcome  # noqa: E402

SCRIPT = [
    ("skipped", None, "too_few_points"),
    ("completed", True, None),
    ("completed", False, None),
    ("error", None, None),
    ("completed", True, None),
    ("skipped", None, "gauge_unanchored"),
    ("skipped", None, "too_few_points"),
    ("completed", True, None),
    ("error", None, None),
    ("completed", False, None),
]


def _run(stats):
    for status, accepted, reason in SCRIPT:
        record_ba_outcome(stats, status, accepted=accepted, reason=reason)
    return stats


def test_identities_and_aliases_from_fresh_stats():
    s = _run({})
    assert s["ba_attempted"] == len(SCRIPT) == 10
    assert s["ba_skipped"] == 3
    assert s["ba_completed"] == 5
    assert s["ba_accepted"] == 3
    assert s["ba_rejected"] == 2
    assert s["ba_errors"] == 2
    assert s["ba_attempted"] == s["ba_skipped"] + s["ba_completed"] + s["ba_errors"]
    assert s["ba_completed"] == s["ba_accepted"] + s["ba_rejected"]
    assert s["ba_success"] == s["ba_accepted"] == 3
    assert s["ba_fail"] == s["ba_rejected"] + s["ba_errors"] == 4
    assert s["ba_skip_reasons"] == {"too_few_points": 2, "gauge_unanchored": 1}


def test_counters_accumulate_on_preinitialised_stats():
    s = {k: 0 for k in ("ba_attempted", "ba_skipped", "ba_completed", "ba_accepted",
                        "ba_rejected", "ba_errors", "ba_success", "ba_fail")}
    s["ba_skip_reasons"] = {}
    _run(s)
    _run(s)
    assert s["ba_attempted"] == 20 and s["ba_completed"] == 10 and s["ba_accepted"] == 6
    assert s["ba_success"] == 6 and s["ba_fail"] == 8


def test_identities_hold_after_every_step():
    s = {}
    for status, accepted, reason in SCRIPT:
        record_ba_outcome(s, status, accepted=accepted, reason=reason)
        assert s["ba_attempted"] == s["ba_skipped"] + s["ba_completed"] + s["ba_errors"]
        assert s["ba_completed"] == s["ba_accepted"] + s["ba_rejected"]
        assert s["ba_success"] == s["ba_accepted"]
        assert s["ba_fail"] == s["ba_rejected"] + s["ba_errors"]


def test_invalid_calls_raise_and_change_nothing():
    s = {}
    with pytest.raises(ValueError):
        record_ba_outcome(s, "completed")  # accepted required
    with pytest.raises(ValueError):
        record_ba_outcome(s, "accepted")   # not a status
    assert s.get("ba_attempted", 0) == 0


def test_summary_lines_have_the_verifier_formats():
    s = _run({})
    line1, line2 = ba_summary_lines(s)
    m1 = re.fullmatch(r"  Bundle adjustment: success=(\d+), fail=(\d+), success/overall=(\S+)", line1)
    assert m1 and m1.group(1) == "3" and m1.group(2) == "4" and m1.group(3) == "42.86%"
    m2 = re.fullmatch(
        r"  Bundle adjustment accounting: attempted=(\d+), skipped=(\d+), completed=(\d+), "
        r"accepted=(\d+), rejected=(\d+), errors=(\d+), skip_reasons=(\{.*\})", line2)
    assert m2 and list(map(int, m2.groups()[:6])) == [10, 3, 5, 3, 2, 2]
    assert eval(m2.group(7)) == {"too_few_points": 2, "gauge_unanchored": 1}  # noqa: S307
    assert "BA completed" not in line1 + line2
    # printed even when nothing ran
    z1, z2 = ba_summary_lines({})
    assert z1 == "  Bundle adjustment: success=0, fail=0, success/overall=0.00%"
    assert z2.endswith("attempted=0, skipped=0, completed=0, accepted=0, rejected=0, errors=0, skip_reasons={}")
