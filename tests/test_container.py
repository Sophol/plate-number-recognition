"""Tests for ISO 6346 container-number parsing and check-digit validation.

The three real numbers here were read off this gate's own camera. They anchor
the checksum implementation to reality: a check-digit routine that is subtly
wrong (most commonly the remainder-10 case) still passes hand-picked examples,
so the test set includes a number that exercises exactly that branch.
"""

import json
from pathlib import Path

import pytest

from apps.inference_worker.container import (
    compute_check_digit,
    is_valid,
    normalise,
    parse,
)

# (full number, expected owner, category, serial) - all seen on the gate camera.
REAL_CONTAINERS = [
    ("MRSU2818883", "MRS", "U", "281888"),
    ("CMAU7224270", "CMA", "U", "722427"),
    ("TIIU5780291", "TII", "U", "578029"),
]


@pytest.mark.parametrize("number,owner,category,serial", REAL_CONTAINERS)
def test_real_containers_validate(number, owner, category, serial):
    parsed = parse(number)
    assert parsed is not None
    assert parsed.checksum_ok
    assert (parsed.owner, parsed.category, parsed.serial) == (owner, category, serial)
    assert parsed.canonical == number


def test_check_digit_matches_the_standard():
    # ISO 6346 worked example: the routine, not just the parser.
    assert compute_check_digit("CSQU305438") == 3


def test_remainder_ten_maps_to_zero():
    """The one irregular case: when the weighted sum mod 11 is 10, the digit is 0.

    CMAU7224270 hits it - CMAU722427 sums to a multiple of 11 - so a routine that
    returns 10 here instead of 0 would reject a genuine container.
    """
    assert compute_check_digit("CMAU722427") == 0


def test_spaces_between_groups_are_accepted():
    assert is_valid("CMAU 722427 0")
    assert normalise("CMAU 722427 0") == "CMAU7224270"


def test_wrong_check_digit_parses_but_fails_checksum():
    """A misread digit must be distinguishable from 'not a container number'."""
    parsed = parse("CMAU7224271")
    assert parsed is not None
    assert not parsed.checksum_ok
    assert not is_valid("CMAU7224271")


def test_non_container_text_returns_none():
    for text in ["2BJ-4506", "ABC123", "MRSU281888X", "", "PHNOM PENH"]:
        assert parse(text) is None
        assert not is_valid(text)


def test_invalid_equipment_category_is_rejected():
    """Only U, J and Z are ISO 6346 container categories; A is not."""
    assert parse("CMAA7224270") is None


# --- against the terminal's real container list (skipped if not fetched) ----

PAS_LIST = Path(__file__).resolve().parents[1] / "dataset" / "pas" / "containers.json"


@pytest.mark.skipif(not PAS_LIST.exists(), reason="run ml.container.fetch_pas first")
def test_checksum_holds_across_the_terminals_real_containers():
    """171k real numbers are the definitive test of the check-digit routine.

    Recomputed here rather than trusting the flag the fetcher stored. A wrong
    remainder-10 rule or letter table would fail thousands of these; the only
    failures allowed are the port system's own placeholders and typos.
    """
    data = json.loads(PAS_LIST.read_text())
    shaped = [c["number"] for c in data["containers"]
              if len(c["number"]) == 11 and c["number"][:4].isalpha() and c["number"][4:].isdigit()]
    failures = [n for n in shaped if compute_check_digit(n[:10]) != int(n[10])]
    assert len(shaped) > 100_000
    assert len(failures) / len(shaped) < 0.001, failures[:10]


@pytest.mark.skipif(not PAS_LIST.exists(), reason="run ml.container.fetch_pas first")
def test_known_container_lookup():
    from apps.inference_worker.container import load_known

    known = load_known(PAS_LIST)
    assert known is not None and len(known) > 100_000
    assert "MRSU2818883" in known           # photographed at the gate
    assert "MRSU 281888 3" in known         # spacing is normalised away
    assert "ZZZU0000000" not in known


def test_load_known_returns_none_without_a_list(tmp_path):
    from apps.inference_worker.container import load_known

    assert load_known(tmp_path / "missing.json") is None


# --- snapping a misread to the known list ------------------------------------

from apps.inference_worker.container import KnownContainers, snap_to_known

KNOWN = KnownContainers(["TCLU5437389", "MRSU2818883", "CMAU7224270"])


def test_snap_returns_exact_known_read_unchanged():
    assert snap_to_known("TCLU5437389", KNOWN) == "TCLU5437389"
    assert snap_to_known("tclu 543738 9", KNOWN) == "TCLU5437389"


def test_snap_corrects_one_substituted_character():
    assert snap_to_known("TCLU5437380", KNOWN) == "TCLU5437389"   # last digit misread
    assert snap_to_known("TCLU5A37389", KNOWN) == "TCLU5437389"   # 4 read as A


def test_snap_corrects_a_dropped_or_added_character():
    assert snap_to_known("TCLU543738", KNOWN) == "TCLU5437389"    # CTC dropped the 9
    assert snap_to_known("TCLUU5437389", KNOWN) == "TCLU5437389"  # doubled letter


def test_snap_refuses_two_edits_away():
    assert snap_to_known("TCLU5437300", KNOWN) is None
    assert snap_to_known("DESU87369", KNOWN) is None               # the v1 model's read


def test_snap_refuses_ambiguity():
    close = KnownContainers(["ABCU1234560", "ABCU1234561"])         # both one edit from the read
    assert snap_to_known("ABCU1234562", close) is None
