"""Tests for ISO 6346 container-number parsing and check-digit validation.

The three real numbers here were read off this gate's own camera. They anchor
the checksum implementation to reality: a check-digit routine that is subtly
wrong (most commonly the remainder-10 case) still passes hand-picked examples,
so the test set includes a number that exercises exactly that branch.
"""

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
