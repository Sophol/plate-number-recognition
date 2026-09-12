"""ISO 6346 shipping-container number recognition and validation.

A container number is four letters, six digits, and a check digit: owner code
(three letters), an equipment category (U for freight containers, J and Z for
related equipment), a six-digit serial, and one check digit computed from the
other ten characters. The check digit is what makes this worth more than plate
OCR: a read either satisfies the checksum or it does not, so most OCR mistakes
are caught rather than silently trusted.

The validator here is exact and needs no model. The detector and OCR that find
and read the marking are trained separately (ml/container), the same split as
plates: this module is the deterministic half that both the pipeline and the
tests rely on.

Verified against two containers seen on the gate camera: MRSU2818883 and
CMAU7224270 both satisfy the checksum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Letter values skip every multiple of 11 (11, 22, 33), per ISO 6346.
_LETTER_VALUES = {
    letter: value
    for letter, value in zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26,
         27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38],
    )
}

# Four letters, six digits, one check digit. Whitespace between groups is
# common in the painted form and is stripped before matching.
CONTAINER_RE = re.compile(r"^([A-Z]{3})([UJZ])(\d{6})(\d)$")

# Equipment category: only these three are ISO 6346 container categories.
VALID_CATEGORIES = frozenset("UJZ")


@dataclass(slots=True)
class ContainerNumber:
    owner: str          # three-letter owner prefix
    category: str       # U / J / Z
    serial: str         # six digits
    check_digit: int
    checksum_ok: bool

    @property
    def canonical(self) -> str:
        return f"{self.owner}{self.category}{self.serial}{self.check_digit}"


def compute_check_digit(owner_serial: str) -> int:
    """Check digit for the first ten characters (owner+category+serial).

    A remainder of 10 maps to 0: the standard's one irregular case, and the
    reason a naive "sum mod 11" implementation passes most numbers but rejects
    the occasional valid one.
    """
    total = sum(
        (_LETTER_VALUES[ch] if ch.isalpha() else int(ch)) * (2 ** i)
        for i, ch in enumerate(owner_serial)
    )
    remainder = total % 11
    return 0 if remainder == 10 else remainder


def normalise(text: str) -> str:
    """Strip spaces and hyphens, uppercase - the painted number carries spaces."""
    return re.sub(r"[\s\-]", "", text).upper()


def parse(text: str) -> ContainerNumber | None:
    """Parse an OCR string into a ContainerNumber, or None if it is not one.

    Returns the parse even when the checksum fails, with checksum_ok=False, so a
    caller can tell "not a container number at all" from "a container number the
    OCR misread". The gate wants that distinction; a failed checksum on an
    otherwise well-formed read is a signal to re-read, not to discard.
    """
    match = CONTAINER_RE.match(normalise(text))
    if not match:
        return None
    owner, category, serial, check = match.groups()
    body = owner + category + serial
    return ContainerNumber(
        owner=owner,
        category=category,
        serial=serial,
        check_digit=int(check),
        checksum_ok=compute_check_digit(body) == int(check),
    )


def is_valid(text: str) -> bool:
    """True only for a well-formed number whose check digit is correct."""
    parsed = parse(text)
    return parsed is not None and parsed.checksum_ok


# --- known-container lookup ------------------------------------------------

class KnownContainers:
    """Container numbers the terminal has actually handled, from the PAS list.

    A stronger check than the checksum. The check digit catches most misreads,
    but a misread that happens to land on *another* valid number sails through
    it; the odds of that number also being one the terminal has seen are tiny.
    So a read that is both checksum-valid and known is trustworthy enough to act
    on, and a checksum-valid read that is unknown is worth a second look.

    Only checksum-valid entries are loaded: the PAS column carries a handful of
    placeholders and typos that must not become a whitelist.
    """

    def __init__(self, numbers) -> None:
        self._numbers = frozenset(normalise(n) for n in numbers)

    @classmethod
    def from_pas_json(cls, path) -> "KnownContainers":
        import json
        from pathlib import Path

        data = json.loads(Path(path).read_text())
        return cls(c["number"] for c in data["containers"] if c.get("checksum_ok"))

    def __contains__(self, number: str) -> bool:
        return normalise(number) in self._numbers

    def __len__(self) -> int:
        return len(self._numbers)


def load_known(path) -> KnownContainers | None:
    """The known-container set, or None when the PAS list has not been fetched.

    None rather than an empty set, so callers can tell "no list available" from
    "list available and this number is not in it" -- the two mean different
    things at a gate.
    """
    from pathlib import Path

    return KnownContainers.from_pas_json(path) if Path(path).exists() else None


def snap_to_known(read: str, known: "KnownContainers") -> str | None:
    """Correct an OCR read to the one known container within a single edit.

    The list of numbers the terminal has handled works as a dictionary. An OCR
    that drops, adds or swaps one character produces a string that is not a
    container number, or is one that fails the checksum; if exactly one known
    container is one substitution, insertion or deletion away, that is almost
    certainly what was painted on the box.

    Ambiguity returns None rather than a guess. It is rare: changing one serial
    digit changes the check digit too, so two real containers are seldom within
    one edit of each other -- the checksum is what makes this snap safe.
    """
    read = normalise(read)
    if read in known:
        return read
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    candidates: set[str] = set()
    n = len(read)
    if n == 11:
        for i in range(11):
            for ch in alphabet:
                if ch != read[i]:
                    candidates.add(read[:i] + ch + read[i + 1:])
    elif n == 10:
        for i in range(11):
            for ch in alphabet:
                candidates.add(read[:i] + ch + read[i:])
    elif n == 12:
        for i in range(12):
            candidates.add(read[:i] + read[i + 1:])
    else:
        return None
    hits = [c for c in candidates if c in known]
    return hits[0] if len(hits) == 1 else None
