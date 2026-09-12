import re

# Per plate-type formats. Types marked TBD in the plan still need real samples
# collected before their rule can be written; they validate as unknown for now.
PLATE_PATTERNS: dict[str, re.Pattern[str]] = {
    "private_car": re.compile(r"^[1-9][A-Z]{1,2}-\d{4}$"),
    "motorcycle": re.compile(r"^[1-9][A-Z]{2}-\d{4}$"),
}

# Digit-only and letter-only positions differ per plate type, so corrections are
# applied by position rather than globally.
DIGIT_CONFUSIONS = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2"}
LETTER_CONFUSIONS = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z"}


def validate(plate_text: str, plate_type: str | None) -> bool:
    pattern = PLATE_PATTERNS.get(plate_type or "")
    if pattern is None:
        return False
    return bool(pattern.match(plate_text))


# A hyphen is far shorter than a character, so glyph segmentation discards it
# along with dirt and screw heads. Rather than loosen that filter, re-insert the
# separator at the position the plate type implies: the digits always occupy the
# last four characters.
# The digit block is matched loosely: OCR may still be reporting O/I/B/S/Z where
# digits belong, and those are only corrected once the hyphen tells `correct`
# which positions are numeric. Matching \d{4} here would deadlock the two steps.
_DIGITISH = "0-9" + "".join(DIGIT_CONFUSIONS)
_SEPARATORLESS = {
    "private_car": re.compile(rf"^([1-9][A-Z]{{1,2}})([{_DIGITISH}]{{4}})$"),
    "motorcycle": re.compile(rf"^([1-9][A-Z]{{2}})([{_DIGITISH}]{{4}})$"),
}


def restore_separator(plate_text: str, plate_type: str | None) -> str:
    """Re-insert the hyphen OCR dropped, when the text is otherwise well formed."""
    pattern = _SEPARATORLESS.get(plate_type or "")
    if pattern is None or "-" in plate_text:
        return plate_text
    match = pattern.match(plate_text)
    return f"{match.group(1)}-{match.group(2)}" if match else plate_text


def correct(plate_text: str, plate_type: str | None) -> str:
    """Fix common OCR confusions using the expected character class per position."""
    if plate_type not in PLATE_PATTERNS:
        return plate_text

    plate_text = restore_separator(plate_text, plate_type)

    parts = plate_text.split("-")
    if len(parts) != 2:
        return plate_text
    prefix, digits = parts

    # Prefix: leading province digit, then letters.
    fixed_prefix = ""
    for i, ch in enumerate(prefix):
        if i == 0:
            fixed_prefix += DIGIT_CONFUSIONS.get(ch, ch)
        else:
            fixed_prefix += LETTER_CONFUSIONS.get(ch, ch)

    fixed_digits = "".join(DIGIT_CONFUSIONS.get(ch, ch) for ch in digits)
    return f"{fixed_prefix}-{fixed_digits}"
