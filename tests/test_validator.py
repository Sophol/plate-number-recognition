from apps.inference_worker.validator import correct, validate


def test_validates_private_car_plate():
    assert validate("2D-0888", "private_car")
    assert validate("1AB-1234", "private_car")


def test_rejects_malformed_plate():
    assert not validate("2D-088", "private_car")
    assert not validate("0D-0888", "private_car")
    assert not validate("2D0888", "private_car")


def test_unknown_plate_type_does_not_validate():
    assert not validate("2D-0888", None)
    assert not validate("2D-0888", "government")


def test_corrects_letters_in_digit_positions():
    assert correct("2D-O8BB", "private_car") == "2D-0888"


def test_corrects_digits_in_letter_positions():
    # Position 0 is the province digit; later prefix positions are letters.
    assert correct("21D-0888", "private_car") == "2ID-0888"
    # A letter already in a letter position is left alone.
    assert correct("2O-0888", "private_car") == "2O-0888"


def test_correction_is_noop_for_unknown_type():
    assert correct("2D-O888", None) == "2D-O888"


def test_restores_hyphen_dropped_by_ocr():
    """Glyph segmentation discards the hyphen, so `correct` puts it back."""
    assert correct("2D0888", "private_car") == "2D-0888"
    assert validate(correct("2D0888", "private_car"), "private_car")


def test_restores_hyphen_alongside_digit_confusions():
    """The separator regex must accept O/I/B/S/Z, or the two steps deadlock:
    no hyphen means `correct` bails, and no correction means no digits match."""
    assert correct("2DO888", "private_car") == "2D-0888"
    assert correct("2D08B8", "private_car") == "2D-0888"
    assert correct("2DOIBS", "private_car") == "2D-0185"


def test_restoring_the_hyphen_does_not_invent_plates():
    """Text that is not plate-shaped must stay invalid, signs included."""
    for text in ("CAFE24H", "STOPAHDD", "2D08881", "2D088", "0D0888"):
        assert not validate(correct(text, "private_car"), "private_car"), text


def test_existing_hyphen_is_left_alone():
    assert correct("2D-0888", "private_car") == "2D-0888"
