from scripts.transcribe_and_score import normalize_text, word_error_rate


def test_normalize_text_is_case_and_accent_insensitive():
    assert normalize_text("Esta, PRUÉBA!") == "esta prueba"


def test_word_error_rate_reports_substitution():
    errors, words, wer = word_error_rate("Esta prueba", "Es prueba")

    assert (errors, words, wer) == (1, 2, 0.5)
