"""Tests for the text splitting logic in split_into_best_sentences."""

import pytest

from pocket_tts.conditioners.text import get_default_tokenizer
from pocket_tts.models.tts_model import split_into_best_sentences
from pocket_tts.text_normalization import (
    CURRENCY_WORDS,
    DECIMAL_WORD,
    NORMALIZERS,
    normalize_text,
)


def _normalize_decimals(text: str, language: str = "english") -> str:
    """Test helper: apply only the decimal normalizer."""
    decimal = next(n for n in NORMALIZERS if n.name == "decimal")
    return decimal.pattern.sub(lambda m: decimal.handler(m, language), text)


def _normalize_money(text: str, language: str = "english") -> str:
    """Test helper: apply only the money normalizer."""
    money = next(n for n in NORMALIZERS if n.name == "money")
    return money.pattern.sub(lambda m: money.handler(m, language), text)


_DECIMAL_WORD = DECIMAL_WORD


@pytest.fixture(scope="session")
def tokenizer():
    return get_default_tokenizer()


def test_short_text_single_chunk(tokenizer):
    """Short text should produce a single chunk."""
    chunks = split_into_best_sentences(
        tokenizer,
        "Hello world.",
        50,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
    )
    assert len(chunks) == 1


def test_multiple_sentences_split(tokenizer):
    """Multiple sentences should be split when they exceed max_tokens."""
    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
    chunks = split_into_best_sentences(
        tokenizer, text, 10, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    assert len(chunks) > 1


def test_long_sentence_with_commas_is_split(tokenizer):
    """A long sentence with only commas (no periods) should be split on commas."""
    # This is the core bug from issue #38 - the Tale of Two Cities example
    text = (
        "It was the best of times, it was the worst of times, "
        "it was the age of wisdom, it was the age of foolishness, "
        "it was the epoch of belief, it was the epoch of incredulity, "
        "it was the season of Light, it was the season of Darkness, "
        "it was the spring of hope, it was the winter of despair"
    )
    chunks = split_into_best_sentences(
        tokenizer, text, 50, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    assert len(chunks) > 1, "Long comma-separated text should be split into multiple chunks"

    # Verify all content is preserved (no words should be lost in splitting)
    rejoined = " ".join(chunks).lower()
    for phrase in ["best of times", "worst of times", "age of foolishness", "winter of despair"]:
        assert phrase in rejoined, f"'{phrase}' should be preserved after splitting"


def test_long_sentence_with_commas_respects_max_tokens(tokenizer):
    """Each chunk from comma splitting should respect max_tokens (when possible)."""
    text = (
        "It was the best of times, it was the worst of times, "
        "it was the age of wisdom, it was the age of foolishness, "
        "it was the epoch of belief, it was the epoch of incredulity"
    )
    max_tokens = 20
    chunks = split_into_best_sentences(
        tokenizer, text, max_tokens, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    for chunk in chunks:
        token_count = len(tokenizer(chunk.strip()).tokens[0].tolist())
        # Allow some tolerance since comma clauses may vary in size
        assert token_count <= max_tokens * 2, (
            f"Chunk '{chunk[:50]}...' has {token_count} tokens, expected ~{max_tokens}"
        )


def test_mixed_sentences_and_commas(tokenizer):
    """Text with both sentence boundaries and long comma-separated clauses."""
    text = (
        "Short sentence. "
        "This is a very long sentence with many clauses, separated by commas, "
        "that goes on and on, and on some more, without any periods at all, "
        "until it finally reaches a period. "
        "Another short one."
    )
    chunks = split_into_best_sentences(
        tokenizer, text, 20, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    assert len(chunks) >= 3


def test_no_commas_no_periods_stays_single_chunk(tokenizer):
    """Text with no splitting characters stays as a single chunk."""
    text = "one two three four five six seven eight nine ten eleven twelve"
    chunks = split_into_best_sentences(
        tokenizer, text, 5, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    # Should be 1 chunk since there are no split points
    assert len(chunks) == 1


def test_semicolons_and_colons_also_split(tokenizer):
    """Semicolons and colons should also serve as fallback split points."""
    text = (
        "First clause here; second clause here; third clause here; "
        "fourth clause here; fifth clause here; sixth clause here"
    )
    chunks = split_into_best_sentences(
        tokenizer, text, 15, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    assert len(chunks) > 1


def test_short_sentence_not_affected_by_comma_splitting(tokenizer):
    """Sentences under max_tokens should not be affected by comma logic."""
    text = "Hello, world."
    chunks = split_into_best_sentences(
        tokenizer, text, 50, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    assert len(chunks) == 1
    assert "hello" in chunks[0].lower()
    assert "world" in chunks[0].lower()


class TestNormalizeDecimals:
    """Unit tests for _normalize_decimals (issue #162)."""

    def test_simple_decimal(self):
        assert _normalize_decimals("98.6") == "98 point 6"

    def test_pi(self):
        assert _normalize_decimals("Pi is 3.14") == "Pi is 3 point 14"

    def test_multiple_decimals(self):
        assert _normalize_decimals("Pi is 3.14 and e is 2.718.") == (
            "Pi is 3 point 14 and e is 2 point 718."
        )

    def test_prose_period_untouched(self):
        """A period at end of a sentence (not between digits) must not be changed."""
        assert _normalize_decimals("Hello world.") == "Hello world."

    def test_no_decimals_unchanged(self):
        assert _normalize_decimals("No numbers here at all.") == "No numbers here at all."

    def test_integer_period_not_matched(self):
        """A period following digits but not followed by digits is untouched."""
        assert _normalize_decimals("See section 4. It is important.") == (
            "See section 4. It is important."
        )

    def test_german_uses_komma(self):
        assert _normalize_decimals("37.0°C", language="german") == "37 Komma 0°C"

    def test_french_uses_virgule(self):
        assert _normalize_decimals("3.14", language="french") == "3 virgule 14"

    def test_french_24l_uses_virgule(self):
        assert _normalize_decimals("3.14", language="french_24l") == "3 virgule 14"

    def test_spanish_uses_coma(self):
        assert _normalize_decimals("2.5", language="spanish") == "2 coma 5"

    def test_portuguese_uses_virgula(self):
        assert _normalize_decimals("1.5", language="portuguese") == "1 vírgula 5"

    def test_italian_uses_virgola(self):
        assert _normalize_decimals("9.8", language="italian") == "9 virgola 8"

    def test_unknown_language_falls_back_to_point(self):
        assert _normalize_decimals("3.14", language="klingon") == "3 point 14"

    def test_all_supported_languages_have_mapping(self):
        """Every language config that pocket-tts ships must have an entry in _DECIMAL_WORD."""
        expected = {
            "english",
            "french",
            "french_24l",
            "german",
            "german_24l",
            "spanish",
            "spanish_24l",
            "portuguese",
            "portuguese_24l",
            "italian",
            "italian_24l",
        }
        assert expected == set(_DECIMAL_WORD.keys())


def test_decimal_not_split_into_separate_chunks(tokenizer):
    """Decimals must not be treated as sentence boundaries (issue #162).

    '98.6°F' was previously split into ['98.', '6°F…'] because the period
    token matched the sentence-boundary set.  After normalisation the decimal
    is rewritten to '98 point 6' before tokenisation, so the chunk is kept
    whole.
    """
    text = "The average human body temperature is 98.6°F, which is normal."
    chunks = split_into_best_sentences(
        tokenizer,
        text,
        max_tokens=50,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
    )
    # The whole sentence fits in 50 tokens — must be a single chunk.
    assert len(chunks) == 1
    # The decimal value must survive intact (rewritten form expected).
    assert "point" in chunks[0].lower()


def test_multiple_decimals_preserved(tokenizer):
    """Multiple decimals in one sentence are all normalised correctly."""
    text = "Pi is 3.14 and e is 2.718."
    chunks = split_into_best_sentences(
        tokenizer,
        text,
        max_tokens=50,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
    )
    assert len(chunks) == 1
    rejoined = chunks[0].lower()
    assert "3 point 14" in rejoined or "point" in rejoined


class TestNormalizeMoney:
    """Unit tests for the money normalizer."""

    def test_dollars_and_cents(self):
        assert _normalize_money("$3.02") == "3 dollars and 2 cents"

    def test_singular_dollar_zero_cents(self):
        assert _normalize_money("$1.00") == "1 dollar"

    def test_singular_dollar_singular_cent(self):
        assert _normalize_money("$1.01") == "1 dollar and 1 cent"

    def test_cents_only(self):
        assert _normalize_money("$0.50") == "50 cents"

    def test_one_cent(self):
        assert _normalize_money("$0.01") == "1 cent"

    def test_dot_cents_form(self):
        assert _normalize_money("$.50") == "50 cents"

    def test_dollars_only_no_decimal(self):
        assert _normalize_money("$5") == "5 dollars"
        assert _normalize_money("$1") == "1 dollar"

    def test_zero_dollars(self):
        assert _normalize_money("$0") == "0 dollars"
        assert _normalize_money("$0.00") == "0 dollars"

    def test_thousands_separator(self):
        assert _normalize_money("$1,234.56") == "1234 dollars and 56 cents"
        assert _normalize_money("$1,000") == "1000 dollars"

    def test_inline_in_sentence(self):
        assert _normalize_money("It costs $3.02 today.") == (
            "It costs 3 dollars and 2 cents today."
        )

    def test_multiple_amounts(self):
        assert _normalize_money("Bought it for $5 and sold for $7.50.") == (
            "Bought it for 5 dollars and sold for 7 dollars and 50 cents."
        )

    def test_euro(self):
        assert _normalize_money("€3.02") == "3 euros and 2 cents"

    def test_pound_singular_uses_penny(self):
        assert _normalize_money("£0.01") == "1 penny"

    def test_pound_plural_uses_pence(self):
        assert _normalize_money("£3.02") == "3 pounds and 2 pence"

    def test_currency_words_dictionary_complete(self):
        """Every currency symbol in the regex must have a CURRENCY_WORDS entry."""
        # Each entry is (singular, plural, frac_singular, frac_plural).
        for symbol, words in CURRENCY_WORDS.items():
            assert len(words) == 4, f"{symbol} entry must be 4-tuple"
            assert all(isinstance(w, str) and w for w in words)


class TestNormalizeText:
    """Tests for the registry orchestrator."""

    def test_money_runs_before_decimals(self):
        """``$3.02`` must be rewritten as currency, not as ``$3 point 02``."""
        assert normalize_text("$3.02") == "3 dollars and 2 cents"

    def test_plain_decimal_still_works(self):
        assert normalize_text("Pi is 3.14") == "Pi is 3 point 14"

    def test_decimal_language_aware(self):
        assert normalize_text("Es ist 37.0°C", language="german") == (
            "Es ist 37 Komma 0°C"
        )

    def test_money_and_decimal_in_same_text(self):
        text = "It cost $3.02 and the temperature was 98.6°F."
        assert normalize_text(text) == (
            "It cost 3 dollars and 2 cents and the temperature was 98 point 6°F."
        )

    def test_no_match_returns_unchanged(self):
        assert normalize_text("Hello world.") == "Hello world."


def test_money_not_split_into_separate_chunks(tokenizer):
    """``$3.02`` must not be split on its decimal point either."""
    text = "It costs $3.02 today, which is reasonable."
    chunks = split_into_best_sentences(
        tokenizer,
        text,
        max_tokens=50,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
    )
    assert len(chunks) == 1
    rejoined = chunks[0].lower()
    assert "dollars" in rejoined and "cents" in rejoined
    # Must not have leaked through as a raw decimal rewrite.
    assert "$3 point 02" not in rejoined


def test_empty_string_raises(tokenizer):
    """Empty input should raise ValueError from prepare_text_prompt."""
    with pytest.raises(ValueError, match="empty"):
        split_into_best_sentences(
            tokenizer, "", 50, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
        )


def test_oversized_clause_without_commas_still_returns(tokenizer):
    """A long clause with no split points should still be returned (not dropped)."""
    # 20 words with no punctuation at all - no way to split
    text = " ".join(f"word{i}" for i in range(20))
    chunks = split_into_best_sentences(
        tokenizer, text, 5, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    assert len(chunks) == 1
    # prepare_text_prompt capitalizes the first char and adds a trailing period,
    # so compare case-insensitively and strip punctuation
    assert chunks[0].lower().rstrip(".") == text.lower()
