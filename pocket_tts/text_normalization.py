"""Text normalization registry for TTS preprocessing.

Each registered normalizer rewrites a specific surface pattern (decimals,
currency, ...) into a spoken form *before* tokenisation, preventing the
sentence splitter in :mod:`pocket_tts.models.tts_model` from breaking on
punctuation that is structural rather than prosodic.

The registry is ordered: normalizers run sequentially via
:func:`normalize_text`, and earlier entries see the original text while
later entries see the partially-rewritten output.  Order matters when one
pattern is a substring of another -- e.g. money (``$3.02``) must run
before plain decimals (``3.02``), otherwise ``$3.02`` would first be
rewritten to ``$3 point 02`` and then never recognised as currency.

Adding a new pattern is one entry in :data:`NORMALIZERS`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Normalizer:
    """A single text-normalization pattern.

    Attributes:
        name: Human-readable identifier (used in tests and debugging).
        pattern: Compiled regex matching the surface form to rewrite.
        handler: Callable receiving the regex ``Match`` and the language
            stem; returns the spoken-form replacement string.
    """

    name: str
    pattern: re.Pattern[str]
    handler: Callable[[re.Match[str], str], str]


# ---------------------------------------------------------------------------
# Decimals
# ---------------------------------------------------------------------------

# Spoken form of the decimal point for each supported language config stem.
# Languages not listed here fall back to ``"point"``.
DECIMAL_WORD: dict[str, str] = {
    "english": "point",
    "french": "virgule",
    "french_24l": "virgule",
    "german": "Komma",
    "german_24l": "Komma",
    "spanish": "coma",
    "spanish_24l": "coma",
    "portuguese": "vírgula",
    "portuguese_24l": "vírgula",
    "italian": "virgola",
    "italian_24l": "virgola",
}

_DECIMAL_RE = re.compile(r"(\d+)\.(\d+)")


def _decimal_handler(match: re.Match[str], language: str) -> str:
    word = DECIMAL_WORD.get(language, "point")
    return f"{match.group(1)} {word} {match.group(2)}"


# ---------------------------------------------------------------------------
# Currency / money
# ---------------------------------------------------------------------------

# Currency symbol -> (unit_singular, unit_plural, fraction_singular, fraction_plural).
# Wording is English: the symbols imply USD/EUR/GBP and inline use of these
# symbols in non-English text is rare enough that we don't translate.  Add
# language-specific overrides here if needed.
CURRENCY_WORDS: dict[str, tuple[str, str, str, str]] = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
}

# Match a currency symbol followed by either:
#   - integer-with-optional-thousands and optional .cents:  $3 / $1,234 / $3.02 / $1,234.56
#   - cents-only form with no integer part:                  $.50
_MONEY_RE = re.compile(
    r"([$€£])(?:(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?|\.(\d{2}))"
)


def _money_handler(match: re.Match[str], language: str) -> str:
    del language  # currency wording is symbol-driven, not language-driven
    symbol = match.group(1)
    int_str = match.group(2)
    cents_str = match.group(3) if match.group(3) is not None else match.group(4)

    unit_singular, unit_plural, frac_singular, frac_plural = CURRENCY_WORDS[symbol]

    units = int(int_str.replace(",", "")) if int_str is not None else 0
    fractional = int(cents_str) if cents_str is not None else None

    parts: list[str] = []
    if units > 0:
        word = unit_singular if units == 1 else unit_plural
        parts.append(f"{units} {word}")
    if fractional is not None and fractional > 0:
        if parts:
            parts.append("and")
        word = frac_singular if fractional == 1 else frac_plural
        parts.append(f"{fractional} {word}")
    if not parts:
        # $0 / $0.00 -- preserve "0 dollars" rather than emitting nothing
        parts.append(f"0 {unit_plural}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NORMALIZERS: tuple[Normalizer, ...] = (
    # Money runs first: ``$3.02`` contains a digit.digit pattern that the
    # decimal normalizer would otherwise rewrite to ``$3 point 02``.
    Normalizer(name="money", pattern=_MONEY_RE, handler=_money_handler),
    Normalizer(name="decimal", pattern=_DECIMAL_RE, handler=_decimal_handler),
)


def normalize_text(text: str, language: str = "english") -> str:
    """Apply every registered normalizer to ``text`` in registry order.

    Args:
        text: Input text to normalise.
        language: Language config stem (e.g. ``"english"``, ``"german"``).
            Controls the spoken form chosen by language-aware normalizers
            such as :data:`DECIMAL_WORD`.  Defaults to ``"english"``.

    Returns:
        Text with all matched patterns rewritten to spoken form.
    """
    for normalizer in NORMALIZERS:
        handler = normalizer.handler
        text = normalizer.pattern.sub(lambda m, _h=handler: _h(m, language), text)
    return text
