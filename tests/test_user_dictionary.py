"""Tests for the UserDictionary pronunciation-override layer."""

import json
from pathlib import Path

import pytest

from pocket_tts.text_normalization import DictionaryEntry, UserDictionary, normalize_text


class TestDictionaryEntry:
    def test_literal_match_word_boundary(self):
        entry = DictionaryEntry(match="API", replace="ay pee eye")
        pattern = entry.compile()
        assert pattern.sub(entry.replace, "The API is good") == "The ay pee eye is good"

    def test_literal_does_not_match_substring(self):
        """'API' must NOT match inside 'RAPID'."""
        entry = DictionaryEntry(match="API", replace="ay pee eye")
        pattern = entry.compile()
        assert pattern.sub(entry.replace, "RAPID growth") == "RAPID growth"

    def test_literal_case_sensitive_by_default(self):
        entry = DictionaryEntry(match="API", replace="ay pee eye")
        pattern = entry.compile()
        assert pattern.sub(entry.replace, "The api is good") == "The api is good"

    def test_literal_case_insensitive_flag(self):
        entry = DictionaryEntry(match="lol", replace="laugh out loud", case_insensitive=True)
        pattern = entry.compile()
        assert pattern.sub(entry.replace, "lol LOL Lol") == (
            "laugh out loud laugh out loud laugh out loud"
        )

    def test_literal_with_punctuation_escaped(self):
        """A literal containing regex metacharacters must be escaped."""
        entry = DictionaryEntry(match="Mr.", replace="Mister")
        pattern = entry.compile()
        # Without escaping, '.' would match any char; with escaping, only the literal
        # period is matched.  Word boundaries also keep this from matching 'Mrx'.
        assert pattern.sub(entry.replace, "Mr. Smith and Mrx") == "Mister Smith and Mrx"

    def test_regex_mode(self):
        entry = DictionaryEntry(match=r"Mr\.?\s+", replace="Mister ", regex=True)
        pattern = entry.compile()
        assert pattern.sub(entry.replace, "Mr Smith and Mr. Jones") == (
            "Mister Smith and Mister Jones"
        )

    def test_regex_with_backreferences(self):
        entry = DictionaryEntry(
            match=r"(\w+)@example\.com", replace=r"\1 at example dot com", regex=True
        )
        pattern = entry.compile()
        assert pattern.sub(entry.replace, "Email me at bob@example.com.") == (
            "Email me at bob at example dot com."
        )

    def test_regex_case_insensitive(self):
        entry = DictionaryEntry(
            match=r"http[s]?://\S+", replace="LINK", regex=True, case_insensitive=True
        )
        pattern = entry.compile()
        assert pattern.sub(entry.replace, "Visit HTTPS://x.com today") == ("Visit LINK today")


class TestUserDictionaryConstruction:
    def test_from_dict_basic(self):
        d = UserDictionary.from_dict({"english": [{"match": "API", "replace": "ay pee eye"}]})
        assert d.apply("The API is great.", "english") == ("The ay pee eye is great.")

    def test_from_dict_accepts_dictionary_entry_instances(self):
        d = UserDictionary.from_dict(
            {"english": [DictionaryEntry(match="API", replace="ay pee eye")]}
        )
        assert d.apply("The API.", "english") == "The ay pee eye."

    def test_from_dict_validates_required_keys(self):
        with pytest.raises(ValueError, match="'match' and 'replace'"):
            UserDictionary.from_dict({"english": [{"match": "API"}]})

    def test_invalid_regex_rejected_at_construction(self):
        with pytest.raises(ValueError, match="Invalid regex"):
            UserDictionary.from_dict(
                {"english": [{"match": "(unclosed", "replace": "x", "regex": True}]}
            )

    def test_empty_dictionary_passes_through(self):
        d = UserDictionary()
        assert d.apply("Hello world.", "english") == "Hello world."

    def test_unknown_language_returns_unchanged(self):
        d = UserDictionary.from_dict({"english": [{"match": "API", "replace": "ay pee eye"}]})
        # Apply with a language that has no section -> no rewrites happen.
        assert d.apply("The API.", "klingon") == "The API."


class TestUserDictionaryFromFile:
    def test_load_json(self, tmp_path: Path):
        path = tmp_path / "dict.json"
        path.write_text(
            json.dumps(
                {
                    "english": [
                        {"match": "API", "replace": "ay pee eye"},
                        {"match": "lol", "replace": "laugh out loud", "case_insensitive": True},
                    ]
                }
            ),
            encoding="utf-8",
        )
        d = UserDictionary.from_file(path)
        assert d.apply("API LOL", "english") == "ay pee eye laugh out loud"

    def test_load_yaml_when_pyyaml_available(self, tmp_path: Path):
        pytest.importorskip("yaml")
        path = tmp_path / "dict.yaml"
        path.write_text(
            "english:\n"
            "  - match: API\n"
            "    replace: ay pee eye\n"
            "common:\n"
            "  - match: '&'\n"
            "    replace: ' and '\n",
            encoding="utf-8",
        )
        d = UserDictionary.from_file(path)
        assert d.apply("API & friends", "english") == "ay pee eye  and  friends"

    def test_unsupported_extension_rejected(self, tmp_path: Path):
        path = tmp_path / "dict.txt"
        path.write_text("english: []", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported"):
            UserDictionary.from_file(path)

    def test_top_level_must_be_mapping(self, tmp_path: Path):
        path = tmp_path / "dict.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="top-level mapping"):
            UserDictionary.from_file(path)


class TestUserDictionaryApply:
    def test_language_section_runs_first(self):
        d = UserDictionary.from_dict(
            {
                "english": [{"match": "color", "replace": "colour"}],
                "german": [{"match": "API", "replace": "ah pe ih"}],
            }
        )
        assert d.apply("color", "english") == "colour"
        assert d.apply("color", "german") == "color"
        assert d.apply("API", "german") == "ah pe ih"

    def test_common_section_applies_to_all(self):
        d = UserDictionary.from_dict(
            {
                "english": [{"match": "API", "replace": "ay pee eye"}],
                "common": [{"match": "&", "replace": " and "}],
            }
        )
        # 'common' fires for english AND for any language without its own section.
        assert d.apply("Tom & Jerry", "english") == "Tom  and  Jerry"
        assert d.apply("Tom & Jerry", "klingon") == "Tom  and  Jerry"

    def test_language_runs_before_common(self):
        """Language entries fire first; common can act on the rewritten text."""
        d = UserDictionary.from_dict(
            {
                "english": [{"match": "Bob", "replace": "Robert"}],
                "common": [{"match": "Robert", "replace": "Bobby"}],
            }
        )
        # English rewrites Bob -> Robert; common then rewrites Robert -> Bobby.
        assert d.apply("Hi Bob.", "english") == "Hi Bobby."

    def test_multiple_entries_in_order(self):
        d = UserDictionary.from_dict(
            {"english": [{"match": "A", "replace": "B"}, {"match": "B", "replace": "C"}]}
        )
        # First A->B (turning every A into B), then B->C (turning every B into C).
        assert d.apply("A B", "english") == "C C"


class TestUserDictionaryMerge:
    def test_merge_appends_in_other(self):
        a = UserDictionary.from_dict({"english": [{"match": "API", "replace": "ay pee eye"}]})
        b = UserDictionary.from_dict({"english": [{"match": "lol", "replace": "laugh out loud"}]})
        merged = a.merge(b)
        assert merged.apply("API lol", "english") == "ay pee eye laugh out loud"

    def test_merge_other_runs_after_self(self):
        """Merged dict applies self's entries first, then other's."""
        a = UserDictionary.from_dict({"english": [{"match": "Bob", "replace": "Robert"}]})
        b = UserDictionary.from_dict({"english": [{"match": "Robert", "replace": "Bobby"}]})
        merged = a.merge(b)
        assert merged.apply("Hi Bob.", "english") == "Hi Bobby."

    def test_merge_does_not_mutate_inputs(self):
        a = UserDictionary.from_dict({"english": [{"match": "API", "replace": "ay pee eye"}]})
        b = UserDictionary.from_dict({"english": [{"match": "lol", "replace": "laugh out loud"}]})
        a.merge(b)
        # Original 'a' must still only know about API.
        assert a.apply("API lol", "english") == "ay pee eye lol"

    def test_merge_adds_new_section(self):
        a = UserDictionary.from_dict({"english": [{"match": "API", "replace": "ay pee eye"}]})
        b = UserDictionary.from_dict({"common": [{"match": "&", "replace": " and "}]})
        merged = a.merge(b)
        assert merged.apply("API & lol", "english") == "ay pee eye  and  lol"


class TestNormalizeTextWithDictionary:
    def test_dictionary_runs_after_builtin_normalizers(self):
        """Builtin money/decimal rewrites happen first, then user overrides."""
        d = UserDictionary.from_dict({"english": [{"match": "dollars", "replace": "bucks"}]})
        # Money normalizer turns $3.02 -> "3 dollars and 2 cents".
        # User dictionary then rewrites 'dollars' -> 'bucks'.
        assert normalize_text("It costs $3.02.", dictionary=d) == ("It costs 3 bucks and 2 cents.")

    def test_dictionary_none_is_default(self):
        """No dictionary argument means only built-ins run."""
        assert normalize_text("It costs $3.02.") == "It costs 3 dollars and 2 cents."

    def test_advisorium_persona_pattern(self):
        """Per-advisor dict layered on a global dict (the Advisorium use case)."""
        global_dict = UserDictionary.from_dict(
            {
                "english": [{"match": "API", "replace": "ay pee eye"}],
                "common": [{"match": "&", "replace": " and "}],
            }
        )
        # An advisor adds their own pronunciation for their name.
        advisor_dict = UserDictionary.from_dict(
            {"english": [{"match": "Hormozi", "replace": "hor moh zee"}]}
        )
        composed = global_dict.merge(advisor_dict)
        text = "Hormozi & the API guy."
        assert normalize_text(text, dictionary=composed) == (
            "hor moh zee  and  the ay pee eye guy."
        )
