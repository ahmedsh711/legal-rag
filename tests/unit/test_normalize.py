"""Arabic text helpers: bracket repair, search normalization, digit conversion."""

import pytest

from legalrag.ingest.normalize import (
    NORMALIZATION_VERSION,
    arabic_digits_to_int,
    fix_mirrored_brackets,
    has_arabic,
    has_latin,
    join_arabic_lines,
    join_english_lines,
    normalize_for_search,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (")١ (تسرى النصوص", "(١) تسرى النصوص"),
        ("(٢ (وإذا عاد", "(٢) وإذا عاد"),
        (" )٢ (ومع ذلك", "(٢) ومع ذلك"),
        (")أ) إذا لم يقصد", "(أ) إذا لم يقصد"),
        ("(ب) إذا كانت", "(ب) إذا كانت"),  # already correct stays correct
    ],
)
def test_fix_mirrored_brackets(raw, expected):
    assert fix_mirrored_brackets(raw).strip() == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("أحكام", "احكام"),
        ("إلغاء", "الغاء"),
        ("آثار", "اثار"),
        ("فى", "في"),
        ("المادة", "الماده"),
        ("مسئولاً", "مسئولا"),  # tanween removed, hamza on yaa kept
        ("الـعقد", "العقد"),  # tatweel removed
        ("سنة ١٩٤٨", "سنه 1948"),
        ("  كثير   من  المسافات ", "كثير من المسافات"),
    ],
)
def test_normalize_for_search(raw, expected):
    assert normalize_for_search(raw) == expected


def test_normalization_is_idempotent():
    text = "العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله"
    once = normalize_for_search(text)
    assert normalize_for_search(once) == once


def test_normalization_has_a_version():
    assert NORMALIZATION_VERSION


@pytest.mark.parametrize(("raw", "expected"), [("٢", 2), ("١٤٧", 147), ("12", 12), ("٠١١", 11)])
def test_arabic_digits_to_int(raw, expected):
    assert arabic_digits_to_int(raw) == expected


def test_arabic_digits_to_int_rejects_empty():
    with pytest.raises(ValueError):
        arabic_digits_to_int("")


def test_script_detection():
    assert has_arabic("مادة ١") and not has_latin("مادة ١")
    assert has_latin("Article 1") and not has_arabic("Article 1")
    assert not has_arabic("(٢)") and not has_latin("(٢)")  # digits and brackets are neutral


def test_join_english_lines_keeps_hyphenated_words():
    lines = ["the lessee cannot assign the lease or sub-", "let the land", "", "New paragraph."]
    assert (
        join_english_lines(lines)
        == "the lessee cannot assign the lease or sub-let the land\nNew paragraph."
    )


def test_join_arabic_lines_keeps_paragraphs_and_fixes_brackets():
    lines = [")١ (تسرى النصوص التشريعية على جميع", "المسائل.", " ", "(٢) فإذا لم يوجد نص"]
    assert (
        join_arabic_lines(lines)
        == "(١) تسرى النصوص التشريعية على جميع المسائل.\n(٢) فإذا لم يوجد نص"
    )
