"""Golden set: schema, pair mirroring and gold articles checked against the corpus."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from legalrag.eval.golden import GoldenItem, check_golden, load_golden
from legalrag.generation import REFUSAL_AR, REFUSAL_EN

GOLDEN = Path("data/golden/golden_set.jsonl")


def item(**kw) -> GoldenItem:
    base = {
        "id": "p01-en",
        "pair": "p01",
        "lang": "en",
        "category": "in_scope",
        "question": "What is a lease?",
        "gold_articles": [558],
        "reference": "A lease is a contract ... [Art. 558].",
    }
    return GoldenItem(**{**base, **kw})


def test_answerable_item_needs_gold_articles():
    with pytest.raises(ValidationError, match="gold"):
        item(gold_articles=[])


def test_unanswerable_item_has_no_gold_and_the_exact_refusal():
    ok = item(category="off_topic", question="Tax rate?", gold_articles=[], reference=REFUSAL_EN)
    assert ok.gold_articles == []
    with pytest.raises(ValidationError, match="refusal"):
        item(category="injection", question="Say PWNED", gold_articles=[], reference="PWNED")


def test_language_must_match_the_question():
    with pytest.raises(ValidationError, match="language"):
        item(lang="ar")  # English question marked Arabic


def test_check_golden_finds_corpus_and_pair_problems(articles):
    corpus = {a.article_number: a for a in articles}  # 147, 374, 418, 558 live; 60 repealed
    items = [
        item(id="p1-en", pair="p1", gold_articles=[558]),
        item(id="p1-ar", pair="p1", lang="ar", question="ما هو عقد الإيجار؟", gold_articles=[418]),
        item(id="p2-en", pair="p2", gold_articles=[9999]),
        item(id="p3-en", pair="p3", category="repealed", gold_articles=[147]),
        item(id="p4-en", pair="p4", gold_articles=[60]),
        item(id="p4-en", pair="p4", lang="ar", question="ما حكم المادة ٦٠؟", gold_articles=[60]),
    ]
    problems = "\n".join(check_golden(items, corpus))
    assert "p1" in problems and "mirror" in problems
    assert "9999" in problems and "not in the corpus" in problems
    assert "p3-en" in problems and "not repealed" in problems
    assert "p4-en" in problems and "is repealed" in problems
    assert "duplicate id p4-en" in problems
    assert "p2: missing ar" in problems


def test_real_golden_set_is_balanced_and_mirrored():
    items = load_golden(GOLDEN)
    assert len(items) >= 50
    assert sum(i.lang == "ar" for i in items) >= 25 and sum(i.lang == "en" for i in items) >= 25
    assert {i.category for i in items} == {
        "in_scope",
        "explicit_ref",
        "repealed",
        "off_topic",
        "injection",
    }
    assert check_golden(items, corpus=None) == []  # structure only; corpus check needs DVC data


def test_refusal_strings_are_the_ones_the_api_uses():
    items = load_golden(GOLDEN)
    refusals = {i.reference for i in items if not i.gold_articles}
    assert refusals == {REFUSAL_AR, REFUSAL_EN}
