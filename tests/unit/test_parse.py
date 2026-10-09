"""Parser behaviour on a synthetic page that reproduces every quirk found in the real PDF.

Quirks covered (see docs/research/03_corpus_inspection.md):
- a preamble with its own "مادة ١ / مادة ٢" before the code starts
- Arabic article numbers printed digit-reversed ("مادة٢١" is Article 12)
- Arabic and English headers on one line ("مادة٣١  Article 13")
- an article with Arabic text only (no English header)
- an English header with text on the same line ("Article 17 The ...")
- a cross-reference that starts a line ("Article 2 .") inside English text
- a repeal note in a heading block ("Articles 15-16 repealed")
- heading blocks in both languages that set book / chapter / section / topic
"""

import textwrap

import pytest

from legalrag.ingest.parse import Article, longest_increasing, parse_lines

SAMPLE = [
    (1, "القانون المدني المصري"),
    (1, "قانون الإصدار"),
    (1, "مادة ١"),
    (1, "يلغي القانون المدني المعمول به"),
    (1, "مادة ٢"),
    (1, "على وزير العدل تنفيذ هذا القانون"),
    (1, "نصوص القانون المدنى"),
    (1, "باب تمهيدي"),
    (1, "أحكام عامة"),
    (1, "الفصل الأول"),
    (1, "القانون وتطبيقه"),
    (1, "SECTION I"),
    (1, "Laws and their Applications"),
    (1, "١ -القانون والحق 1. Laws and Rights"),
    (1, "مادة١ ("),
    (1, ")١ (تسرى النصوص التشريعية على جميع المسائل."),
    (1, "(٢) فإذا لم يوجد نص تشريعي حكم القاضي بمقتضى العرف."),
    (1, "Article 1"),
    (1, "Provisions of laws govern all matters to which these"),
    (1, "provisions apply."),
    (1, "مادة٢ ("),
    (1, "لا يجوز إلغاء نص تشريعي إلا بتشريع لاحق."),
    (1, "Article 2"),
    (1, "A provision of a law can only be repealed by a"),
    (2, "subsequent law."),
    (2, "الكتاب الثاني"),
    (2, "العقود المسماة"),
    (2, "BOOK II"),
    (2, "SPECIFIC CONTRACTS"),
    (2, "الباب الأول"),
    (2, "العقود التي تقع على الملكية"),
    (2, "Chapter I"),
    (2, "Contracts as Regards Ownership"),
    (2, "مادة٢١"),
    (2, "البيع عقد يلتزم به البائع أن ينقل للمشتري ملكية شيء."),
    (2, "Article 12"),
    (2, "Sale is a contract whereby the vendor transfers ownership, see"),
    (2, "Article 2 ."),
    (2, "مادة٣١  Article 13"),
    (2, ")١ (يلتزم البائع بتسليم المبيع."),
    (2, "The vendor must deliver the thing sold."),
    (3, "مادة٤١"),
    (3, "نص عربي فقط بدون ترجمة."),
    (3, "المواد من٥١ إلى ٦١ ملغاة Articles 15-16 repealed"),
    (3, "الفصل الثاني"),
    (3, "المقايضة"),
    (3, "Section II"),
    (3, "Barter"),
    (3, "مادة٧١"),
    (3, "المقايضة عقد."),
    (3, "Article 17 Barter is a contract whereby"),
    (3, "each party transfers property."),
]


@pytest.fixture(scope="module")
def articles() -> dict[int, Article]:
    return {a.article_number: a for a in parse_lines(SAMPLE)}


def test_preamble_is_skipped_and_numbers_come_from_english(articles):
    assert sorted(articles) == [1, 2, 12, 13, 14, 15, 16, 17]


def test_arabic_and_english_text_are_split(articles):
    a1 = articles[1]
    assert a1.text_ar.startswith("(١) تسرى النصوص التشريعية")
    assert "(٢) فإذا لم يوجد" in a1.text_ar
    assert a1.text_en == "Provisions of laws govern all matters to which these provisions apply."
    assert a1.citation == "Egyptian Civil Code, Article 1"
    assert a1.source_page == 1


def test_english_text_continues_across_pages_and_stops_at_headings(articles):
    assert articles[2].text_en == "A provision of a law can only be repealed by a subsequent law."


def test_cross_reference_line_stays_in_the_text(articles):
    assert articles[12].text_en.endswith("see Article 2 .")


def test_combined_header_line(articles):
    a = articles[13]
    assert a.text_ar == "(١) يلتزم البائع بتسليم المبيع."
    assert a.text_en == "The vendor must deliver the thing sold."


def test_arabic_only_article_gets_reversed_number(articles):
    a = articles[14]
    assert a.text_ar == "نص عربي فقط بدون ترجمة."
    assert a.text_en == ""
    assert not a.is_repealed


def test_repeal_note_creates_flagged_records(articles):
    for n in (15, 16):
        assert articles[n].is_repealed
        assert articles[n].text_ar == ""
        assert "repealed" in articles[n].note.lower()


def test_english_header_with_inline_text(articles):
    assert articles[17].text_en == "Barter is a contract whereby each party transfers property."


def test_headings_become_metadata(articles):
    a1 = articles[1]
    assert a1.chapter == "باب تمهيدي: أحكام عامة"
    assert a1.section == "الفصل الأول: القانون وتطبيقه"
    assert a1.topic == "القانون والحق"
    assert "SECTION I" in a1.heading_en
    a12 = articles[12]
    assert a12.book == "الكتاب الثاني: العقود المسماة"
    assert a12.chapter == "الباب الأول: العقود التي تقع على الملكية"
    assert a12.section == ""  # a new part resets the lower levels
    assert a12.topic == ""
    a17 = articles[17]
    assert a17.book == "الكتاب الثاني: العقود المسماة"
    assert a17.section == "الفصل الثاني: المقايضة"
    assert "Barter" in a17.heading_en


def test_articles_are_sorted_and_unique():
    nums = [a.article_number for a in parse_lines(SAMPLE)]
    assert nums == sorted(set(nums))


def test_longest_increasing_drops_cross_references():
    # 444 after 450 is a cross-reference, 901 before 885 too
    seq = [449, 450, 444, 451, 884, 901, 885, 886]
    keep = longest_increasing(seq)
    assert [seq[i] for i in keep] == [449, 450, 451, 884, 885, 886]


# ---- patterns found when running on the real PDF (see reports/module-1.md) ----

SAMPLE2 = [
    (1, "نصوص القانون المدنى"),
    (1, "مادة٠٢١"),
    (1, ""),
    (1, "Article 120"),  # English header sits *inside* the Arabic text
    (1, "إذا وقع المتعاقد فى غلط جوهرى جاز له أن يطلب إبطال العقد."),
    (1, "A party to a contract may demand the avoidance of the contract."),
    (1, "مادة٨٢٤"),
    (1, "يلتزم البائع أن يقوم بما هو ضروري لنقل الحق المبيع إلى المشتري وأن"),
    (1, "Article 428"),
    (
        1,
        "يكف عن أي عمل من شأنه أن يجعل نقل الحق مستحيلا. The vendor is bound to perform everything",
    ),
    (1, "necessary to transfer the right."),
    (2, "مادة٢٥٤"),
    (2, "تسقط بالتقادم دعوى الضمان إذا انقضت سنة."),
    (2, "rticle 452"),  # header lost its first letter in extraction
    (2, "An action on a warranty is prescribed in one year."),
    (2, "٣ -الجمعيات Associations"),
    (2, "المواد من٤٥ إلى ٠٨"),
    (2, "ألغيت المواد من ٤٥ إلى ٠٨ بالقرار الجمهوري"),
    (2, "في .٤٦٩١/٣/٢١"),
    (2, "Article 454"),
    (2, "* Articles 454-456 have been repealed by Presidential"),
    (2, "Decree."),
    (2, "الفصل الثالث"),
    (2, "تقسيم الأشياء"),
    (2, "مادة٧٥٤"),
    (2, "نص المادة."),
    (2, "Article 457"),
    (2, "Text of the article."),
]


@pytest.fixture(scope="module")
def real_patterns() -> dict[int, Article]:
    return {a.article_number: a for a in parse_lines(SAMPLE2)}


def test_english_header_inside_arabic_text(real_patterns):
    a = real_patterns[120]
    assert a.text_ar == "إذا وقع المتعاقد فى غلط جوهرى جاز له أن يطلب إبطال العقد."
    assert a.text_en == "A party to a contract may demand the avoidance of the contract."


def test_line_with_arabic_then_english_is_split(real_patterns):
    a = real_patterns[428]
    assert a.text_ar.endswith("وأن يكف عن أي عمل من شأنه أن يجعل نقل الحق مستحيلا.")
    assert a.text_en == "The vendor is bound to perform everything necessary to transfer the right."


def test_header_missing_first_letter(real_patterns):
    assert real_patterns[452].text_en == "An action on a warranty is prescribed in one year."


def test_repeal_note_inside_english_text(real_patterns):
    for n in (454, 455, 456):
        assert real_patterns[n].is_repealed
    assert "Presidential" in real_patterns[455].note


def test_arabic_repeal_note_and_number_fragments_are_not_headings(real_patterns):
    a = real_patterns[454]
    assert a.topic == "الجمعيات"
    assert "ألغيت" not in a.topic and "٤٦٩١" not in a.heading_en
    assert real_patterns[457].section == "الفصل الثالث: تقسيم الأشياء"
    assert real_patterns[457].topic == ""


SAMPLE3 = [  # paragraph-by-paragraph alternation (Article 6 layout) followed by a real heading block
    (1, "نصوص القانون المدنى"),
    (1, "مادة٦("),
    (1, "(١ (النصوص المتعلقة بالأهلية تسري علي جميع الأشخاص الذين تنطبق عليهم"),
    (1, "Article 6"),
    (1, "Legislative provisions as regards the legal capacity of a person apply to all."),
    (2, "(٢ (وإذا عاد شخص توافرت فيه الأهلية بحسب نصوص قديمة ناقص الأهلية بحسب"),
    (2, "نصوص جديدة فإن ذلك لا يؤثر في تصرفاته السابقة."),
    (2, "When a person becomes legally incapable under a new law, prior acts stay valid."),
    (2, "تنازع القوانين من حيث المكان Conflicts of law as to place:"),
    (2, "مادة٠١("),
    (2, "القانون المصري هو المرجع في تكييف العلاقات."),
    (2, "Article 10"),
    (2, "Egyptian law governs the classification of relationships."),
]


def test_alternating_paragraphs_stay_in_their_article():
    arts = {a.article_number: a for a in parse_lines(SAMPLE3)}
    a6 = arts[6]
    assert a6.text_ar.count("\n") == 1 and "(٢) وإذا عاد شخص" in a6.text_ar
    assert a6.text_ar.endswith("فإن ذلك لا يؤثر في تصرفاته السابقة.")
    assert a6.text_en.endswith("prior acts stay valid.")
    assert "\n" in a6.text_en
    assert arts[10].topic == "تنازع القوانين من حيث المكان"
    assert arts[10].heading_en == "Conflicts of law as to place:"


def test_tanween_ending_does_not_break_the_split():
    sample = [
        (1, "نصوص القانون المدنى"),
        (1, "مادة٢٤("),
        (1, "(١ (موطن القاصر والمحجور عليه والمفقود والغائب هو موطن من ينوب"),
        (1, "Article 42"),
        (1, "The domicile of a minor is the domicile of his"),
        (1, "عن هؤلاء قانوناً ."),
        (1, "(٢ (ومع ذلك يكون للقاصر الذي بلغ ثماني عشرة سنة موطن خاص."),
        (1, "legal representative."),
        (1, "A minor of eighteen shall have a special domicile."),
    ]
    (a,) = parse_lines(sample)
    assert a.topic == ""
    assert a.text_ar.endswith("موطن خاص.")
    assert "عن هؤلاء قانوناً ." in a.text_ar
    assert a.text_en.endswith("special domicile.")


def test_cli_writes_json_and_lineage(tmp_path, monkeypatch):
    import json

    from legalrag.ingest import parse as parse_module

    monkeypatch.setattr(parse_module, "extract_lines", lambda _path: SAMPLE)
    pdf = tmp_path / "code.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    params = tmp_path / "params.yaml"
    params.write_text(
        textwrap.dedent(
            """\
            corpus:
              expected_articles: 17
              max_chars_ar: 500
              max_chars_en: 500
              length_ratio: [0.5, 3.0]
              known_anomalies:
                14: {reason: no English in source, flag: arabic_only, allow: [empty_en]}
            """
        ),
        encoding="utf-8",
    )
    out = tmp_path / "articles.json"
    parse_module.main(["--pdf", str(pdf), "--out", str(out), "--params", str(params)])
    records = json.loads(out.read_text(encoding="utf-8"))
    assert [r["article_number"] for r in records] == [1, 2, 12, 13, 14, 15, 16, 17]
    assert records[0]["id"] == "eg-civil-1"
    assert records[4]["quality_flags"] == ["arabic_only"]
    meta = json.loads((tmp_path / "articles.meta.json").read_text(encoding="utf-8"))
    assert meta["articles"] == 8 and meta["repealed"] == 2
    assert len(meta["source_pdf_md5"]) == 32 and meta["schema_version"]


def test_cli_fails_cleanly_on_missing_pdf(tmp_path):
    from legalrag.ingest import parse as parse_module

    with pytest.raises(SystemExit) as exc:
        parse_module.main(
            ["--pdf", str(tmp_path / "missing.pdf"), "--out", str(tmp_path / "o.json")]
        )
    assert exc.value.code == 1


def test_missing_start_marker_is_an_error():
    with pytest.raises(ValueError, match="start marker"):
        parse_lines([(1, "مادة١"), (1, "نص"), (1, "Article 1"), (1, "Text.")])


def test_duplicate_article_is_an_error():
    lines = [
        (1, "نصوص القانون المدنى"),
        (1, "مادة١"),
        (1, "نص"),
        (1, "Article 1"),
        (1, "Text."),
        (2, "مادة٢"),
        (2, "نص"),
        (2, "Article 2"),
        (2, "Text."),
        (3, "مادة٢"),
        (3, "نص مكرر"),
    ]  # an Arabic-only block whose number resolves to 2 again
    with pytest.raises(ValueError, match="without a number|duplicate"):
        parse_lines(lines)


def test_implausible_repeal_range_is_an_error():
    lines = [
        (1, "نصوص القانون المدنى"),
        (1, "مادة١"),
        (1, "نص"),
        (1, "Article 1"),
        (1, "* Articles 2-999999 have been repealed"),
    ]
    with pytest.raises(ValueError, match="implausible repeal range"):
        parse_lines(lines)


def test_repeal_note_is_not_left_in_english_text(real_patterns):
    assert real_patterns[454].text_en == ""
    assert real_patterns[452].text_en == "An action on a warranty is prescribed in one year."
    assert real_patterns[455].note == "Articles 454-456 have been repealed by Presidential Decree."
