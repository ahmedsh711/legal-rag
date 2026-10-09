"""Egyptian Civil Code PDF -> one ``Article`` record per article.

How the PDF is laid out (found by inspection, see docs/research/03_corpus_inspection.md):

    [Arabic headings]   الكتاب الثاني / العقود المسماة
    [English headings]  BOOK II / SPECIFIC CONTRACTS
    مادة٢١              <- Arabic header; multi-digit numbers come out digit-reversed
    Arabic text ...
    Article 12          <- English header; numbers are reliable
    English text ...

So an article runs from its Arabic header to the next Arabic header and holds, in order:
Arabic text, English text, then the headings that belong to the *next* article.
Exceptions handled below: both headers on one line, Arabic-only articles, English headers with
text on the same line, cross-references that start a line ("Article 444 ."), and repeal notes.

Run as a DVC stage:  python -m legalrag.ingest.parse --pdf data/raw/... --out data/processed/...
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from legalrag.ingest.normalize import (
    arabic_digits_to_int,
    has_arabic,
    has_latin,
    join_arabic_lines,
    join_english_lines,
    starts_with_paragraph_marker,
)
from legalrag.ingest.schema import Article
from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)

Line = tuple[int, str]  # (page number, text)

START_MARKER = re.compile(
    r"نصوص\s*القانون\s*المدن[يى]"
)  # the code starts after the decree preamble
AR_HEADER = re.compile(r"^\s*م\s*ا\s*د\s*ة\s*[()\s]*([٠-٩0-9]+)?[()\s]*$")
COMBINED_HEADER = re.compile(r"^\s*م\s*ا\s*د\s*ة\s*[()\s]*([٠-٩0-9]+)?[()\s]*Article\s*(\d+)\s*$")
EN_HEADER = re.compile(r"^\s*A?rticle\s*(\d+)\b\s*(.*)$")  # one header lost its "A" ("rticle 452")
REPEAL_NOTE = re.compile(r"Articles?\s*(\d+)\s*[-–]\s*(\d+)\s*(?:have\s+been\s+)?repealed", re.I)
# Arabic repeal notes print their numbers digit-reversed, so they are only kept out of the headings
AR_REPEAL = re.compile(r"المواد\s*من|ألغيت|ألغي|ملغاة")

AR_LEVELS = [
    ("book", re.compile(r"^الكتاب\b")),
    ("chapter", re.compile(r"^(?:الباب|باب)\b")),
    ("section", re.compile(r"^الفصل\b")),
    ("subsection", re.compile(r"^الفرع\b")),
]
EN_LEVELS = [
    ("book", re.compile(r"^BOOK\b", re.I)),
    ("chapter", re.compile(r"^(?:CHAPTER|PART)\b", re.I)),
    ("section", re.compile(r"^SECTION\b", re.I)),
    ("subsection", re.compile(r"^SUB-?SECTION\b", re.I)),
]
LEVEL_ORDER = ["book", "chapter", "section", "subsection", "topic"]
_LEADING_NUMBER = re.compile(r"^[\s٠-٩0-9]+\s*[-–.)]\s*")
# last Arabic letter *or diacritic*: words often end in tanween ("قانوناً")
_LAST_ARABIC = re.compile(r"^(.*[ء-ْٰ])(.*)$")
_NUMBERED_HEADING = re.compile(r"^[٠-٩0-9]+\s*[-–]")  # "١ -القانون والحق"
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_LEADING_PUNCT = re.compile(r"^[\s.,،؛:)]*")


# ---------------------------------------------------------------- helpers


def extract_lines(pdf_path: str | Path) -> list[Line]:
    """Every text line of the PDF with its 1-based page number (pypdf keeps Arabic order intact)."""
    from pypdf import PdfReader  # heavy import kept local (course rule)

    reader = PdfReader(str(pdf_path))
    return [
        (page_no, raw.rstrip())
        for page_no, page in enumerate(reader.pages, start=1)
        for raw in (page.extract_text() or "").split("\n")
    ]


def longest_increasing(values: list[int]) -> list[int]:
    """Indices of a longest strictly increasing subsequence.

    Real article headers increase through the document; a cross-reference that happens to
    start a line ("Article 444 ." inside Article 450) breaks the order and is dropped.
    """
    tails: list[int] = []  # tails[k] = smallest last value of an increasing run of length k+1
    tail_idx: list[int] = []
    parent = [-1] * len(values)
    for i, v in enumerate(values):
        k = bisect.bisect_left(tails, v)
        if k == len(tails):
            tails.append(v)
            tail_idx.append(i)
        else:
            tails[k] = v
            tail_idx[k] = i
        parent[i] = tail_idx[k - 1] if k > 0 else -1
    out: list[int] = []
    i = tail_idx[-1] if tail_idx else -1
    while i != -1:
        out.append(i)
        i = parent[i]
    return out[::-1]


def _split_mixed(line: str) -> tuple[str, str]:
    """'١ -القانون والحق 1. Laws and Rights' -> ('١ -القانون والحق', '1. Laws and Rights')."""
    m = _LAST_ARABIC.match(line.strip())
    if not m:
        return "", line.strip()
    ar, en = m.group(1), m.group(2)
    lead = _LEADING_PUNCT.match(en).group(
        0
    )  # "مستحيلا. The vendor" -> keep the "." with the Arabic
    return (ar + lead).strip(), en[len(lead) :].strip()


HEADING_MAX_CHARS = 50  # real headings are short; wrapped body lines are ~65-72 characters


def _looks_like_body(arabic: str) -> bool:
    """Arabic that continues an article's text rather than a heading for the next article."""
    text = arabic.strip()
    if any(rx.match(text) for _, rx in AR_LEVELS) or _NUMBERED_HEADING.match(text):
        return False
    return (
        starts_with_paragraph_marker(text)
        or len(text) >= HEADING_MAX_CHARS
        or text.endswith((".", "،"))
    )


def _english_tail(line: str) -> tuple[str, str] | None:
    """If a line is Arabic text followed by an English sentence, return (arabic, english)."""
    ar, en = _split_mixed(line)
    if ar and len(_LATIN_WORD.findall(en)) >= 3:
        return ar, en
    return None


@dataclass
class Headings:
    """Current position in the code's hierarchy, in Arabic and in English."""

    ar: dict[str, str] = field(default_factory=lambda: dict.fromkeys(LEVEL_ORDER, ""))
    en: dict[str, str] = field(default_factory=lambda: dict.fromkeys(LEVEL_ORDER, ""))

    def snapshot(self) -> dict[str, str]:
        heading_en = " / ".join(v for v in (self.en[k] for k in LEVEL_ORDER) if v)
        return {**{k: self.ar[k] for k in LEVEL_ORDER}, "heading_en": heading_en}

    def _apply(self, side: dict[str, str], levels, lines: list[str]) -> None:
        pending: str | None = None
        topic: list[str] = []
        for text in lines:
            level = next((name for name, rx in levels if rx.match(text)), None)
            if level:
                side[level] = text
                for lower in LEVEL_ORDER[LEVEL_ORDER.index(level) + 1 :]:
                    side[lower] = ""
                pending, topic = level, []
            elif pending:
                side[pending] = f"{side[pending]}: {text}"
                pending = None
            else:
                topic.append(_LEADING_NUMBER.sub("", text))
        if topic:
            side["topic"] = " / ".join(topic)

    def apply(self, lines: list[str]) -> None:
        ar_lines, en_lines = [], []
        in_note = False  # inside an Arabic repeal note, which wraps over several lines
        for line in lines:
            ar, en = _split_mixed(line)
            if any(rx.match(ar) for _, rx in AR_LEVELS) or has_latin(en):
                in_note = False
            if AR_REPEAL.search(ar):
                in_note = True
            if ar and has_arabic(ar) and not in_note:
                ar_lines.append(ar)
            if en and has_latin(en):  # drop number-only fragments like ".٤٦٩١/٣/٢١"
                en_lines.append(en)
        self._apply(self.ar, AR_LEVELS, ar_lines)
        self._apply(self.en, EN_LEVELS, en_lines)


@dataclass
class _Draft:
    page: int
    headings: dict[str, str]
    ar_digits: str | None = None
    number: int | None = None  # from the English header (authoritative)
    mode: str = "ar"  # ar -> en -> tail
    ar: list[str] = field(default_factory=list)
    en: list[str] = field(default_factory=list)
    tail: list[str] = field(default_factory=list)  # headings for the next article


# ---------------------------------------------------------------- parser


def _classify(lines: list[Line]) -> list[tuple[int, str, str, int | None, str | None]]:
    """(page, text, kind, english_number, arabic_digits); kind in AR, COMBINED, EN, TEXT."""
    rows = []
    for page, text in lines:
        if m := COMBINED_HEADER.match(text):
            rows.append((page, text, "COMBINED", int(m.group(2)), m.group(1)))
        elif m := AR_HEADER.match(text):
            rows.append((page, text, "AR", None, m.group(1)))
        elif m := EN_HEADER.match(text):
            rows.append((page, m.group(2).strip(), "EN", int(m.group(1)), None))
        else:
            rows.append((page, text, "TEXT", None, None))
    # keep only English numbers that increase through the document; the rest are cross-references
    en_rows = [i for i, r in enumerate(rows) if r[3] is not None]
    keep = {en_rows[i] for i in longest_increasing([rows[i][3] for i in en_rows])}
    fixed = []
    for i, (page, text, kind, num, digits) in enumerate(rows):
        if num is not None and i not in keep:
            original = lines[i][1]
            kind, num = ("AR", None) if kind == "COMBINED" else ("TEXT", None)
            text = original if kind == "TEXT" else text
        fixed.append((page, text, kind, num, digits))
    return fixed


def _resolve_numbers(drafts: list[_Draft]) -> None:
    """Number Arabic-only drafts from their (digit-reversed) Arabic header, between known neighbours."""
    for i, d in enumerate(drafts):
        if d.number is not None or not d.ar_digits:
            continue
        prev_n = next((x.number for x in reversed(drafts[:i]) if x.number), 0)
        next_n = next((x.number for x in drafts[i + 1 :] if x.number), 10**6)
        reversed_n = arabic_digits_to_int(d.ar_digits[::-1])
        straight_n = arabic_digits_to_int(d.ar_digits)
        for candidate in (reversed_n, straight_n):
            if prev_n < candidate < next_n:
                d.number = candidate
                break


def parse_lines(lines: list[Line]) -> list[Article]:
    start = next((i for i, (_, t) in enumerate(lines) if START_MARKER.search(t)), -1)
    rows = _classify(lines[start + 1 :])

    headings = Headings()
    repeals: list[tuple[int, int, str, int, dict[str, str]]] = []
    drafts: list[_Draft] = []
    pending_head: list[str] = []  # heading lines waiting to be applied
    pending_page = 1

    def flush_headings(lines_: list[str], page: int) -> None:
        normal = []
        for text in lines_:
            if m := REPEAL_NOTE.search(text):
                headings.apply(normal)
                normal = []
                repeals.append(
                    (int(m.group(1)), int(m.group(2)), text.strip(), page, headings.snapshot())
                )
            else:
                normal.append(text)
        headings.apply(normal)

    def new_draft(page: int) -> _Draft:
        nonlocal pending_head
        if drafts:
            pending_head = drafts[-1].tail + pending_head
            drafts[-1].tail = []
        flush_headings(pending_head, pending_page)
        pending_head = []
        draft = _Draft(page=page, headings=headings.snapshot())
        drafts.append(draft)
        return draft

    cur: _Draft | None = None
    for page, text, kind, num, digits in rows:
        if kind in ("AR", "COMBINED"):
            cur = new_draft(page)
            cur.ar_digits, cur.number = digits, num
            continue
        if cur is None:  # headings before the first article
            pending_head.append(text)
            pending_page = page
            continue
        if kind == "EN":
            if cur.number is None and cur.mode == "ar":
                cur.number, cur.mode = num, "en"
            else:  # an English header with no Arabic header of its own
                cur = new_draft(page)
                cur.number, cur.mode = num, "en"
            if text:
                cur.en.append(text)
            continue
        english_started = any(t.strip() for t in cur.en)
        if cur.mode == "ar" or (cur.mode == "en" and not english_started and has_arabic(text)):
            # Arabic text. It can continue after an early English header (the "Article 120" layout).
            if cur.mode == "ar" and REPEAL_NOTE.search(text):
                cur.mode = "tail"
                cur.tail.append(text)
            elif split := _english_tail(text):  # "... وأن يكف عن أي عمل. The vendor is bound ..."
                cur.ar.append(split[0])
                cur.en.append(split[1])
                cur.mode = "en"
            elif has_latin(text) and not has_arabic(text):
                cur.mode = "en"
                cur.en.append(text)
            else:
                cur.ar.append(text)
        elif cur.mode == "ar2":  # a later Arabic paragraph of the same article
            if has_latin(text) and not has_arabic(text):
                cur.mode = "en"
                cur.en.append("")  # new English paragraph
                cur.en.append(text)
            elif split := _english_tail(text):
                cur.ar.append(split[0])
                cur.en.extend(["", split[1]])
                cur.mode = "en"
            else:
                cur.ar.append(text)
        elif cur.mode == "en":
            if has_arabic(text):
                arabic = _split_mixed(text)[0]
                if _looks_like_body(arabic):  # paragraphs alternate: AR (1), EN (1), AR (2), EN (2)
                    cur.mode = "ar2"
                    cur.ar.append("")  # new Arabic paragraph
                    if split := _english_tail(text):
                        cur.ar.append(split[0])
                        cur.en.extend(["", split[1]])
                        cur.mode = "en"
                    else:
                        cur.ar.append(text)
                else:
                    cur.mode = "tail"
                    cur.tail.append(text)
            else:
                if m := REPEAL_NOTE.search(text):  # "* Articles 54-80 have been repealed ..."
                    note = text.strip(" *")
                    repeals.append((int(m.group(1)), int(m.group(2)), note, page, cur.headings))
                cur.en.append(text)
        else:
            cur.tail.append(text)
        pending_page = page

    _resolve_numbers(drafts)
    records: dict[int, Article] = {}
    for d in drafts:
        if d.number is None:
            log.warning("article_without_number", page=d.page, text_ar=join_arabic_lines(d.ar)[:60])
            continue
        if d.number in records:
            log.warning("duplicate_article", article=d.number, page=d.page)
            continue
        records[d.number] = Article(
            article_number=d.number,
            text_ar=join_arabic_lines(d.ar),
            text_en=join_english_lines(d.en),
            source_page=d.page,
            **d.headings,
        )
    for first, last, note, page, snap in repeals:
        for n in range(first, last + 1):
            if n in records:
                records[n] = records[n].model_copy(update={"is_repealed": True, "note": note})
            else:
                records[n] = Article(
                    article_number=n, is_repealed=True, note=note, source_page=page, **snap
                )
    return [records[n] for n in sorted(records)]


def parse_pdf(pdf_path: str | Path) -> list[Article]:
    return parse_lines(extract_lines(pdf_path))


def main(argv: list[str] | None = None) -> None:
    from legalrag.settings import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Parse the Egyptian Civil Code PDF into articles.json"
    )
    parser.add_argument("--pdf", default=settings.raw_pdf_path)
    parser.add_argument("--out", default=settings.articles_path)
    args = parser.parse_args(argv)

    configure_logging(settings.log_level)
    articles = parse_pdf(args.pdf)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([a.model_dump() for a in articles], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    log.info(
        "parsed",
        articles=len(articles),
        repealed=sum(a.is_repealed for a in articles),
        arabic_only=sum(1 for a in articles if a.text_ar and not a.text_en),
        out=str(out),
    )


if __name__ == "__main__":
    main()
