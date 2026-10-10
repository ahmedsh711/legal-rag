"""Parse the Egyptian Civil Code PDF into one ``Article`` record per article (DVC stage).

uv run python -m legalrag.ingest.parse --pdf data/raw/... --out data/processed/...
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from legalrag.ingest.normalize import (
    LATIN_WORD,
    NORMALIZATION_VERSION,
    arabic_digits_to_int,
    has_arabic,
    has_latin,
    join_arabic_lines,
    join_english_lines,
    starts_with_paragraph_marker,
)
from legalrag.ingest.params import CorpusInputError, CorpusParams, load_params
from legalrag.ingest.schema import SCHEMA_VERSION, Article
from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)

Line = tuple[int, str]  # (page number, text)

# ---------------------------------------------------------------- patterns and limits

# the code starts after the decree preamble
START_MARKER = re.compile(r"نصوص\s*القانون\s*المدن[يى]")
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

HEADING_MAX_CHARS = 50  # real headings are short; wrapped body lines are ~65-72 characters
MAX_REPEAL_SPAN = 100  # a note claiming more than this many repealed articles is a parsing bug
NO_UPPER_BOUND = sys.maxsize

_LEADING_NUMBER = re.compile(r"^[\s٠-٩0-9]+\s*[-–.)]\s*")
# last Arabic letter *or diacritic*: words often end in tanween ("قانوناً")
_LAST_ARABIC = re.compile(r"^(.*[ء-ْٰ])(.*)$")
_NUMBERED_HEADING = re.compile(r"^[٠-٩0-9]+\s*[-–]")  # "١ -القانون والحق"
_LEADING_PUNCT = re.compile(r"^[\s.,،؛:)]*")


# ---------------------------------------------------------------- small helpers


def extract_lines(pdf_path: str | Path) -> list[Line]:
    """Every text line of the PDF with its 1-based page number (pypdf keeps Arabic order intact)."""
    from pypdf import PdfReader  # heavy import kept local

    reader = PdfReader(str(pdf_path))
    return [
        (page_no, raw.rstrip())
        for page_no, page in enumerate(reader.pages, start=1)
        for raw in (page.extract_text() or "").split("\n")
    ]


def longest_increasing(values: list[int]) -> list[int]:
    """Indices of a longest strictly increasing subsequence (patience sorting, O(n log n)).

    Real headers increase through the document; a cross-reference that starts a line breaks
    the order and is dropped. Example: headers [449, 450, 444, 451] -> keep 449, 450, 451.
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
    # "مستحيلا. The vendor": the "." stays with the Arabic
    lead = _LEADING_PUNCT.match(en).group(0)
    return (ar + lead).strip(), en[len(lead) :].strip()


def _english_tail(line: str) -> tuple[str, str] | None:
    """If a line is Arabic text followed by an English sentence, return (arabic, english)."""
    ar, en = _split_mixed(line)
    if ar and len(LATIN_WORD.findall(en)) >= 3:
        return ar, en
    return None


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


# ---------------------------------------------------------------- headings


@dataclass
class Headings:
    """Current position in the code's hierarchy, in Arabic and in English."""

    ar: dict[str, str] = field(default_factory=lambda: dict.fromkeys(LEVEL_ORDER, ""))
    en: dict[str, str] = field(default_factory=lambda: dict.fromkeys(LEVEL_ORDER, ""))

    def snapshot(self) -> dict[str, str]:
        heading_en = " / ".join(v for v in (self.en[k] for k in LEVEL_ORDER) if v)
        return {**{k: self.ar[k] for k in LEVEL_ORDER}, "heading_en": heading_en}

    @staticmethod
    def _apply_side(
        side: dict[str, str], levels: Sequence[tuple[str, re.Pattern[str]]], lines: list[str]
    ) -> None:
        pending: str | None = None  # a keyword line ("الكتاب الثاني") waits for its title line
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
        self._apply_side(self.ar, AR_LEVELS, ar_lines)
        self._apply_side(self.en, EN_LEVELS, en_lines)


# ---------------------------------------------------------------- line classification


class Kind(Enum):
    ARABIC_HEADER = "arabic_header"
    COMBINED_HEADER = "combined_header"  # "مادة٣٨( Article 83"
    ENGLISH_HEADER = "english_header"
    TEXT = "text"


class Row(NamedTuple):
    page: int
    text: str  # for an English header: only the text after "Article N"
    kind: Kind
    number: int | None = None  # English article number (authoritative)
    ar_digits: str | None = None  # digits printed in the Arabic header (often reversed)


def _classify(lines: list[Line]) -> list[Row]:
    rows = []
    for page, text in lines:
        if m := COMBINED_HEADER.match(text):
            rows.append(Row(page, text, Kind.COMBINED_HEADER, int(m.group(2)), m.group(1)))
        elif m := AR_HEADER.match(text):
            rows.append(Row(page, text, Kind.ARABIC_HEADER, None, m.group(1)))
        elif m := EN_HEADER.match(text):
            rows.append(Row(page, m.group(2).strip(), Kind.ENGLISH_HEADER, int(m.group(1))))
        else:
            rows.append(Row(page, text, Kind.TEXT))
    # keep only English numbers that increase through the document; the rest are cross-references
    # (a forward reference that keeps the order increasing still slips through)
    numbered = [i for i, r in enumerate(rows) if r.number is not None]
    keep = {numbered[i] for i in longest_increasing([rows[i].number for i in numbered])}
    for i in numbered:
        if i not in keep:
            row = rows[i]
            if row.kind is Kind.COMBINED_HEADER:
                rows[i] = row._replace(kind=Kind.ARABIC_HEADER, number=None)
            else:
                rows[i] = Row(row.page, lines[i][1], Kind.TEXT)
    return rows


# ---------------------------------------------------------------- the parser


# An article runs from its Arabic header to the next one: Arabic header ("مادة٢١", multi-digit
# numbers digit-reversed), Arabic text, English header ("Article 12", reliable number), English
# text, then the headings of the next article.
class Mode(Enum):
    ARABIC = "arabic"
    ENGLISH = "english"
    ARABIC_AGAIN = "arabic_again"  # a later Arabic paragraph (AR1, EN1, AR2, EN2 layout)
    NOTE = "note"  # continuation lines of a repeal note found in English text
    HEADINGS = "headings"  # everything here belongs to the next article


@dataclass
class Repeal:
    first: int
    last: int
    note: str
    page: int
    headings: dict[str, str]

    def __post_init__(self) -> None:
        if self.first > self.last or self.last - self.first > MAX_REPEAL_SPAN:
            raise ValueError(f"implausible repeal range {self.first}-{self.last}: {self.note!r}")


@dataclass
class _Draft:
    page: int
    headings: dict[str, str]
    ar_digits: str | None = None
    number: int | None = None
    mode: Mode = Mode.ARABIC
    ar: list[str] = field(default_factory=list)
    en: list[str] = field(default_factory=list)
    tail: list[str] = field(default_factory=list)

    @property
    def english_started(self) -> bool:
        return any(t.strip() for t in self.en)

    def take_glued(self, text: str, new_paragraph: bool = False) -> bool:
        """Handle 'Arabic text. English text' on one line; True if the line was split."""
        split = _english_tail(text)
        if not split:
            return False
        self.ar.append(split[0])
        self.en.extend(["", split[1]] if new_paragraph else [split[1]])
        self.mode = Mode.ENGLISH
        return True


class _Parser:
    def __init__(self) -> None:
        self.headings = Headings()
        self.repeals: list[Repeal] = []
        self.drafts: list[_Draft] = []
        self.pending: list[str] = []  # heading lines waiting for the next article
        self.pending_page = 1
        self.handlers: dict[Mode, Callable[[_Draft, Row], None]] = {
            Mode.ARABIC: self._on_arabic,
            Mode.ENGLISH: self._on_english,
            Mode.ARABIC_AGAIN: self._on_arabic_again,
            Mode.NOTE: self._on_note,
            Mode.HEADINGS: lambda draft, row: draft.tail.append(row.text),
        }

    # -- article boundaries
    def _flush_headings(self, page: int) -> None:
        normal: list[str] = []
        for text in self.pending:
            if m := REPEAL_NOTE.search(text):
                self.headings.apply(normal)
                normal = []
                self._add_repeal(m, text, page)
            else:
                normal.append(text)
        self.headings.apply(normal)
        self.pending = []

    def _add_repeal(self, match: re.Match[str], text: str, page: int) -> None:
        first, last = int(match.group(1)), int(match.group(2))
        self.repeals.append(Repeal(first, last, text.strip(" *"), page, self.headings.snapshot()))

    def _start_draft(self, page: int) -> _Draft:
        if self.drafts:  # the previous article's tail holds headings for this one
            self.pending = self.drafts[-1].tail + self.pending
            self.drafts[-1].tail = []
        self._flush_headings(self.pending_page)
        draft = _Draft(page=page, headings=self.headings.snapshot())
        self.drafts.append(draft)
        return draft

    # -- per-mode handlers
    def _on_arabic(self, d: _Draft, row: Row) -> None:
        text = row.text
        if REPEAL_NOTE.search(text):
            d.mode = Mode.HEADINGS
            d.tail.append(text)
        elif d.take_glued(text):  # "... وأن يكف عن أي عمل. The vendor is bound ..."
            pass
        elif has_latin(text) and not has_arabic(text):
            d.mode = Mode.ENGLISH
            d.en.append(text)
        else:
            d.ar.append(text)

    def _on_english(self, d: _Draft, row: Row) -> None:
        text = row.text
        if has_arabic(text):
            if not d.english_started:  # Arabic continues after an early English header
                self._on_arabic(d, row)
            elif _looks_like_body(_split_mixed(text)[0]):  # paragraphs alternate AR/EN
                d.ar.append("")
                if not d.take_glued(text, new_paragraph=True):
                    d.ar.append(text)
                    d.mode = Mode.ARABIC_AGAIN
            else:
                d.mode = Mode.HEADINGS
                d.tail.append(text)
        elif m := REPEAL_NOTE.search(text):  # "* Articles 54-80 have been repealed ..."
            self._add_repeal(m, text, row.page)
            d.mode = Mode.NOTE
        else:
            d.en.append(text)

    def _on_arabic_again(self, d: _Draft, row: Row) -> None:
        text = row.text
        if has_latin(text) and not has_arabic(text):
            d.en.extend(["", text])
            d.mode = Mode.ENGLISH
        elif not d.take_glued(text, new_paragraph=True):
            d.ar.append(text)

    def _on_note(self, d: _Draft, row: Row) -> None:
        if has_arabic(row.text):
            d.mode = Mode.HEADINGS
            d.tail.append(row.text)
        elif has_latin(row.text):  # the note wraps: "... by Presidential" / "Decree."
            self.repeals[-1].note = f"{self.repeals[-1].note} {row.text.strip()}"

    # -- main loop
    def run(self, rows: list[Row]) -> tuple[list[_Draft], list[Repeal]]:
        current: _Draft | None = None
        for row in rows:
            if row.kind in (Kind.ARABIC_HEADER, Kind.COMBINED_HEADER):
                current = self._start_draft(row.page)
                current.ar_digits, current.number = row.ar_digits, row.number
            elif current is None:  # headings before the first article
                self.pending.append(row.text)
                self.pending_page = row.page
            elif row.kind is Kind.ENGLISH_HEADER:
                if current.number is not None or current.mode is not Mode.ARABIC:
                    current = self._start_draft(row.page)  # English header with no Arabic header
                current.number, current.mode = row.number, Mode.ENGLISH
                if row.text:
                    current.en.append(row.text)
            else:
                self.handlers[current.mode](current, row)
                self.pending_page = row.page
        return self.drafts, self.repeals


def _resolve_numbers(drafts: list[_Draft]) -> None:
    """Number Arabic-only drafts from their digit-reversed header, bounded by known neighbours."""
    for i, d in enumerate(drafts):
        if d.number is not None or not d.ar_digits:
            continue
        prev_n = next((x.number for x in reversed(drafts[:i]) if x.number is not None), 0)
        next_n = next((x.number for x in drafts[i + 1 :] if x.number is not None), NO_UPPER_BOUND)
        candidates = [
            n
            for n in (arabic_digits_to_int(d.ar_digits[::-1]), arabic_digits_to_int(d.ar_digits))
            if prev_n < n < next_n
        ]
        if len(set(candidates)) > 1:
            log.warning("ambiguous_arabic_number", digits=d.ar_digits, chose=candidates[0])
        if candidates:
            d.number = candidates[0]  # prefer the reversed reading, the usual extraction order


def _build_articles(drafts: list[_Draft], repeals: list[Repeal]) -> list[Article]:
    records: dict[int, Article] = {}
    for d in drafts:
        if d.number is None:
            raise ValueError(f"article text without a number on page {d.page}: {d.ar[:1]}")
        if d.number in records:
            raise ValueError(
                f"duplicate article {d.number} (pages {records[d.number].source_page}, {d.page})"
            )
        records[d.number] = Article(
            article_number=d.number,
            text_ar=join_arabic_lines(d.ar),
            text_en=join_english_lines(d.en),
            source_page=d.page,
            **d.headings,
        )
    for rp in repeals:
        for n in range(rp.first, rp.last + 1):
            if n in records:
                old = records[n]
                if old.text_ar or old.text_en:
                    log.warning("repeal_note_overrides_text", article=n)
                records[n] = old.model_copy(
                    update={"is_repealed": True, "note": rp.note, "text_ar": "", "text_en": ""}
                )
            else:
                records[n] = Article(
                    article_number=n,
                    is_repealed=True,
                    note=rp.note,
                    source_page=rp.page,
                    **rp.headings,
                )
    return [records[n] for n in sorted(records)]


def parse_lines(lines: list[Line]) -> list[Article]:
    """PDF lines -> sorted articles. Raises ValueError if this is not the expected PDF."""
    start = next((i for i, (_, t) in enumerate(lines) if START_MARKER.search(t)), None)
    if start is None:
        raise ValueError(
            "start marker 'نصوص القانون المدنى' not found: is this the Civil Code PDF?"
        )
    drafts, repeals = _Parser().run(_classify(lines[start + 1 :]))
    _resolve_numbers(drafts)
    return _build_articles(drafts, repeals)


def parse_pdf(pdf_path: str | Path) -> list[Article]:
    return parse_lines(extract_lines(pdf_path))


def apply_quality_flags(articles: list[Article], params: CorpusParams) -> list[Article]:
    """Copy documented source defects from params.yaml onto the records."""
    return [
        a.model_copy(update={"quality_flags": [params.known_anomalies[a.article_number].flag]})
        if a.article_number in params.known_anomalies
        else a
        for a in articles
    ]


# ---------------------------------------------------------------- CLI (DVC stage)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - content fingerprint, not security


def _write_atomic(path: Path, text: str) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")  # same bytes on every OS (DVC hash)
    tmp.replace(path)


def build_meta(pdf: Path, articles: list[Article]) -> dict[str, object]:
    """Lineage for articles.json: which source bytes and which code versions produced it."""
    import pypdf

    from legalrag import __version__

    return {
        "schema_version": SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "source_pdf_md5": _md5(pdf),
        "source_pdf_bytes": pdf.stat().st_size,
        "articles": len(articles),
        "repealed": sum(a.is_repealed for a in articles),
        "parser": {"legalrag": __version__, "pypdf": pypdf.__version__},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Parse the Egyptian Civil Code PDF into articles.json"
    )
    parser.add_argument("--pdf", default=None, help="source PDF (default: settings.raw_pdf_path)")
    parser.add_argument("--out", default=None, help="output JSON (default: settings.articles_path)")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args(argv)

    from legalrag.settings import get_settings  # after argparse so --help works without a valid env

    settings = get_settings()
    configure_logging(settings.log_level)
    pdf = Path(args.pdf or settings.raw_pdf_path)
    out = Path(args.out or settings.articles_path)
    try:
        if not pdf.is_file():
            raise CorpusInputError(f"PDF not found: {pdf} (run `dvc pull`)")
        params = load_params(args.params)
        articles = apply_quality_flags(parse_pdf(pdf), params)
        if not articles:
            raise CorpusInputError(f"no articles found in {pdf}")
    except (CorpusInputError, ValueError) as exc:
        log.error("parse_failed", detail=str(exc))
        sys.exit(1)

    _write_atomic(out, json.dumps([a.model_dump() for a in articles], ensure_ascii=False, indent=1))
    meta = build_meta(pdf, articles)
    _write_atomic(out.with_suffix(".meta.json"), json.dumps(meta, indent=2))
    log.info("parsed", out=str(out), **{k: v for k, v in meta.items() if k != "parser"})


if __name__ == "__main__":
    main()
