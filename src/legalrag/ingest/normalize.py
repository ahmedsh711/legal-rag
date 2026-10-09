"""Arabic/English text helpers used by the parser and by search.

Two kinds of cleaning, kept separate on purpose:
- *display* cleaning (``join_arabic_lines``, ``join_english_lines``, ``fix_mirrored_brackets``):
  repairs extraction damage but keeps the legal wording exactly as written.
- *search* normalization (``normalize_for_search``): folds spelling variants so that
  "فى" and "في", or "أحكام" and "احكام", match. Never shown to users.

Bump ``NORMALIZATION_VERSION`` whenever ``normalize_for_search`` changes: the index stores it
and the API refuses to start when the index was built with a different version.
"""

from __future__ import annotations

import re

NORMALIZATION_VERSION = "v1"

_ARABIC_LETTER = re.compile(r"[ء-ي]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_DIACRITICS = re.compile(r"[ً-ْٰ]")  # tashkeel incl. tanween, shadda, sukun, dagger alef
_TATWEEL = "ـ"
_ALEF_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# A paragraph marker at the start of a line, extracted with mirrored brackets:
# ")١ (", "(٢ (", " )٢ (", ")أ)". Anchored to line start so prose like "(أ) و (ب)" is untouched.
_MIRRORED_MARKER = re.compile(r"^\s*[()]\s*([٠-٩0-9]{1,3}|[ء-ي])\s*[()]\s*")
_SPACES = re.compile(r"\s+")


def has_arabic(text: str) -> bool:
    return bool(_ARABIC_LETTER.search(text))


def has_latin(text: str) -> bool:
    return bool(_LATIN_LETTER.search(text))


def arabic_digits_to_int(text: str) -> int:
    """'١٤٧' -> 147. Raises ValueError on empty or non-digit input."""
    digits = text.strip().translate(_DIGITS)
    if not digits.isdigit():
        raise ValueError(f"not a number: {text!r}")
    return int(digits)


def starts_with_paragraph_marker(line: str) -> bool:
    """True for lines opening a numbered paragraph: '(٢) ...', ')٢ (...', '(أ) ...'."""
    return bool(_MIRRORED_MARKER.match(line))


def fix_mirrored_brackets(line: str) -> str:
    """Repair a paragraph marker at the start of a line: ')١ (نص' -> '(١) نص'."""
    return _MIRRORED_MARKER.sub(lambda m: f"({m.group(1)}) ", line, count=1)


def normalize_for_search(text: str) -> str:
    """Fold Arabic spelling variants for keyword search (never for display)."""
    text = text.translate(_ALEF_FORMS).translate(_DIGITS)
    text = text.replace("ى", "ي").replace("ة", "ه").replace(_TATWEEL, "")
    text = _DIACRITICS.sub("", text)
    return _SPACES.sub(" ", text).strip()


def _collapse(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


def join_arabic_lines(lines: list[str]) -> str:
    """Join wrapped Arabic lines into paragraphs; a blank line or a paragraph marker starts a new one."""
    paragraphs: list[list[str]] = [[]]
    for raw in lines:
        line = fix_mirrored_brackets(raw).strip()
        if not line:
            paragraphs.append([])
            continue
        if line.startswith("(") and paragraphs[-1] and _MIRRORED_MARKER.match(raw or ""):
            paragraphs.append([])
        paragraphs[-1].append(line)
    return "\n".join(_collapse(" ".join(p)) for p in paragraphs if p)


def join_english_lines(lines: list[str]) -> str:
    """Join wrapped English lines; keep words hyphenated across a line break ('sub-' + 'let')."""
    paragraphs: list[str] = []
    current = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append(current)
            current = ""
            continue
        if current.endswith("-") and current[-2:-1].isalpha():
            current += line
        else:
            current = f"{current} {line}" if current else line
    if current:
        paragraphs.append(current)
    return "\n".join(_collapse(p) for p in paragraphs)
