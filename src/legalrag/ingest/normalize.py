"""Arabic/English text helpers: display cleanup that keeps the wording, folding for search."""

from __future__ import annotations

import re

# bump whenever normalize_for_search changes; the API refuses an index built with another version
NORMALIZATION_VERSION = "v1"

# Arabic letters only: U+0621-U+064A minus tatweel (U+0640), which is decoration, not a letter
_ARABIC_LETTER = re.compile(r"[ء-ؿف-ي]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
# tashkeel: tanween, fatha..sukun, maddah/hamza marks (U+064B-U+0655) and dagger alef (U+0670)
_DIACRITICS = re.compile(r"[ً-ٰٕ]")
_TATWEEL = "ـ"
_ALEF_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# paragraph marker extracted with mirrored brackets: ")١ (", "(٢ (", " )٢ (", ")أ)".
# anchored to line start so prose like "(أ) و (ب)" is untouched
_MIRRORED_MARKER = re.compile(r"^\s*[()]\s*([٠-٩0-9]{1,3}|[ء-ي])\s*[()]\s*")
_SPACES = re.compile(r"\s+")


def has_arabic(text: str) -> bool:
    """True if the text contains at least one Arabic letter (digits and tatweel do not count)."""
    return bool(_ARABIC_LETTER.search(text))


def has_latin(text: str) -> bool:
    """True if the text contains at least one Latin letter."""
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
    """Join wrapped Arabic lines; a blank line or a paragraph marker starts a new paragraph."""
    paragraphs: list[list[str]] = [[]]
    for raw in lines:
        line = fix_mirrored_brackets(raw).strip()
        if not line:
            paragraphs.append([])
            continue
        if paragraphs[-1] and starts_with_paragraph_marker(raw):
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
