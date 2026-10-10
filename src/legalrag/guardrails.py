"""Input guardrails, run on every question before retrieval, the decider or the LLM see it.

- PII (Egyptian national ID, mobile number, email) is replaced by a typed placeholder, so it never
  reaches an external API (Gemini, OpenRouter/Jev) or a log line. The question is still answered.
- Prompt injection (override / prompt leak / role play / chat-template delimiters, Arabic and
  English) is blocked: the user gets the normal refusal and no tokens are spent.

Every guard that fires is reported by name ("pii:phone", "injection:override"), so the API can
count them and the evaluation can measure detection rate and false-positive rate per guard.

These are regular expressions, chosen on purpose: microseconds per question, no model, no
download, and their mistakes are readable. They miss paraphrased or obfuscated attacks
(measured in reports/module-4.md); prompt rule 7 and the citation check stay behind them.
Presidio was rejected: its value is NER for names/addresses, it has no Arabic model, and the
three PII types that matter here are fixed-format numbers that a pattern finds exactly.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass

from legalrag.ingest.normalize import normalize_for_search

_DIGITS = str.maketrans(  # Arabic-Indic, Persian and fullwidth digits -> 0-9, one char for one
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹０１２３４５６７８９", "012345678901234567890123456789"
)
# zero-width and direction marks: invisible, meaningless here, and a classic way to split a word
# ("ig​nore") so a pattern no longer sees it
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿]")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'"})

# national ID: century (2=1900s, 3=2000s) + YYMMDD + governorate(2) + serial(4) + check digit
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "national_id": re.compile(
        r"(?<!\d)[23]\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{7}(?!\d)"
    ),
    "phone": re.compile(r"(?<![\d+])(?:\+20|0020|20|0)[\s-]?1[0125](?:[\s-]?\d){8}(?!\d)"),
    # bounded parts (RFC limits): no slow backtracking on a long run of letters without "@"
    "email": re.compile(r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){1,8}"),
}

# Matched against _fold(text): Arabic spelling folded (أ→ا, ة→ه, ى→ي, no diacritics), lower case,
# so the Arabic patterns are written in that folded form.
# Each pattern needs a sign that the target is the assistant ("your rules", "previous
# instructions", "you are now an AI"): legal questions say "ignore the clause" or "you are now
# the tenant", and those must pass (tested in tests/unit/test_guardrails.py).
_EN_NOUN = r"(?:instructions?|rules|prompts?|directions|guidelines|directives)\b"
_EN_EARLIER = r"(?:previous|prior|above|earlier|preceding)"
_EN_AI = r"(?:assistant|ai|bot|chatbot|model|poet|unrestricted|unfiltered|jailbroken)\b"
_AR_NOUN = r"(?:ال)?(?:تعليمات|قواعد|اوامر|توجيهات|ارشادات)"
_AR_AI = r"(?:مساعد|نموذج|بوت|ذكاء|شات|شاعر)"
_AND = r"\b[وف]?"  # Arabic glues "and"/"so" (و/ف) to the next word: وتجاهل = "and ignore"
_AR_FORGET = _AND + r"(?:تجاهل|انس[يا]?|اهمل|تخط[يا]?|اترك)\s+"
INJECTION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "override": [
        re.compile(
            r"\b(?:ignore|disregard|forget|skip|bypass)\s+"
            rf"(?:all\s+(?:of\s+)?(?:the\s+|your\s+)?(?:{_EN_EARLIER}\s+)?"  # all (the previous)
            r"|(?:your|my)\s+(?:\w+\s+)?"  # your (previous)
            rf"|(?:the|these|those)\s+(?:{_EN_EARLIER}|original|system)\s+"  # the above
            rf"|{_EN_EARLIER}\s+)" + _EN_NOUN
        ),
        re.compile(r"\bnew\s+instructions\s*:"),
        re.compile(
            r"\b(?:do\s+not|don't)\s+(?:follow|obey)\s+(?:your|these|my)\s+(?:rules|instructions)\b"
        ),
        re.compile(
            r"\boverride\s+(?:your|all)\s+(?:previous\s+)?(?:rules|instructions|guidelines)\b"
        ),
        # Arabic: "ignore ALL the instructions", "ignore your instructions", "... the previous ones"
        re.compile(_AR_FORGET + r"(?:كل|جميع)\s+" + _AR_NOUN),
        re.compile(_AR_FORGET + _AR_NOUN + r"(?:ك\b|\s+(?:السابقه|الماضيه|اعلاه|المعطاه))"),
        re.compile(_AND + r"سيبك\s+من\s+(?:كل\s+)?(?:ال)?(?:تعليمات|قواعد|اوامر|كلام)"),
        re.compile(_AND + r"لا\s+(?:تتبع|تطيع|تلتزم\s+ب)\s*" + _AR_NOUN),
        re.compile(r"\bتعليمات\s+جديده\s*:"),
    ],
    "prompt_leak": [
        re.compile(r"\bsystem\s+prompt\b"),
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|display|tell\s+me|write\s+out|what\s+(?:are|is|were))"
            r"\s+(?:me\s+)?(?:your\s+(?:(?:system|initial|original|hidden|secret)\s+)?"
            r"|the\s+(?:system|hidden|secret)\s+)(?:prompt|instructions|rules)\b"
        ),
        re.compile(r"\b(?:instructions|rules|prompt)\s+you\s+(?:were|have\s+been)\s+given\b"),
        re.compile(r"\b(?:موجه|رساله|تعليمات)\s+النظام\b"),
        re.compile(r"\b(?:ال)?برومبت\b"),
        re.compile(
            _AND + r"(?:اعرض|اكتب|اطبع|اظهر|كرر|ما\s+هي|ايه)\s+(?:لي\s+)?(?:تعليماتك|قواعدك|اوامرك)"
        ),
        re.compile(r"\b(?:تعليماتك|قواعدك)\s+(?:الاصليه|السريه|المخفيه)"),
    ],
    "role_play": [
        re.compile(
            r"\byou\s+are\s+(?:now|no\s+longer)\s+"
            rf"(?:dan\b|(?:an?\s+|my\s+)?(?:\w+\s+){{0,2}}?{_EN_AI})"
        ),
        re.compile(
            r"\bfrom\s+now\s+on,?\s+(?:answer\s+(?:any|every|all)\b"
            rf"|you\s+(?:are|will\s+be|act\s+as)\s+(?:an?\s+)?(?:\w+\s+){{0,2}}?{_EN_AI})"
        ),
        re.compile(r"\bpretend\s+(?:that\s+)?you(?:'re|\s+are)\b"),
        re.compile(r"\b(?:developer|jailbreak|god|dan)\s+mode\b"),
        re.compile(_AND + r"انت\s+(?:الان|لم\s+تعد)\s+(?:\S+\s+){0,2}?" + _AR_AI),
        re.compile(
            _AND + r"من\s+الان\s+(?:فصاعدا\s+)?(?:انت|ستكون|ستصبح|تصرف)\s+(?:\S+\s+){0,3}?" + _AR_AI
        ),
        re.compile(_AND + r"تظاهر\s+(?:بانك|انك)"),
        re.compile(
            _AND + r"(?:العب|خذ)\s+دور\b"
        ),  # not مثل: "مثل دور" also means "such as the role"
    ],
    "delimiter": [  # pieces of a chat template or of our own prompt, typed by the user
        re.compile(r"</?\s*(?:system|articles|assistant|user|instructions?)\s*>"),
        re.compile(r"\[/?inst\]"),
        re.compile(r"<\|im_(?:start|end)\|>"),
    ],
}


@dataclass(frozen=True)
class QuestionCheck:
    text: str  # the question with PII replaced; what the rest of the pipeline sees
    fired: list[str]  # e.g. ["pii:phone", "injection:override"]
    blocked: bool
    latency_ms: float


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII with [NATIONAL_ID] / [PHONE] / [EMAIL]; return the new text and the types found."""
    text = _INVISIBLE.sub("", text)
    folded = text.translate(_DIGITS)  # one char for one char: positions stay valid in `text`
    spans: list[tuple[int, int, str]] = []
    for label, pattern in PII_PATTERNS.items():
        for m in pattern.finditer(folded):
            if not any(m.start() < end and start < m.end() for start, end, _ in spans):
                spans.append((m.start(), m.end(), label))
    for start, end, label in sorted(spans, reverse=True):
        text = f"{text[:start]}[{label.upper()}]{text[end:]}"
    found = {label for _, _, label in spans}
    return text, [label for label in PII_PATTERNS if label in found]


def _fold(text: str) -> str:
    """One spelling for everything the patterns compare: no invisible characters, NFKC (fullwidth
    and presentation forms -> plain letters), straight quotes, folded Arabic, lower case."""
    text = unicodedata.normalize("NFKC", _INVISIBLE.sub("", text)).translate(_QUOTES)
    return normalize_for_search(text).lower()


def detect_injection(text: str) -> list[str]:
    """Which injection kinds the text matches (empty list = none)."""
    folded = _fold(text)
    return [
        kind
        for kind, patterns in INJECTION_PATTERNS.items()
        if any(p.search(folded) for p in patterns)
    ]


def check_question(question: str) -> QuestionCheck:
    t0 = time.perf_counter()
    text, pii = redact_pii(question)
    injection = detect_injection(text)
    fired = [f"pii:{label}" for label in pii] + [f"injection:{kind}" for kind in injection]
    latency_ms = (time.perf_counter() - t0) * 1000
    return QuestionCheck(text, fired, bool(injection), round(latency_ms, 3))
