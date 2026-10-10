"""Input guardrails: PII redaction and prompt-injection detection, run before retrieval."""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass

from legalrag.ingest.normalize import normalize_for_search

_DIGITS = str.maketrans(  # Arabic-Indic, Persian and fullwidth digits -> 0-9, one-to-one
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹０１２３４５６７８９", "012345678901234567890123456789"
)
# zero-width and direction marks, often used to split a word so a pattern misses it
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿]")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'"})

# national ID: century (2=1900s, 3=2000s) + YYMMDD + governorate(2) + serial(4) + check digit
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "national_id": re.compile(
        r"(?<!\d)[23]\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{7}(?!\d)"
    ),
    "phone": re.compile(r"(?<![\d+])(?:\+20|0020|20|0)[\s-]?1[0125](?:[\s-]?\d){8}(?!\d)"),
    # bounded parts (RFC limits) avoid slow backtracking on long input without "@"
    "email": re.compile(r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){1,8}"),
}

# Patterns run on _fold(text), so Arabic ones are written folded (أ→ا, ة→ه, ى→ي, no diacritics).
# Each needs the assistant as target ("your rules", "you are now an AI"): legal questions such as
# "ignore the clause" or "you are now the tenant" must pass.
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
        # not مثل: "مثل دور" also means "such as the role"
        re.compile(_AND + r"(?:العب|خذ)\s+دور\b"),
    ],
    "delimiter": [  # chat-template or prompt delimiters typed by the user
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
    """Matching form: strip invisible chars, NFKC, straight quotes, fold Arabic, lowercase."""
    text = unicodedata.normalize("NFKC", _INVISIBLE.sub("", text)).translate(_QUOTES)
    return normalize_for_search(text).lower()


def detect_injection(text: str) -> list[str]:
    """Injection kinds the text matches."""
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
