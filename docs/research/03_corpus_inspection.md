# Corpus inspection — `data/raw/egyptian_civil_code.pdf`

Result of a read-only inspection with pypdf 6.10, PyMuPDF 1.28, pdfplumber 0.11, tiktoken (done 2026-10-08).

## Basics
- 170 A4 pages, real text layer (Word for Office 365, 2019), not scanned. Smallest page 2,052 chars → no blank/image-only pages. 30 images, all 2×2 px decorations. No running headers/footers/page numbers.
- Bilingual: each article has Arabic (`مادة N`) then its English translation (`Article N`). ~216K Arabic letters, ~339K Latin letters. Fonts Arial (Identity-H) for Arabic, Calibri for English.
- Size: ~718K chars; ~299K tokens (cl100k_base), ~196K (o200k_base). Expect ~1,100 article chunks of ~180–270 tokens each (AR+EN together).

## Extractor comparison (use pypdf)
| Library | Arabic word order | Lam-alef (`لا`, `الإ`) | Multi-digit Arabic-Indic numbers |
|---|---|---|---|
| **pypdf 6.10** | correct | correct (`لا يجوز` ×125, broken form once) | **reversed** |
| PyMuPDF 1.28 | correct | broken: `اإلصدار`, `ال يجوز`, `اإلسالمية` | reversed |
| pdfplumber 0.11 | reversed (visual order) | — | correct |
"Reversed numbers": `٨٢` where 28 is meant, `٨٤٩١` for 1948, `مادة٥٢٦` for Article 625. Single digits and some numbers (`١٥`) come out right, so you cannot reliably reverse every number in code → **take article numbers from the English side**.

## Structure
- Hierarchy: Arabic `الكتاب` / `الباب` / `الفصل` / `الفرع`; English `BOOK` / `PART` / `SECTION` / `Chapter` / `Section` headings mixed in; numbered sub-headings like `١ -القانون والحق 1. Laws and Rights`. Inconsistent counts (Part ×12, BOOK ×4, Section ×51).
- The law has 1,149 articles. 1,086 have an English `Article N` line on its own; they never repeat and always increase.
- Gaps: Articles **55–80 missing** (PDF note: repealed by Presidential Decree); Articles **389–417 missing with no note** (evidence chapter, repealed); Article **452** exists in Arabic only; Article 1022 is written `Article1022` (no space).
- Order varies: Arabic usually precedes its English header, but sometimes English comes first (Article 83). Some English lines are scrambled (Art. 1022: `f an agreement to the contrary, the cost In the absence o`).
- Articles cross page breaks (page 85 starts mid-article) → join all pages before splitting.

## Samples (pypdf)
Page 1:
```
القانون المدني المصري
قانون الإصدار
مادة ١
يلغي القانون المدني المعمول به أمام المحاكم الوطنية والصادر في ٨٢ أكتوبر سنة
٣٨٨١ ...
الفصل الأول
القانون وتطبيقه
SECTION I
Laws and their Applications
١ -القانون والحق 1. Laws and Rights
مادة١ (
)١ (تسرى النصوص التشريعية على جميع المسائل التي تتناولها هذه
...
Article 1
Provisions of laws govern all matters to which these
```
Page 85:
```
مادة٥٢٦
لا يجوز فى المزارعة أن يترل المستأجر عن الإيجار أو أن يؤجر الأرض
من الباطن إلا برضاء المؤجر.
Article 625
In amodiation, the lessee cannot assign the lease or sub-
let the land amodiated without the consent of the lessor.
مادة ٧٢٦
)١ (إذا انتهت المزارعة قبل انقضاء مدتها ، وجب ان يرد للمستأجر او
```

## Quirks to handle in `ingest/parse.py` and `ingest/normalize.py`
- Paragraph markers come out as mirrored brackets: `)١ (`, `(٢ (`, `( ٢ (` → regex `\)\s*([٠-٩]+)\s*\(` → `(N)`.
- `مادة` header can be split across a line break or written `ماد ة`.
- Stray spaces/line breaks inside words: `ا سترداد`, `وق ت`, `مد\nته ا` (collapse carefully).
- `'` sometimes where `،` should be; full stops at line start in PyMuPDF output.
- Old spellings as-is: `فى`, `الذى`, `مسئول`. Almost no diacritics (tanween in `عوضاً`).
- Normalization for BM25/embeddings: `ى`/`ي`, `أ`/`إ`/`آ`/`ا`, `ة`/`ه`; strip tatweel + diacritics; Arabic-Indic digits → Western.

## Chunking recommendation (adopted)
- One chunk per article; split on `^\s*Article\s*(\d+)\s*$`; assign the Arabic text that precedes the English header to that article (fallback for Art. 83 flip). Keep AR and EN of one article under the same `article_number` metadata (two vector points sharing payload). Carry the current book/chapter/section headings as metadata rather than chunks. Fixed-size split only as a fallback for articles > ~800 tokens (split by numbered paragraph, keep article number on every chunk).
- Treat 55–80 and 389–417 as known repealed ranges (records with `is_repealed=true`, empty text) so validation and evaluation don't flag them; Art. 452 `text_en=""`.

## Machine (Windows host; WSL2 Ubuntu is the dev environment)
- Windows: Python 3.13.11 (Miniconda), pypdf/PyMuPDF/pdfplumber/tiktoken/transformers installed, uv 0.12, docker 29.7, kubectl client 1.36, git 2.55, gh 2.102 (not logged in), no minikube/kind/ollama, no poppler. RTX 3050 Laptop GPU 4 GB VRAM, driver 591.86, Intel Iris Xe; 23.6 GB RAM; i5-12500H 12c/16t.
- WSL2 Ubuntu (user `shobaki`, kernel 6.18): python3 3.14 (use uv to install 3.12), uv 0.12.8, git, gh (`~/.local/bin/gh`), docker via Docker Desktop, `nvidia-smi` works (RTX 3050 4096 MiB), 11 GB RAM visible before `.wslconfig` (now set to 16 GB), no `claude`/`node` yet.
