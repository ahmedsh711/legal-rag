# Module 1 report: corpus pipeline (PDF → articles.json, versioned with DVC)

**Goal:** turn the bilingual Egyptian Civil Code PDF into one validated record per article, reproducibly, so every later answer can cite an article number a lawyer can check.

## Result

| Metric (from `dvc metrics show`) | Value |
|---|---|
| Records (articles 1–1149, no gaps) | 1,149 |
| Live articles (Arabic + English text) | 1,093 |
| Repealed articles (flagged, kept as records) | 56 (54–80, 389–417) |
| Known source anomalies (flagged in `params.yaml` and on the records) | 5 (1022, 1021, 970, 519, 1060) |
| Validation errors / warnings | 0 / 5 (the 5 documented anomalies) |
| Longest article text, Arabic / English (chars) | 1,419 / 2,141 |
| Mean Arabic text length (chars) | 235.8 |
| Hierarchy captured | preliminary title + 4 books, 18 chapters, 54 sections, 136 topics |

## How the PDF was read

Extractor comparison on the same pages (inspection notes in `docs/research/03_corpus_inspection.md`):

| Library | Arabic word order | lam-alef `لا` | multi-digit Arabic numbers | Verdict |
|---|---|---|---|---|
| **pypdf** | correct | correct | digit-reversed | **chosen** |
| PyMuPDF | correct | broken (`ال يجوز`) | digit-reversed | rejected |
| pdfplumber | reversed | n/a | correct | rejected |

Because Arabic multi-digit numbers come out reversed (`مادة٥٢٦` = Article 625) and not consistently, **article numbers come from the English `Article N` headers**. Arabic header digits, read reversed, are only used for an Arabic block with no English header (they matched the next English number in 1,078 of 1,087 cases).

Layouts the parser handles, all found in the real PDF and each covered by a unit test:

| Layout | Example | Handling |
|---|---|---|
| Arabic block, then `Article N`, then English | most articles | default |
| Both headers on one line (`مادة٣٨( Article 83`) | 83, 203, 387, 676, 888, 1110, 1115 | `COMBINED_HEADER` regex |
| English header in the middle of the Arabic text | 120, 187, 428, 901 | Arabic keeps collecting until English text starts |
| Arabic sentence and English sentence on one line | 428, 901 | split at the last Arabic letter or diacritic |
| Paragraphs alternate AR(1), EN(1), AR(2), EN(2) | 6, 42 | mode `ar2`; Arabic after English is body if it starts with a paragraph marker, is ≥ 50 chars, or ends with `.`/`،` |
| Header lost its first letter (`rticle 452`) | 452 | `A?rticle` |
| Cross-reference starting a line (`Article 444 .`) | inside 450, 885, 988 | dropped by a longest-increasing-subsequence over header numbers |
| Repeal notes | `* Articles 54-80 have been repealed by Presidential Decree.` / `Articles 389-417 repealed` | repealed records generated from the notes, not hard-coded |
| Mirrored paragraph brackets `)١ (` | everywhere | fixed to `(١)` |

Text is stored as written (old spellings like `فى`, `مسئول` kept); only a *search copy* is normalized later (`normalize_for_search`, version `v1`).

## Spot check: 20 random live articles (seed 42)

Articles 52, 82, 89, 93, 206, 219, 237, 256, 313, 464, 504, 508, 514, 533, 558, 620, 916, 921, 976, 1092. I read the Arabic and English of each:

- 20/20: Arabic and English belong to the same article and say the same thing; headings match the code's structure (e.g. 558 → `الفصل الأول: الإيجار`, topic `الإيجار بوجه عام / أركان الإيجار`).
- 1/20 with extraction damage: Article 219's Arabic starts `كون أعذار المدين` (first letter of `يكون` lost). Same class of defect as the `rticle 452` header.
- Source typos are kept as they are (`هلال` for `هلاك` in 504, `وغلى` in 533); fixing the law's text is out of scope.
- One heading prefix reads `أو- أركان العقد` (93): an ordinal lost in extraction; cosmetic.

## Known limitations (honest list)

Every item below is listed in `params.yaml` under `corpus.known_anomalies` with a reason, copied onto the record as a `quality_flag`, and allowed only for the one check it breaks.

1. **Article 1022:** its Arabic text is merged into Article 1021's Arabic block in the source PDF, and its English lines are scrambled (`f an agreement to the contrary, the cost In the absence o`). Kept English-only (flag `arabic_text_inside_article_1021`); Article 1021 carries the mirror flag `contains_arabic_of_article_1022`.
2. **Numbers inside Arabic article text** (cross-references like `للمادة ٣٦٢`) are digit-reversed by extraction and cannot be fixed reliably. Citations never use them: they always come from `article_number`.
3. A few lines lost their first character in extraction (Article 219, the `rticle 452` header).
4. **Arabic and English differ in the source** (found by the length-ratio check): 970's Arabic includes later amendments the English lacks; 519 and 1060 have shorter Arabic than English in the PDF. Mirrored AR/EN evaluation questions on these articles may legitimately disagree.

## Versioning with DVC

- Remote: MinIO bucket `s3://dvc` (credentials in `.dvc/config.local`, never committed). MinIO's community images were withdrawn from Docker Hub in 2025, so local dev uses the frozen `bitnamilegacy/minio:2025.7.23`; production would use AWS S3 or a maintained S3 store.
- `data/raw/egyptian_civil_code.pdf.dvc` pins the PDF (md5 `5086ef5f…`); `dvc.lock` pins `articles.json` (md5 `d21eead0…`).
- Pipeline `dvc.yaml`: `parse → validate`; validation thresholds live in `params.yaml`.

What was demonstrated (commands and output):

```text
$ dvc repro            # second run
Stage 'parse' didn't change, skipping
Stage 'validate' didn't change, skipping

# change corpus.max_chars 2500 -> 2400 in params.yaml
$ dvc status
validate: changed deps: params.yaml: modified: corpus
$ dvc repro
Stage 'parse' didn't change, skipping
Running stage 'validate':           # only the stage that depends on the param re-ran

# wipe the cache and both data files, then restore from MinIO
$ rm -rf .dvc/cache data/raw/egyptian_civil_code.pdf data/processed/articles.json
$ dvc pull
2 files fetched and 2 files added
5086ef5f16f15d856278db943e58e948  data/raw/egyptian_civil_code.pdf
d21eead0566355413ae895f92f722748  data/processed/articles.json   # same bytes as before
```

A mistake worth recording: the first `dvc push` silently skipped the PDF. The repo's `.gitignore` ignored `data/raw/*`, and DVC does not collect `.dvc` files from a folder git ignores. The fix was to let DVC write its own `.gitignore` next to each tracked file. The wipe-and-pull test above is what caught it.

## Decisions

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Extractor | pypdf | PyMuPDF, pdfplumber | only one that keeps both Arabic order and `لا` intact |
| Source of article numbers | English headers | Arabic headers | Arabic digits are reversed inconsistently |
| Repealed articles | kept as flagged records | dropped | "that article no longer exists" is a correct answer; a silent gap invites hallucination |
| Unfixable source defect (1022) | flagged in params, warning | silently patched / dropped | honesty clause; the reviewer can see it |
| Validation thresholds | `params.yaml` (DVC param) | constants in code | a change is versioned and re-runs only `validate` |
| Object store | frozen Bitnami MinIO | SeaweedFS, RustFS | it is the tool the course names; alternatives noted for production |

## Code review and what changed

Two independent reviews ran on this branch (a Python reviewer and an ML-pipeline reviewer). Accepted and fixed:

| Finding | Fix |
|---|---|
| A wrong PDF without the start marker would be parsed from page 1 | `parse_lines` raises `ValueError` |
| Bad inputs crashed with raw tracebacks; a failed run left stale metrics | typed `params.py` loader, per-record errors with position, exit code 1, metrics always rewritten |
| `parse_lines` was ~130 lines with string modes | `Mode` enum, one handler per mode, `Row`/`Kind` types, `_build_articles` split out |
| The repeal note ended up inside Article 54's English text; duplicates were silently dropped; absurd repeal ranges accepted | `NOTE` mode keeps notes out of text; duplicates and unnumbered text raise; spans over 100 raise |
| Validation could not see text moving between articles | English/Arabic length-ratio band, 8 pinned golden phrases, book start articles, contiguous books/chapters, per-language length caps at ~1.25x the measured max |
| Anomalies silenced every check on an article | each anomaly allows only named checks and sets `quality_flags` on the record |
| No lineage on the artifact | `articles.meta.json` (PDF md5, schema/normalization/pypdf versions) is a DVC output |
| DVC deps incomplete (a pypdf upgrade would not rebuild) | parse/validate depend on the whole `ingest` package, settings, logging and `uv.lock` |
| Schema: no stable id, citation could drift, unknown fields accepted | `id` (`eg-civil-147`), `citation`, `citation_ar` derived and checked; `extra="forbid"` |
| Tatweel counted as an Arabic letter; some hamza marks not stripped | fixed in `normalize.py` (search normalization stays `v1`: letters and digits are unchanged) |

Declined, with reasons (cost if wrong in brackets):
- Compare paragraph-marker counts between languages: the English translation does not number paragraphs, so counts never match (none: the ratio check covers mis-splits).
- Git SHA inside `articles.meta.json`: it would change the output on every commit without any data change; MLflow runs tag the git SHA instead (low).
- Page ranges and paragraph-level citations: not needed for article-level citation in v1 (low).
- CI cannot run the real-PDF tests because the DVC remote is a local MinIO; the skip states the reason, and the full suite runs locally before every PR (medium: a parser regression could pass CI; mitigated by the local run and the DVC validate stage).

## Tests

`uv run pytest`: 84 passed, coverage 97.20% (unit tests on synthetic samples of every layout above + integration tests on the real PDF, skipped when the PDF is not pulled).

## Definition of done (Phase 1)

- [x] PDF converted to structured JSON, one record per article, numbers normalized to integers
- [x] Extraction validated: repealed articles flagged, Arabic text spot-checked on 20 articles
- [x] Validation stage fails the pipeline on gaps, empty text, mixed scripts or failed splits
- [x] Source PDF and derived JSON tracked with DVC; `dvc pull` restores them; `dvc repro` rebuilds the JSON
- [x] Walkthrough page `docs/walkthrough/01-corpus.html` explains every file and function
