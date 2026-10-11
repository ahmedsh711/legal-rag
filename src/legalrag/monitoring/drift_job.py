"""Drift job: compare a recent window of prediction events with the evaluation reference.

    uv run python -m legalrag.monitoring.drift_job --hours 24                  # report + metrics
    uv run python -m legalrag.monitoring.drift_job --hours 24 --fail-on-drift  # as a CI gate
    uv run python -m legalrag.monitoring.drift_job aa --since <time>           # false-alarm rate
    uv run python -m legalrag.monitoring.drift_job reference   # rebuild reference + query centroid

Features are input (language, question length, both via MMD and domain AUC; the question's cosine
to the reference centroid), retrieval (book of the top article) and behaviour (refusals, decider
score; only compared when the reference's model and prompt served the window). P-value tests
share a Bonferroni-corrected alpha; effect sizes (JS, Wasserstein, domain AUC) only confirm a
significant test. Writes a JSON report, Prometheus textfile metrics (``rag_drift_*``) and, with a
database URL, ``drift_runs``/``drift_metrics`` rows.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from legalrag.logging_conf import configure_logging, get_logger
from legalrag.monitoring import drift_stats as ds
from legalrag.monitoring.events import PredictionEvent, read_events
from legalrag.monitoring.files import write_atomic

log = get_logger(__name__)

ALPHA = 0.01
JS_MAX, WASSERSTEIN_MAX, AUC_MAX = 0.1, 0.25, 0.7  # smallest effect sizes that count as material
DETECTORS = (("question_chars", "adwin"), ("refused", "hddm_a"))  # streaming change detection


@dataclass
class Row:
    feature: str
    test: str
    statistic: float
    p_value: float | None
    drift: bool
    kind: str = "input"  # input | retrieval | behaviour


@dataclass
class ChangePoint:
    feature: str
    detector: str
    index: int  # in the time-ordered window; detectors fire a few events after the change
    ts: str


@dataclass
class DriftReport:
    n_reference: int
    n_current: int
    drift: bool
    reason: str
    rows: list[Row] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    change_points: list[ChangePoint] = field(default_factory=list)


def _counts(events: Sequence[PredictionEvent], key: str) -> dict[str, int]:
    return dict(Counter(str(getattr(e, key) or "none") for e in events))


def _serving(events: Sequence[PredictionEvent]) -> tuple[str, str]:
    """The model and prompt that served most of a window."""
    return Counter((e.llm_model, e.prompt_version) for e in events).most_common(1)[0][0]


def _cosines(events: Sequence[PredictionEvent]) -> list[float]:
    return [e.query_cosine for e in events if e.query_cosine is not None]


def _inputs(events: Sequence[PredictionEvent]) -> np.ndarray:
    """Input features only: log question length and is-Arabic."""
    return np.array([[math.log1p(e.question_chars), float(e.lang == "ar")] for e in events])


def _p_rows(
    ref: Sequence[PredictionEvent],
    cur: Sequence[PredictionEvent],
    seed: int,
    behaviour: bool,
    min_samples: int,
    cosines: tuple[list[float], list[float]] | None,
) -> list[Row]:
    """P-value tests, flagged against a Bonferroni-corrected alpha."""
    tests: list[tuple[str, str, Callable[[], Any]]] = [
        ("lang", "input", lambda: ds.chi2_test(_counts(ref, "lang"), _counts(cur, "lang"))),
        (
            "question_chars",
            "input",
            lambda: ds.ks_test([e.question_chars for e in ref], [e.question_chars for e in cur]),
        ),
        (
            "top_book",
            "retrieval",
            lambda: ds.chi2_test(_counts(ref, "top_book"), _counts(cur, "top_book")),
        ),
    ]
    if cosines:
        tests.append(("query_cosine", "input", lambda: ds.ks_test(*cosines)))
    if behaviour:
        tests.append(
            (
                "refused",
                "behaviour",
                lambda: ds.chi2_test(_counts(ref, "refused"), _counts(cur, "refused")),
            )
        )
        scores = [
            [e.answerable_score for e in x if e.answerable_score is not None] for x in (ref, cur)
        ]
        if min(len(s) for s in scores) >= min_samples:  # only when a decider ran in both windows
            tests.append(("answerable_score", "behaviour", lambda: ds.ks_test(*scores)))
    rows = []
    for feature, kind, run in tests:
        r = run()
        test = "chi2" if feature in ("lang", "top_book", "refused") else "ks"
        rows.append(Row(feature, test, r.statistic, r.p_value, False, kind))
    mmd = ds.mmd_test(_inputs(ref), _inputs(cur), alpha=ALPHA / (len(rows) + 1), seed=seed)
    rows.append(Row("inputs", "mmd", mmd.statistic, mmd.p_value, False, "input"))
    alpha = ALPHA / len(rows)  # Bonferroni over the p-value tests actually run
    for row in rows:
        row.drift = row.p_value is not None and row.p_value < alpha
    return rows


def _effect_rows(
    ref: Sequence[PredictionEvent],
    cur: Sequence[PredictionEvent],
    seed: int,
    behaviour: bool,
    cosines: tuple[list[float], list[float]] | None,
) -> list[Row]:
    rows = []
    for feature, kind in (
        ("lang", "input"),
        ("top_book", "retrieval"),
        *([("refused", "behaviour")] if behaviour else []),
    ):
        js = ds.js_divergence(_counts(ref, feature), _counts(cur, feature))
        rows.append(Row(feature, "js", js, None, js > JS_MAX, kind))
    w = ds.wasserstein_distance([e.question_chars for e in ref], [e.question_chars for e in cur])
    rows.append(Row("question_chars", "wasserstein", w, None, w > WASSERSTEIN_MAX, "input"))
    if cosines:
        w = ds.wasserstein_distance(*cosines)
        rows.append(Row("query_cosine", "wasserstein", w, None, w > WASSERSTEIN_MAX, "input"))
    auc = ds.domain_classifier_auc(_inputs(ref), _inputs(cur), seed=seed)
    rows.append(Row("inputs", "domain_auc", auc, None, auc > AUC_MAX, "input"))
    return rows


def compare(
    reference: Sequence[PredictionEvent],
    current: Sequence[PredictionEvent],
    min_samples: int = 50,
    seed: int = 0,
) -> DriftReport:
    if not reference:
        raise ValueError("empty reference: build it first (drift_job reference)")
    n_ref, n_cur = len(reference), len(current)
    if n_cur < min_samples:
        return DriftReport(n_ref, n_cur, False, f"too few samples ({n_cur} < {min_samples})")
    notes = []
    behaviour = _serving(reference) == _serving(current)
    if not behaviour:
        notes.append(
            f"behaviour not compared: model or prompt changed "
            f"({_serving(reference)} -> {_serving(current)}), that is a change, not drift"
        )
    cosines = (_cosines(reference), _cosines(current))
    if min(len(c) for c in cosines) < min_samples:  # e.g. events from before it was recorded
        notes.append(
            f"query_cosine not compared: {len(cosines[0])} reference and {len(cosines[1])} "
            f"current events have one, below {min_samples}"
        )
        cosines = None
    rows = _p_rows(reference, current, seed, behaviour, min_samples, cosines)
    # on small windows an effect size alone fires on noise, so it only confirms a significant test
    significant = {r.feature for r in rows if r.drift}
    rows += [
        replace(r, drift=r.drift and r.feature in significant)
        for r in _effect_rows(reference, current, seed, behaviour, cosines)
    ]
    parts = []
    for kind in ("input", "retrieval", "behaviour"):
        if hits := sorted({f"{r.feature}/{r.test}" for r in rows if r.drift and r.kind == kind}):
            parts.append(f"{kind} drift: {', '.join(hits)}")
    return DriftReport(n_ref, n_cur, bool(parts), "; ".join(parts) or "no drift", rows, notes)


def aa_check(events: Sequence[PredictionEvent], runs: int = 100, seed: int = 0) -> dict[str, float]:
    """A/A test: false-alarm rate per test over ``runs`` random half splits of one window."""
    rng = random.Random(seed)
    flags: Counter[str] = Counter()
    for i in range(runs):
        shuffled = list(events)
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        report = compare(shuffled[:half], shuffled[half:], min_samples=1, seed=i)
        for row in report.rows:
            flags[f"{row.feature}/{row.test}"] += row.drift
        flags["any"] += report.drift
    keys = sorted(flags)
    return {k: round(flags[k] / runs, 3) for k in keys}


def change_points(events: Sequence[PredictionEvent]) -> list[ChangePoint]:
    """Where in the window the traffic changed: ADWIN on question length, HDDM_A on refusals.

    HDDM_A is DDM with Hoeffding bounds; plain DDM fires on about a third of steady 300-event
    windows at a 10 % refusal rate."""
    from river import drift

    make = {"adwin": drift.ADWIN, "hddm_a": drift.binary.HDDMA}
    ordered = sorted(events, key=lambda e: e.ts)
    points = []
    for feature, name in DETECTORS:
        detector = make[name]()
        for i, event in enumerate(ordered):
            detector.update(getattr(event, feature))
            if detector.drift_detected:
                points.append(ChangePoint(feature, name, i, event.ts.isoformat()))
    return points


def textfile_lines(report: DriftReport, alert: bool, last_trigger: datetime | None) -> list[str]:
    """Prometheus text format for node-exporter's textfile collector."""
    lines = [
        "# HELP rag_drift_score Drift test statistic (current window vs reference)",
        "# TYPE rag_drift_score gauge",
    ]
    lines += [
        f'rag_drift_score{{feature="{r.feature}",test="{r.test}",kind="{r.kind}"}} '
        f"{r.statistic:.6g}"
        for r in report.rows
    ]
    lines += [
        "# HELP rag_drift_detected 1 when the latest run found drift (the DriftDetected alert)",
        "# TYPE rag_drift_detected gauge",
        f"rag_drift_detected {int(report.drift)}",
        "# HELP rag_drift_alert 1 when the latest drift passed the storm guard",
        "# TYPE rag_drift_alert gauge",
        f"rag_drift_alert {int(alert)}",
        "# HELP rag_drift_window_events Events in the current window",
        "# TYPE rag_drift_window_events gauge",
        f"rag_drift_window_events {report.n_current}",
        "# HELP rag_drift_change_points Changes streaming detectors found inside the window",
        "# TYPE rag_drift_change_points gauge",
    ]
    found = Counter((p.feature, p.detector) for p in report.change_points)
    lines += [
        f'rag_drift_change_points{{feature="{feature}",detector="{name}"}} {found[feature, name]}'
        for feature, name in DETECTORS
    ]
    if last_trigger is not None:
        lines += [
            "# HELP rag_drift_last_trigger_timestamp_seconds When drift last passed the guard",
            "# TYPE rag_drift_last_trigger_timestamp_seconds gauge",
            f"rag_drift_last_trigger_timestamp_seconds {last_trigger.timestamp():.0f}",
        ]
    return lines


def parse_since(value: str) -> datetime:
    """An ISO time; without a timezone it is UTC (events are stored in UTC)."""
    since = datetime.fromisoformat(value)
    return since if since.tzinfo else since.replace(tzinfo=UTC)


def _trigger_history(path: Path) -> list[datetime]:
    if not path.is_file():
        return []
    return [datetime.fromisoformat(t) for t in json.loads(path.read_text(encoding="utf-8"))]


def _load_reference(path: Path) -> list[PredictionEvent]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [PredictionEvent.model_validate_json(x) for x in lines if x.strip()]


def run(args: argparse.Namespace) -> int:
    now = datetime.now(UTC)
    reference = _load_reference(Path(args.reference))
    since = parse_since(args.since) if args.since else now - timedelta(hours=args.hours)
    until = parse_since(args.until) if args.until else None
    current = read_events(args.events, since=since, until=until)
    report = compare(reference, current, min_samples=args.min_samples)
    # informational, not part of the verdict: a change inside the window is not a difference
    # from the reference, and these detectors are outside the Bonferroni-corrected alpha
    report = replace(report, change_points=change_points(current))
    history_path = Path(args.state) / "drift_triggers.json"
    history = _trigger_history(history_path)
    guard = ds.StormGuard(min_samples=args.guard_min_samples)
    alert, why = guard.allow(report.drift, report.n_current, now, history)
    if alert:
        history = [*history, now]
    out = {
        "run_at": now.isoformat(),
        "window_hours": args.hours,
        "since": since.isoformat(),
        "until": until.isoformat() if until else None,
        "triggered": alert,
        "guard": why,
        **asdict(report),
    }
    write_atomic(Path(args.out) / f"{now:%Y%m%dT%H%M%SZ}.json", json.dumps(out, indent=2) + "\n")
    if args.textfile:
        last = max(history) if history else None
        write_atomic(Path(args.textfile), "\n".join(textfile_lines(report, alert, last)) + "\n")
    if args.db_url:
        from legalrag.monitoring.drift_store import store_run

        try:
            store_run(args.db_url, out)
        except Exception:  # noqa: BLE001 - losing one history row must not block the alert
            log.exception("drift_store_failed")
    if alert:  # written last so a crash above does not use up the cooldown
        write_atomic(history_path, json.dumps([t.isoformat() for t in history][-20:]) + "\n")
    log.info(
        "drift_run",
        drift=report.drift,
        triggered=alert,
        guard=why,
        reason=report.reason,
        notes=report.notes,
        n_reference=report.n_reference,
        n_current=report.n_current,
        change_points=len(report.change_points),
    )
    return 1 if args.fail_on_drift and report.drift else 0


def run_aa(args: argparse.Namespace) -> int:
    since = (
        parse_since(args.since) if args.since else datetime.now(UTC) - timedelta(hours=args.hours)
    )
    until = parse_since(args.until) if args.until else None
    current = read_events(args.events, since=since, until=until)
    rates = aa_check(current, runs=args.runs)
    out = {
        "since": since.isoformat(),
        "until": until.isoformat() if until else None,
        "events": len(current),
        "runs": args.runs,
        "fpr": rates,
    }
    write_atomic(
        Path(args.out) / f"aa-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json",
        json.dumps(out, indent=2) + "\n",
    )
    log.info("drift_aa", events=len(current), runs=args.runs, **{"fpr_any": rates.get("any")})
    return 0


def build_reference(
    predictions: list[Path],
    articles: Path,
    out: Path,
    serving: dict[str, str],
    embedder: Any,
    centroid_path: Path,
) -> int:
    """Reference events from evaluation runs, and the centroid of their questions' embeddings."""
    from legalrag.eval.metrics import Prediction
    from legalrag.guardrails import check_question
    from legalrag.ingest.validate import load_articles
    from legalrag.monitoring.events import event_from_prediction
    from legalrag.monitoring.query_centroid import reference_centroid

    books = {a.article_number: a.book for a in load_articles(articles)}
    preds = [
        Prediction.model_validate_json(line)
        for path in predictions
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # the API embeds the PII-redacted question, and nothing when a guard blocks it
    checks = [check_question(p.question) for p in preds]
    searched = [i for i, c in enumerate(checks) if not c.blocked]
    vectors = [e.dense for e in embedder.encode([checks[i].text for i in searched])]
    centroid, cosines = reference_centroid(vectors, embedder.model_name, embedder.revision)
    cosine_of = dict(zip(searched, cosines, strict=True))
    events = [
        event_from_prediction(p, books, serving, cosine_of.get(i)) for i, p in enumerate(preds)
    ]
    write_atomic(out, "".join(e.model_dump_json() + "\n" for e in events))
    centroid.save(centroid_path)
    meta = {
        "built_at": datetime.now(UTC).isoformat(),
        "from_runs": [str(p) for p in predictions],
        "events": len(events),
        "query_centroid": str(centroid_path),
        "embedding_model": centroid.model,
        "embedding_revision": centroid.revision,
        **serving,
    }
    write_atomic(out.with_suffix(".meta.json"), json.dumps(meta, indent=2) + "\n")
    log.info("reference_built", events=len(events), out=str(out), **serving)
    return 0


def main(argv: list[str] | None = None) -> None:
    from legalrag.settings import get_settings

    s = get_settings()
    p = argparse.ArgumentParser(description="drift job")
    p.add_argument("command", nargs="?", default="run", choices=["run", "aa", "reference"])
    p.add_argument("--reference", default="data/monitoring/reference_events.jsonl")
    p.add_argument("--centroid", default=s.query_centroid_path, help="reference: query centroid")
    p.add_argument("--events", default=s.events_dir)
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--since", default=None, help="ISO time (UTC if no zone); overrides --hours")
    p.add_argument("--until", default=None, help="ISO time (UTC if no zone); end of the window")
    p.add_argument("--min-samples", type=int, default=50, help="below this: no statistics")
    p.add_argument("--guard-min-samples", type=int, default=200, help="below this: no trigger")
    p.add_argument("--runs", type=int, default=100, help="aa: random splits")
    p.add_argument("--out", default="reports/drift")
    p.add_argument("--state", default="reports/drift", help="where the trigger history lives")
    p.add_argument("--textfile", default="monitoring/textfile/drift.prom")
    p.add_argument("--db-url", default=s.monitoring_db_url.get_secret_value() or None)
    p.add_argument("--fail-on-drift", action="store_true")
    p.add_argument(
        "--from-runs",
        nargs="+",
        default=[
            "reports/eval/e2e-dense-jev/predictions.jsonl",
            "reports/eval/heldout-dense-jev/predictions.jsonl",
        ],
    )
    p.add_argument(
        "--serving",
        nargs=3,
        metavar=("LLM_MODEL", "PROMPT_VERSION", "DECIDER"),
        default=["gemini-3.1-flash-lite", "v3", "jev"],
        help="reference: what produced the evaluation runs",
    )
    args = p.parse_args(argv)
    configure_logging("INFO")
    if args.command == "reference":
        from legalrag.index.embedder import BgeM3Embedder

        serving = {
            "llm_model": args.serving[0],
            "prompt_version": args.serving[1],
            "decider": args.serving[2],
            "index_collection": "eval",
        }
        sys.exit(
            build_reference(
                [Path(x) for x in args.from_runs],
                Path(s.articles_path),
                Path(args.reference),
                serving,
                # the API's query settings, so reference and live cosines are comparable
                BgeM3Embedder(
                    s.embedding_model,
                    s.embedding_device,
                    s.query_max_length,
                    revision=s.embedding_revision,
                ),
                Path(args.centroid),
            )
        )
    sys.exit(run_aa(args) if args.command == "aa" else run(args))


if __name__ == "__main__":
    main()
