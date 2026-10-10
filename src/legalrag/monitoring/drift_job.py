"""The drift job: is the last window of traffic still like the traffic we evaluated on?

    uv run python -m legalrag.monitoring.drift_job --hours 24                  # report + metrics
    uv run python -m legalrag.monitoring.drift_job --hours 24 --fail-on-drift  # as a CI gate
    uv run python -m legalrag.monitoring.drift_job reference                   # rebuild reference

Reference: prediction events of the golden + held-out evaluation (``data/monitoring/
reference_events.jsonl``, built from an evaluation run; no question text). Current: the API's
events of the last ``--hours``. Per feature, one or two tests from ``drift_stats``; p-value tests
use a Bonferroni-corrected alpha (several tests at 1 % each would raise false alarms), distance
tests use effect-size thresholds. Outputs, all from the same numbers:
- ``reports/drift/<time>.json`` (what a person reads),
- Prometheus textfile metrics (``rag_drift_score``, ``rag_drift_alert``) for Grafana and the
  ``DriftDetected`` alert, through node-exporter,
- rows in Postgres ``drift_metrics`` / ``drift_runs`` (history, Grafana table) when a database
  URL is configured,
- the exit code with ``--fail-on-drift`` (CI gate).
The StormGuard decides whether a drift may *trigger* anything (alert flag, Phase 6 retraining).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from legalrag.logging_conf import configure_logging, get_logger
from legalrag.monitoring import drift_stats as ds
from legalrag.monitoring.events import PredictionEvent, read_events

log = get_logger(__name__)

ALPHA = 0.01
JS_MAX, WASSERSTEIN_MAX, AUC_MAX = 0.1, 0.25, 0.7  # effect sizes that matter for this traffic


@dataclass
class Row:
    feature: str
    test: str
    statistic: float
    p_value: float | None
    drift: bool


@dataclass
class DriftReport:
    n_reference: int
    n_current: int
    drift: bool
    reason: str
    rows: list[Row] = field(default_factory=list)


def _counts(events: Sequence[PredictionEvent], key: str) -> dict[str, int]:
    return dict(Counter(str(getattr(e, key) or "none") for e in events))


def _matrix(events: Sequence[PredictionEvent], books: list[str]) -> np.ndarray:
    """One numeric row per event for the multivariate tests: length, language, book, refused."""
    return np.array([[math.log1p(e.question_chars), float(e.lang == "ar"), float(e.refused),
                      *[float(e.top_book == b) for b in books]] for e in events])  # fmt: skip


def _categorical(rows: list[Row], name: str, ref: Sequence[PredictionEvent],
                 cur: Sequence[PredictionEvent], alpha: float) -> None:  # fmt: skip
    r, c = _counts(ref, name), _counts(cur, name)
    chi = ds.chi2_test(r, c, alpha)
    rows.append(Row(name, "chi2", chi.statistic, chi.p_value, chi.drift))
    js = ds.js_divergence(r, c)
    rows.append(Row(name, "js", js, None, js > JS_MAX))


def compare(reference: Sequence[PredictionEvent], current: Sequence[PredictionEvent],
            min_samples: int = 50, seed: int = 0) -> DriftReport:  # fmt: skip
    n_ref, n_cur = len(reference), len(current)
    if n_cur < min_samples:
        return DriftReport(n_ref, n_cur, False, f"too few samples ({n_cur} < {min_samples})")
    alpha = ALPHA / 6  # Bonferroni over the p-value tests below (4 chi2/KS + MMD + spare)
    rows: list[Row] = []
    for name in ("lang", "top_book", "refused"):
        _categorical(rows, name, reference, current, alpha)
    ref_len = [e.question_chars for e in reference]
    cur_len = [e.question_chars for e in current]
    ks = ds.ks_test(ref_len, cur_len, alpha)
    rows.append(Row("question_chars", "ks", ks.statistic, ks.p_value, ks.drift))
    w = ds.wasserstein_distance(ref_len, cur_len)
    rows.append(Row("question_chars", "wasserstein", w, None, w > WASSERSTEIN_MAX))
    scores = [[e.answerable_score for e in x if e.answerable_score is not None]
              for x in (reference, current)]  # fmt: skip
    if min(len(s) for s in scores) >= min_samples:  # only when a decider ran in both windows
        ks = ds.ks_test(scores[0], scores[1], alpha)
        rows.append(Row("answerable_score", "ks", ks.statistic, ks.p_value, ks.drift))
    books = sorted({e.top_book for e in [*reference, *current]})
    x_ref, x_cur = _matrix(reference, books), _matrix(current, books)
    mmd = ds.mmd_test(x_ref, x_cur, alpha=alpha, seed=seed)
    rows.append(Row("all", "mmd", mmd.statistic, mmd.p_value, mmd.drift))
    auc = ds.domain_classifier_auc(x_ref, x_cur, seed=seed)
    rows.append(Row("all", "domain_auc", auc, None, auc > AUC_MAX))
    drifted = [f"{r.feature}/{r.test}" for r in rows if r.drift]
    reason = f"drift in {', '.join(drifted)}" if drifted else "no drift"
    return DriftReport(n_ref, n_cur, bool(drifted), reason, rows)


def textfile_lines(report: DriftReport, alert: bool) -> list[str]:
    """Prometheus text format for node-exporter's textfile collector."""
    lines = ["# HELP rag_drift_score Drift test statistic (current window vs reference)",
             "# TYPE rag_drift_score gauge"]  # fmt: skip
    lines += [f'rag_drift_score{{feature="{r.feature}",test="{r.test}"}} {r.statistic:.6g}'
              for r in report.rows]  # fmt: skip
    lines += ["# HELP rag_drift_alert 1 when drift passed the storm guard (may trigger action)",
              "# TYPE rag_drift_alert gauge", f"rag_drift_alert {int(alert)}",
              "# HELP rag_drift_window_events Events in the current window",
              "# TYPE rag_drift_window_events gauge", f"rag_drift_window_events {report.n_current}"]  # fmt: skip
    return lines


def _write_atomic(path: Path, text: str) -> None:
    """node-exporter may read at any moment: write a temp file, then rename over the old one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def _trigger_history(path: Path) -> list[datetime]:
    if not path.is_file():
        return []
    return [datetime.fromisoformat(t) for t in json.loads(path.read_text(encoding="utf-8"))]


def run(args: argparse.Namespace) -> int:
    now = datetime.now(UTC)
    lines = Path(args.reference).read_text(encoding="utf-8").splitlines()
    reference = [PredictionEvent.model_validate_json(x) for x in lines if x.strip()]
    current = read_events(args.events, since=now - timedelta(hours=args.hours))
    report = compare(reference, current, min_samples=args.min_samples)
    history_path = Path(args.state) / "drift_triggers.json"
    history = _trigger_history(history_path)
    guard = ds.StormGuard(min_samples=args.min_samples)
    alert, why = guard.allow(report.drift, report.n_current, now, history)
    if alert:
        _write_atomic(history_path, json.dumps([t.isoformat() for t in [*history, now]][-20:]))
    out = {"run_at": now.isoformat(), "window_hours": args.hours, "triggered": alert,
           "guard": why, **asdict(report)}  # fmt: skip
    _write_atomic(Path(args.out) / f"{now:%Y%m%dT%H%M%SZ}.json", json.dumps(out, indent=2) + "\n")
    if args.textfile:
        _write_atomic(Path(args.textfile), "\n".join(textfile_lines(report, alert)) + "\n")
    if args.db_url:
        from legalrag.monitoring.drift_store import store_run

        store_run(args.db_url, out)
    log.info("drift_run", drift=report.drift, triggered=alert, guard=why, reason=report.reason,
             n_reference=report.n_reference, n_current=report.n_current)  # fmt: skip
    return 1 if args.fail_on_drift and report.drift else 0


def build_reference(predictions: list[Path], articles: Path, out: Path) -> int:
    from legalrag.eval.metrics import Prediction
    from legalrag.ingest.validate import load_articles
    from legalrag.monitoring.events import event_from_prediction

    books = {a.article_number: a.book for a in load_articles(articles)}
    serving = {"prompt_version": "eval", "llm_model": "eval", "index_collection": "eval",
               "decider": "eval"}  # fmt: skip
    events = [event_from_prediction(Prediction.model_validate_json(line), books, serving)
              for path in predictions
              for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]  # fmt: skip
    _write_atomic(out, "".join(e.model_dump_json() + "\n" for e in events))
    log.info("reference_built", events=len(events), out=str(out))
    return 0


def main(argv: list[str] | None = None) -> None:
    from legalrag.settings import get_settings

    s = get_settings()
    p = argparse.ArgumentParser(description="drift job")
    p.add_argument("command", nargs="?", default="run", choices=["run", "reference"])
    p.add_argument("--reference", default="data/monitoring/reference_events.jsonl")
    p.add_argument("--events", default=s.events_dir)
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--min-samples", type=int, default=50)
    p.add_argument("--out", default="reports/drift")
    p.add_argument("--state", default="reports/drift", help="where the trigger history lives")
    p.add_argument("--textfile", default="monitoring/textfile/drift.prom")
    p.add_argument("--db-url", default=s.monitoring_db_url.get_secret_value() or None)
    p.add_argument("--fail-on-drift", action="store_true")
    p.add_argument("--from-runs", nargs="+", default=["reports/eval/e2e-dense-jev/predictions.jsonl",
                                                       "reports/eval/heldout-dense-jev/predictions.jsonl"])  # fmt: skip
    args = p.parse_args(argv)
    configure_logging("INFO")
    if args.command == "reference":
        sys.exit(build_reference([Path(x) for x in args.from_runs], Path(s.articles_path),
                                 Path(args.reference)))  # fmt: skip
    sys.exit(run(args))


if __name__ == "__main__":
    main()
