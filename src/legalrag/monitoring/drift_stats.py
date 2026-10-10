"""Drift tests: is today's traffic still like the traffic the system was evaluated on?

No labels are needed: these compare the *inputs* and the system's own *behaviour* (language mix,
question length, which book retrieval lands in, the decider's answerability score) between a
reference window and the current one.

- ``ks_test``: two numeric samples, any shape difference (Kolmogorov-Smirnov), p-value.
- ``wasserstein_distance``: how far the numbers moved, in reference standard deviations
  (an effect size: KS on 10,000 samples flags differences too small to matter).
- ``chi2_test``: two category mixes (language, book), p-value; a new category counts.
- ``js_divergence``: the same, as a bounded distance 0..1 (Jensen-Shannon, base 2).
- ``mmd_test``: several numbers at once (maximum mean discrepancy, RBF kernel, permutation
  p-value): catches a shift no single feature shows.
- ``domain_classifier_auc``: train a model to tell reference from current; AUC ~0.5 = cannot,
  so no drift; close to 1 = the two windows are easy to tell apart.
- ``StormGuard``: drift is a *signal*, not an order. It may trigger an alert or a retraining
  only with enough samples, after a cooldown, and a few times a day at most.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest test class

    statistic: float
    p_value: float
    drift: bool


def ks_test(ref: Sequence[float], cur: Sequence[float], alpha: float = 0.01) -> TestResult:
    res = stats.ks_2samp(ref, cur)
    return TestResult(float(res.statistic), float(res.pvalue), bool(res.pvalue < alpha))


def wasserstein_distance(ref: Sequence[float], cur: Sequence[float]) -> float:
    scale = float(np.std(ref)) or 1.0
    return float(stats.wasserstein_distance(ref, cur)) / scale


def _aligned(ref: Mapping[str, int], cur: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(ref) | set(cur))
    return (np.array([ref.get(k, 0) for k in keys], float),
            np.array([cur.get(k, 0) for k in keys], float))  # fmt: skip


def chi2_test(ref: Mapping[str, int], cur: Mapping[str, int], alpha: float = 0.01) -> TestResult:
    """Goodness of fit of the current counts to the reference mix. +0.5 on every reference
    count, so a category the reference never saw is "very unlikely", not a division by zero."""
    r, c = _aligned(ref, cur)
    expected = (r + 0.5) / (r + 0.5).sum() * c.sum()
    res = stats.chisquare(c, expected)
    return TestResult(float(res.statistic), float(res.pvalue), bool(res.pvalue < alpha))


def js_divergence(ref: Mapping[str, int], cur: Mapping[str, int]) -> float:
    r, c = _aligned(ref, cur)
    p, q = r / r.sum(), c / c.sum()
    m = (p + q) / 2

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return (kl(p, m) + kl(q, m)) / 2


def _rbf_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    def k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None, :] - 2 * a @ b.T
        return np.exp(-gamma * d)

    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


def mmd_test(ref: np.ndarray, cur: np.ndarray, n_perm: int = 200, alpha: float = 0.01,
             seed: int = 0) -> TestResult:  # fmt: skip
    """Permutation test: shuffle the window labels n_perm times; how often is the shuffled MMD
    as large as the real one? Kernel width = median distance (a standard, tuning-free choice)."""
    x, y = np.asarray(ref, float), np.asarray(cur, float)
    pooled = np.vstack([x, y])
    d2 = np.sum((pooled[:, None, :] - pooled[None, :, :]) ** 2, -1)
    gamma = 1.0 / (np.median(d2[d2 > 0]) or 1.0)
    observed = _rbf_mmd2(x, y, gamma)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        hits += _rbf_mmd2(pooled[idx[: len(x)]], pooled[idx[len(x) :]], gamma) >= observed
    p = (hits + 1) / (n_perm + 1)
    return TestResult(observed, p, bool(p < alpha))


def domain_classifier_auc(ref: np.ndarray, cur: np.ndarray, seed: int = 0) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    x = np.vstack([ref, cur])
    y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    model = LogisticRegression(max_iter=1000, random_state=seed)
    return float(cross_val_score(model, x, y, cv=5, scoring="roc_auc").mean())


@dataclass(frozen=True)
class StormGuard:
    min_samples: int = 200  # fewer questions than this: the test has no power, stay quiet
    cooldown: timedelta = timedelta(hours=6)  # one trigger, then time to look at it
    daily_cap: int = 2  # never more than this per 24 h, whatever the tests say

    def allow(self, drifted: bool, samples: int, now: datetime,
              history: Sequence[datetime]) -> tuple[bool, str]:  # fmt: skip
        if samples < self.min_samples:
            return False, f"too few samples ({samples} < {self.min_samples})"
        if not drifted:
            return False, "no drift"
        if history and now - max(history) < self.cooldown:
            return False, f"cooldown ({now - max(history)} since the last trigger)"
        if sum(now - t < timedelta(hours=24) for t in history) >= self.daily_cap:
            return False, f"daily cap ({self.daily_cap}) reached"
        return True, "drift"
