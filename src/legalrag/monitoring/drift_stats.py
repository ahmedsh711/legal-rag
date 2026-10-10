"""Drift tests: is today's traffic still like the traffic the system was evaluated on?

No labels are needed: these compare the *inputs* and the system's own *behaviour* (language mix,
question length, which book retrieval lands in, the decider's answerability score) between a
reference window and the current one.

- ``ks_test``: two numeric samples, any shape difference (Kolmogorov-Smirnov), p-value.
- ``wasserstein_distance``: how far the numbers moved, in reference standard deviations
  (an effect size: KS on 10,000 samples flags differences too small to matter).
- ``chi2_test``: two category mixes (language, book), p-value. A two-sample test: both windows
  are samples, the reference is not treated as the truth. Categories too rare for the
  approximation (expected count < 5) are merged into "other" first.
- ``js_divergence``: the same comparison as a bounded distance 0..1 (Jensen-Shannon, base 2).
- ``mmd_test``: several numbers at once (maximum mean discrepancy, RBF kernel, permutation
  p-value): catches a shift no single feature shows.
- ``domain_classifier_auc``: train a model to tell reference from current; AUC ~0.5 = cannot,
  so no drift; close to 1 = the two windows are easy to tell apart.
- ``StormGuard``: drift is a *signal*, not an order. It may trigger an alert or a retraining
  only with enough samples, after a cooldown, and a few times a day at most.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from scipy import stats

MIN_EXPECTED = 5  # chi-square's approximation needs this many expected counts per cell
MMD_MAX_POINTS = 1000  # per window: MMD costs O(n^2), a random subsample keeps it seconds


@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest test class

    statistic: float
    p_value: float
    drift: bool


def ks_test(ref: Sequence[float], cur: Sequence[float], alpha: float = 0.01) -> TestResult:
    """Do two numeric samples come from the same distribution? (Kolmogorov-Smirnov)"""
    res = stats.ks_2samp(ref, cur)
    return TestResult(float(res.statistic), float(res.pvalue), bool(res.pvalue < alpha))


def wasserstein_distance(ref: Sequence[float], cur: Sequence[float]) -> float:
    """How far the distribution moved, in standard deviations of the reference."""
    scale = float(np.std(ref)) or 1.0
    return float(stats.wasserstein_distance(ref, cur)) / scale


def _aligned(ref: Mapping[str, int], cur: Mapping[str, int]) -> tuple[list[str], np.ndarray]:
    if sum(ref.values()) == 0 or sum(cur.values()) == 0:
        raise ValueError("both windows need at least one observation")
    keys = sorted(set(ref) | set(cur))
    return keys, np.array([[ref.get(k, 0) for k in keys], [cur.get(k, 0) for k in keys]], float)


def chi2_test(ref: Mapping[str, int], cur: Mapping[str, int], alpha: float = 0.01) -> TestResult:
    """Two-sample chi-square on the 2 x k table of counts, rare categories merged."""
    _, table = _aligned(ref, cur)
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    rare = (expected < MIN_EXPECTED).any(axis=0)
    if rare.any():  # one "other" column instead of several near-empty ones
        table = np.column_stack([table[:, ~rare], table[:, rare].sum(1)])
    table = table[:, table.sum(0) > 0]
    if table.shape[1] < 2:
        return TestResult(0.0, 1.0, False)  # one category left: nothing to compare
    res = stats.chi2_contingency(table, correction=False)
    return TestResult(float(res.statistic), float(res.pvalue), bool(res.pvalue < alpha))


def js_divergence(ref: Mapping[str, int], cur: Mapping[str, int]) -> float:
    """Jensen-Shannon divergence (base 2): 0 = same mix, 1 = nothing in common."""
    _, table = _aligned(ref, cur)
    p, q = table[0] / table[0].sum(), table[1] / table[1].sum()
    m = (p + q) / 2

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return (kl(p, m) + kl(q, m)) / 2


def _subsample(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return x[rng.choice(len(x), MMD_MAX_POINTS, replace=False)] if len(x) > MMD_MAX_POINTS else x


def mmd_test(ref: np.ndarray, cur: np.ndarray, n_perm: int = 2000, alpha: float = 0.01,
             seed: int = 0) -> TestResult:  # fmt: skip
    """Maximum mean discrepancy with an RBF kernel and a permutation p-value.

    The kernel matrix of the pooled points is computed once; each permutation only re-weights
    it (MMD^2 = w^T K w with weights +1/n_x and -1/n_y), done for many permutations at a time.
    The smallest possible p-value is 1 / (n_perm + 1), so n_perm must be large enough for alpha
    (found in review: 200 permutations could never reach a Bonferroni-corrected 0.0017)."""
    if 1 / (n_perm + 1) >= alpha:
        raise ValueError(f"{n_perm} permutations cannot give p < {alpha}; use more")
    rng = np.random.default_rng(seed)
    x, y = _subsample(np.asarray(ref, float), rng), _subsample(np.asarray(cur, float), rng)
    pooled = np.vstack([x, y])
    sq = np.sum(pooled**2, 1)
    d2 = np.clip(sq[:, None] + sq[None, :] - 2 * pooled @ pooled.T, 0, None)
    positive = d2[d2 > 0]
    gamma = 1.0 / float(np.median(positive)) if positive.size else 1.0  # median heuristic
    kernel = np.exp(-gamma * d2)
    labels = np.r_[np.full(len(x), 1 / len(x)), np.full(len(y), -1 / len(y))]
    observed = float(labels @ kernel @ labels)
    hits = 0
    for start in range(0, n_perm, 250):  # a block of permutations per matrix product
        block = np.stack([rng.permutation(labels) for _ in range(min(250, n_perm - start))], 1)
        hits += int(np.sum(np.einsum("ip,ip->p", block, kernel @ block) >= observed))
    p = (hits + 1) / (n_perm + 1)
    return TestResult(observed, p, bool(p < alpha))


def domain_classifier_auc(ref: np.ndarray, cur: np.ndarray, seed: int = 0) -> float:
    """Cross-validated AUC of a model telling the two windows apart (0.5 = it cannot)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.vstack([ref, cur])
    y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    folds = max(2, min(5, len(ref), len(cur)))  # small windows: fewer folds, never empty ones
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    auc = cross_val_score(model, x, y, cv=cv, scoring="roc_auc").mean()
    return float(auc) if not math.isnan(auc) else 0.5


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
