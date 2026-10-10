"""Drift statistics: each test is quiet on a control (same distribution) and fires on a shift."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from legalrag.monitoring.drift_stats import (
    StormGuard,
    chi2_test,
    domain_classifier_auc,
    js_divergence,
    ks_test,
    mmd_test,
    wasserstein_distance,
)


def normal(seed: int, mean: float = 0.0, n: int = 400) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, 1.0, n)


def test_ks_is_quiet_on_controls_and_fires_on_a_mean_shift():
    controls = [ks_test(normal(s), normal(100 + s)).drift for s in range(20)]
    assert sum(controls) <= 1  # alpha 0.01: at most a rare false alarm
    assert ks_test(normal(1), normal(2, mean=0.5)).drift


def test_wasserstein_is_in_reference_standard_deviations():
    assert wasserstein_distance(normal(1), normal(2)) < 0.15
    moved = wasserstein_distance(normal(1, n=5000), normal(2, mean=1.0, n=5000))
    assert moved == pytest.approx(1.0, abs=0.1)  # a 1-sigma shift reads as ~1


def test_chi2_compares_two_samples_and_tolerates_new_categories():
    ref = {"ar": 50, "en": 50}
    assert not chi2_test(ref, {"ar": 48, "en": 52}).drift
    shifted = chi2_test(ref, {"ar": 90, "en": 10})
    assert shifted.drift and shifted.p_value < 0.001
    assert chi2_test(ref, {"ar": 40, "en": 40, "fr": 20}).drift  # a language we never saw


def test_chi2_merges_rare_categories_instead_of_trusting_tiny_counts():
    # found in review: a book seen once in a small reference made one-off counts look decisive
    ref = {"a": 30, "b": 30, "rare": 1}
    cur = {"a": 30, "b": 29, "rare": 0, "new": 1}
    assert not chi2_test(ref, cur).drift


def test_empty_samples_fail_loudly():
    with pytest.raises(ValueError):
        js_divergence({}, {"a": 1})
    with pytest.raises(ValueError):
        chi2_test({"a": 3}, {})


def test_js_divergence_is_zero_for_the_same_mix_and_one_for_disjoint_mixes():
    assert js_divergence({"a": 1, "b": 1}, {"a": 5, "b": 5}) == pytest.approx(0.0, abs=1e-9)
    assert js_divergence({"a": 1}, {"b": 1}) == pytest.approx(1.0)


def test_mmd_can_reach_a_bonferroni_corrected_alpha():
    # found in review: 200 permutations cannot give p below 1/201 = 0.005 > 0.01/6
    rng = np.random.default_rng(0)
    ref, same = rng.normal(size=(150, 4)), rng.normal(size=(150, 4))
    moved = rng.normal(size=(150, 4)) + np.array([0.8, 0, 0, 0])
    alpha = 0.01 / 6
    assert not mmd_test(ref, same, alpha=alpha, seed=1).drift
    shifted = mmd_test(ref, moved, alpha=alpha, seed=1)
    assert shifted.drift and shifted.p_value < alpha


def test_mmd_refuses_too_few_permutations_for_its_alpha():
    x = np.zeros((10, 2))
    with pytest.raises(ValueError, match="permutations"):
        mmd_test(x, x + 1, n_perm=200, alpha=0.001)


def test_mmd_stays_fast_on_a_big_window_by_subsampling():
    rng = np.random.default_rng(0)
    result = mmd_test(rng.normal(size=(4000, 3)), rng.normal(size=(4000, 3)), seed=0)
    assert 0 <= result.p_value <= 1


def test_domain_classifier_auc_is_near_half_when_nothing_changed():
    rng = np.random.default_rng(0)
    ref, same = rng.normal(size=(200, 5)), rng.normal(size=(200, 5))
    assert domain_classifier_auc(ref, same, seed=0) < 0.6
    assert domain_classifier_auc(ref, same + 1.0, seed=0) > 0.8


NOW = datetime(2026, 10, 10, 12, tzinfo=UTC)


def test_storm_guard_needs_enough_samples_a_cooldown_and_a_daily_cap():
    guard = StormGuard(min_samples=50, cooldown=timedelta(hours=6), daily_cap=2)
    assert guard.allow(drifted=True, samples=20, now=NOW, history=[]) == (
        False,
        "too few samples (20 < 50)",
    )
    assert guard.allow(drifted=False, samples=500, now=NOW, history=[]) == (False, "no drift")
    assert guard.allow(drifted=True, samples=500, now=NOW, history=[]) == (True, "drift")
    recent = [NOW - timedelta(hours=1)]
    assert guard.allow(drifted=True, samples=500, now=NOW, history=recent)[1].startswith("cooldown")
    two_today = [NOW - timedelta(hours=10), NOW - timedelta(hours=8)]
    assert guard.allow(drifted=True, samples=500, now=NOW, history=two_today)[1].startswith(
        "daily cap"
    )
