"""STL semantics tests: hand-computed robustness, cross-check with stlpy, online intervals."""

import numpy as np
import pytest
from stlpy.STL import LinearPredicate

from rl_stl.stl.robustness import (
    formula_horizon,
    lipschitz_sup_norm,
    robustness,
    robustness_interval,
    stlpy_robustness,
    top_level_intervals,
)
from rl_stl.stl.specs import ReachAvoidTrackScenario, inside_box, outside_box

P = LinearPredicate([1.0], 1.0)  # y >= 1   -> rho = y - 1
Q = LinearPredicate([-1.0], 0.0)  # y <= 0  -> rho = -y


def sig(*vals):
    return np.array(vals, dtype=float).reshape(-1, 1)


@pytest.mark.parametrize(
    ("formula", "signal", "expected"),
    [
        (P, sig(3.0), 2.0),
        (P.negation(), sig(3.0), -2.0),
        (P.always(0, 2), sig(3, 2, 5), 1.0),
        (P.eventually(0, 2), sig(3, 2, 5), 4.0),
        (P.always(1, 2), sig(-9, 2, 5), 1.0),
        (P.eventually(1, 2), sig(9, 2, 5), 4.0),
        (P.eventually(0, 1).always(0, 1), sig(0, 3, 1), 2.0),
        (P & Q, sig(2), -2.0),
        (P | Q, sig(2), 1.0),
        # p U_[0,2] q with p = [1,2,-2], q = [-2,-3,1]: max(-2, min(1,-3), min(1,2,1)) = 1
        (P.until(Q, 0, 2), sig(2, 3, -1), 1.0),
    ],
)
def test_hand_computed_robustness(formula, signal, expected):
    assert robustness(formula, signal) == pytest.approx(expected)
    assert stlpy_robustness(formula, signal) == pytest.approx(expected)


def test_boxes_are_metric():
    inside = inside_box((0, 2, 0, 3), 0, 1, 2, "b")
    assert robustness(inside, np.array([[1.0, 1.0]])) == pytest.approx(1.0)
    assert robustness(inside, np.array([[2.5, 1.0]])) == pytest.approx(-0.5)
    outside = outside_box((0, 2, 0, 3), 0, 1, 2, "o")
    assert robustness(outside, np.array([[3.0, 1.0]])) == pytest.approx(1.0)
    assert robustness(outside, np.array([[1.0, 1.5]])) == pytest.approx(-1.0)


def test_short_signal_rejected():
    with pytest.raises(ValueError):
        robustness(P.always(0, 3), sig(1, 2))


def test_scenario_matches_stlpy_on_random_signals():
    sc = ReachAvoidTrackScenario()
    spec = sc.specification()
    assert formula_horizon(spec) == sc.horizon
    rng = np.random.default_rng(0)
    for _ in range(25):
        y = rng.uniform([-1, -1, -5, -5], [11, 11, 5, 5], size=(sc.horizon + 1, 4))
        assert robustness(spec, y) == pytest.approx(stlpy_robustness(spec, y), abs=1e-12)
    assert lipschitz_sup_norm(spec) == pytest.approx(1.0)


def test_interval_degenerate_when_prefix_complete():
    f = P.eventually(0, 2)
    y = sig(0, 0.5, 3)
    lo, hi = robustness_interval(f, y, np.array([-1.0]), np.array([5.0]))
    assert lo == hi == pytest.approx(robustness(f, y))


def test_interval_hand_computed():
    lo_box, hi_box = np.array([-1.0]), np.array([5.0])
    # prefix y0 = 0: p(0) = -1, unknown p in [-2, 4]
    assert robustness_interval(P.eventually(0, 2), sig(0), lo_box, hi_box) == (-1.0, 4.0)
    assert robustness_interval(P.always(0, 2), sig(0), lo_box, hi_box) == (-2.0, -1.0)


def test_interval_is_sound_for_random_completions():
    sc = ReachAvoidTrackScenario(horizon=12, goal_deadline=10)
    spec = sc.specification()
    lo, hi = sc.state_bounds()
    rng = np.random.default_rng(1)
    for k in [0, 3, 7, 12]:
        prefix = rng.uniform(lo, hi, size=(k + 1, 4))
        ilo, ihi = robustness_interval(spec, prefix, lo, hi)
        for _ in range(50):
            tail = rng.uniform(lo, hi, size=(sc.horizon - k, 4))
            rho = robustness(spec, np.vstack([prefix, tail]))
            assert ilo - 1e-12 <= rho <= ihi + 1e-12


def test_unreached_eventuality_is_inconclusive_not_violated():
    """Not having reached the goal before the deadline is not yet a violation."""
    sc = ReachAvoidTrackScenario()
    spec = sc.specification()
    lo, hi = sc.state_bounds()
    k = sc.goal_deadline - 1
    prefix = np.tile([1.0, 1.0, 0.0, 0.0], (k + 1, 1))  # parked far from the goal
    parts = top_level_intervals(spec, prefix, lo, hi)
    assert parts["reach_goal"][1] > 0  # trace-only semantics: still satisfiable
    assert parts["always_safe"][0] > 0 or parts["always_safe"][1] > 0
    # ...even though reaching the goal from (1,1) in one step at |v|<=4 m/s is impossible.
    # After the deadline passes, the obligation is violated.
    prefix2 = np.tile([1.0, 1.0, 0.0, 0.0], (sc.goal_deadline + 1, 1))
    assert top_level_intervals(spec, prefix2, lo, hi)["reach_goal"][1] < 0


def test_future_margin_robustness():
    f = P.always(0, 2)  # values y - 1
    y = sig(1.2, 3, 5)  # [0.2, 2, 4]
    assert robustness(f, y) == pytest.approx(0.2)
    # observed sample 0 satisfied -> no longer caps the margin
    assert robustness(f, y, observed_until=0) == pytest.approx(2.0)
    assert robustness(f, y, observed_until=2) == np.inf
    assert robustness(f, sig(0.5, 3, 5), observed_until=0) == -np.inf
    g = P.eventually(0, 2)
    assert robustness(g, sig(1.1, -5, -5), observed_until=0) == np.inf  # discharged


def test_future_margin_sign_equivalence():
    sc = ReachAvoidTrackScenario(horizon=12, goal_deadline=10)
    spec = sc.specification()
    rng = np.random.default_rng(5)
    for _ in range(200):
        y = rng.uniform([-1, -1, -5, -5], [11, 11, 5, 5], size=(13, 4))
        k = int(rng.integers(-1, 13))
        rho = robustness(spec, y)
        rho_f = robustness(spec, y, observed_until=k if k >= 0 else None)
        assert (rho >= 0) == (rho_f >= 0)
        if rho >= 0:
            assert rho_f >= rho  # a satisfied observed predicate no longer caps the margin


def test_interval_with_observed_boundary():
    lo_box, hi_box = np.array([-1.0]), np.array([5.0])
    f = P.always(0, 3)
    # observed y0=1.5 (Boolean, satisfied), predicted y1=1.2 (value 0.2), tail unknown [-2, 4]
    lo, hi = robustness_interval(f, sig(1.5, 1.2), lo_box, hi_box, observed_until=0)
    assert (lo, hi) == pytest.approx((-2.0, 0.2))
