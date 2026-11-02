"""JAX robustness compiler cross-checked against stlpy's reference semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from stlpy.benchmarks.common import inside_rectangle_formula, outside_rectangle_formula
from stlpy.STL import LinearPredicate

from rl_stl.stl.jax_robustness import (
    compile_robustness,
    compile_robustness_trace,
    stl_hinge_penalty,
)


def stlpy_rho(formula, y_td, t=0):
    return float(np.asarray(formula.robustness(np.ascontiguousarray(y_td.T), t)).reshape(-1)[0])


def formulas():
    d = 3
    goal = inside_rectangle_formula((0.5, 1.5, -0.5, 0.5), 0, 1, d)
    obstacle_free = outside_rectangle_formula((-0.2, 0.2, -0.2, 0.2), 0, 1, d)
    low = LinearPredicate([0, 0, -1], -1.0)  # y2 <= 1
    high = LinearPredicate([0, 0, 1], 0.0)  # y2 >= 0
    return {
        "predicate": low,
        "always": obstacle_free.always(0, 10),
        "eventually": goal.eventually(2, 12),
        "reach_avoid": obstacle_free.always(0, 15) & goal.eventually(0, 15),
        "nested_GF": goal.eventually(0, 3).always(0, 8),
        "nested_FG": (low & high).always(0, 4).eventually(1, 6),
        "until": high.until(goal, 0, 10),
        "or_of_and": (low & goal).eventually(0, 5) | obstacle_free.always(3, 9),
    }


@pytest.mark.parametrize("name", list(formulas()))
def test_matches_stlpy(name):
    formula = formulas()[name]
    fn = compile_robustness(formula)
    rng = np.random.default_rng(0)
    jfn = jax.jit(fn.__call__)
    for _ in range(20):
        y = rng.normal(scale=1.0, size=(fn.required_length + 2, 3))
        assert float(jfn(jnp.asarray(y))) == pytest.approx(stlpy_rho(formula, y), abs=1e-5)


def test_start_time_and_trace():
    formula = formulas()["reach_avoid"]
    rng = np.random.default_rng(1)
    y = rng.normal(size=(30, 3))
    starts = [0, 3, 7, 14]
    trace = compile_robustness_trace(formula, starts)
    out = np.asarray(trace(jnp.asarray(y)))
    for i, t in enumerate(starts):
        assert out[i] == pytest.approx(stlpy_rho(formula, y, t), abs=1e-5)
    assert float(compile_robustness(formula, 7)(jnp.asarray(y))) == pytest.approx(stlpy_rho(formula, y, 7), abs=1e-5)


def test_batched_signal_and_vmap():
    formula = formulas()["nested_GF"]
    fn = compile_robustness(formula)
    rng = np.random.default_rng(2)
    ys = rng.normal(size=(8, fn.required_length, 3))
    batched = np.asarray(fn(jnp.asarray(ys)))
    vmapped = np.asarray(jax.vmap(fn.__call__)(jnp.asarray(ys)))
    ref = [stlpy_rho(formula, y) for y in ys]
    np.testing.assert_allclose(batched, ref, atol=1e-5)
    np.testing.assert_allclose(vmapped, ref, atol=1e-5)


def test_short_signal_and_dimension_rejected():
    fn = compile_robustness(formulas()["always"])
    assert fn.required_length == 11
    with pytest.raises(ValueError):
        fn(jnp.zeros((10, 3)))
    with pytest.raises(ValueError):
        fn(jnp.zeros((11, 2)))


def test_hinge_penalty_is_capped():
    rho = jnp.array([-1.0, 0.0, 0.05, 0.2, 5.0])
    np.testing.assert_allclose(np.asarray(stl_hinge_penalty(rho, 0.1)), [1.1, 0.1, 0.05, 0.0, 0.0], atol=1e-7)


def test_differentiable():
    formula = formulas()["always"]
    fn = compile_robustness(formula)
    y = jnp.asarray(np.random.default_rng(3).normal(size=(11, 3)))
    g = jax.grad(lambda s: fn(s))(y)
    assert g.shape == y.shape and bool(jnp.isfinite(g).all())
