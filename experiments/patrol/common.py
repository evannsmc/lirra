"""Shared helpers of the patrol experiment scripts (dataset, BC, PPO, evaluation)."""

from __future__ import annotations

import numpy as np

from rl_stl.evaluation.systems import get_system
from rl_stl.policies.segment import LevelTables, make_level_tables
from rl_stl.stl.nahs import spec_dim


def load_system(name: str, spec: str, reference: str = "rerooted", nominal_predicates: bool = False):
    """``(system, [setup per level], LevelTables)``; ``reference``: mission reference (rerooted/clock);
    ``nominal_predicates``: ``_u`` spec ablation (plan and check with the exact nominal predicates)."""
    system = get_system(name, spec, nominal_predicates=nominal_predicates)
    setups = [system.build(i) for i in range(len(system.levels))]
    T = make_level_tables([s.problem for s in setups], setups[0].task, setups[0].task.waypoints(),
                          np.stack([system.descriptor(i) for i in range(len(system.levels))]), reference,
                          clock_input_scale(system))
    return system, setups, T


def clock_input_scale(system) -> float:
    """Input saturation of the clock reference, as a fraction of the input bounds. The blimp's PD
    patrol flies at its thrust bound; at full scale always-on synthesis falls 1.5 m (mean) behind
    the reference and reaches 2-3 waypoints in 120 s, at 0.8 it keeps pace (10-11). Cube specs
    (same criterion, 3 starts, always-on, eps = 0.10): at 0.8 the blimp visits 6 waypoints while the
    reference passes 10 (the z legs are slow: 0.3 m/s, weak downward thrust); at 0.5 it visits 6-8 of 8."""
    if system.name == "blimp":
        return 0.5 if spec_dim(system.spec) == 3 else 0.8
    return 1.0


def latency_default(system) -> int:
    """Worst-case solve latency ``d`` in planning steps used by the paper runs."""
    return {"di": 2, "x2": 1, "blimp": 1}[system.name]


def feedback(system) -> bool:
    """Planning-level feedback ``K (x - x_hat)`` only where the tube comes from ``K`` and ``W``."""
    return system.name == "di"


def level_tables_summary(T: LevelTables) -> dict:
    return {"reference": T.reference, "ref_input_scale": T.ref_input_scale, "obs_size": T.obs_size, "action_size": T.action_size, "h": T.h, "N": T.n_plan,
            "levels": T.w_levels.tolist()}
