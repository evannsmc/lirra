"""Shared Gurobi environment and solver status enum."""

from __future__ import annotations

import enum

import gurobipy as gp

_ENV: gp.Env | None = None


def gurobi_env() -> gp.Env:
    """Process-wide silent Gurobi environment (avoids the licence banner per model)."""
    global _ENV
    if _ENV is None:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        _ENV = env
    return _ENV


class SynthesisStatus(enum.StrEnum):
    OPTIMAL = "optimal"
    SUBOPTIMAL = "suboptimal"  # feasible incumbent, optimality not proven (time limit / gap)
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"  # time limit without any feasible solution
    NUMERIC = "numeric"
    ERROR = "error"
