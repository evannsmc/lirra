"""Discrete-time linear synthesis models ``f_syn`` and helpers.

The synthesis model must be explicit and versioned (model-fidelity
rule). :class:`DiscreteLinearModel` records the matrices, timestep, and a
version string, and converts to a ``stlpy`` :class:`~stlpy.systems.LinearSystem`.

Note: ``stlpy.systems.DoubleIntegrator`` uses a unit timestep with ``B = [0; I]``
(no ``dt^2/2`` position term). :func:`zoh_double_integrator` is the exact
zero-order-hold discretisation for an arbitrary ``dt`` and is what this project
uses.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import scipy.linalg
from stlpy.systems import LinearSystem


@dataclasses.dataclass(frozen=True, eq=False)
class DiscreteLinearModel:
    """``x+ = A x + B u``, ``y = C x + D u`` with timestep ``dt`` [s]."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    dt: float
    name: str
    version: str
    state_names: tuple[str, ...] = ()
    input_names: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> int:
        return self.B.shape[1]

    @property
    def p(self) -> int:
        return self.C.shape[0]

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A @ x + self.B @ u

    def output(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        y = self.C @ x
        if u is not None:
            y = y + self.D @ u
        return y

    def rollout(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """States ``(H+1, n)`` from ``x0`` under inputs ``U`` of shape ``(H, m)``."""
        X = np.empty((U.shape[0] + 1, self.n))
        X[0] = x0
        for t in range(U.shape[0]):
            X[t + 1] = self.A @ X[t] + self.B @ U[t]
        return X

    def to_stlpy(self) -> LinearSystem:
        return LinearSystem(self.A, self.B, self.C, self.D)

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "dt": self.dt,
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "C": self.C.tolist(),
            "D": self.D.tolist(),
            "state_names": list(self.state_names),
            "input_names": list(self.input_names),
        }


def zoh_double_integrator(dim: int, dt: float) -> DiscreteLinearModel:
    """Exact ZOH double integrator with state ``[p; v]`` and acceleration input.

    Output is the full state (``C = I``, ``D = 0``) so STL predicates can use
    position and velocity.
    """
    eye = np.eye(dim)
    zero = np.zeros((dim, dim))
    A = np.block([[eye, dt * eye], [zero, eye]])
    B = np.block([[0.5 * dt**2 * eye], [dt * eye]])
    C = np.eye(2 * dim)
    D = np.zeros((2 * dim, dim))
    axes = "xyz"[:dim] if dim <= 3 else [str(i) for i in range(dim)]
    return DiscreteLinearModel(
        A=A, B=B, C=C, D=D, dt=float(dt),
        name=f"double_integrator_{dim}d_zoh",
        version="di-zoh-v1",
        state_names=tuple([f"p{a}" for a in axes] + [f"v{a}" for a in axes]),
        input_names=tuple(f"a{a}" for a in axes),
    )


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Discrete LQR gain ``K`` for ``u = -K x``."""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
