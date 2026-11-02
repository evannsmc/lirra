"""Error tubes for nominal tube MPC and interval-STL margins.

The vehicle flies ``u = u_hat_i + K (x - x_hat_i)`` around a nominal plan ``(x_hat, u_hat)``
of ``x+ = A x + B u + G w`` with ``w in W = [-w_bar, w_bar]``. The error
``e = x - x_hat`` then obeys ``e+ = A_K e + G w`` (``A_K = A + B K``) and, starting from
``e_0 = 0``, lies in

    Z(i) = sum_{l < i} A_K^l G W.

Only the interval hull is needed (NAHS Sec. 7.1, ``[x]_{t+i} = x_{t+i} + [Z](i)``):

    hull Z(i) = [-z(i), z(i)],   z(i) = sum_{l < i} |A_K^l G| w_bar,

and for an affine predicate ``a^T y - b`` with ``y = C x`` the interval lower bound is
``a^T C x_hat - b - |a^T C| z(i)``. Inputs are tightened by ``|K A_K^l G| w_bar`` summed the
same way. ``z(i)`` is monotone and, for Schur ``A_K``, converges; :attr:`Tube.z_inf` is an
outer bound of the limit (the geometric tail is added explicitly).

Hulls compose as ``z(a + b) = z(b) + |A_K^b| z(a)`` only as an outer bound; the code
always uses the exact sum ``z(offset + i)``, which is what a plan rooted at a nominal
state whose error tube already has age ``offset`` needs.

:class:`ConstantTube` is the NAHS hardware choice: a constant empirical box ``eps`` on the
output (no ``K``, no ``W``), stated as an assumption on the tracking error.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True, eq=False)
class Tube:
    """Interval hulls ``z(i)`` (state) and ``kz(i)`` (input) for ``i = 0 .. n_max``."""

    z: np.ndarray  # (n_max + 1, n)
    kz: np.ndarray  # (n_max + 1, m)
    z_inf: np.ndarray  # (n,) outer bound of lim z(i)
    kz_inf: np.ndarray  # (m,)
    w_bar: np.ndarray
    description: str = ""

    @property
    def n_max(self) -> int:
        return self.z.shape[0] - 1

    def state(self, i: int) -> np.ndarray:
        return self.z[i] if i <= self.n_max else self.z_inf

    def input(self, i: int) -> np.ndarray:
        return self.kz[i] if i <= self.n_max else self.kz_inf

    def output(self, C: np.ndarray, i: int) -> np.ndarray:
        """Hull of ``C Z(i)``."""
        return np.abs(C) @ self.state(i)

    def output_inf(self, C: np.ndarray) -> np.ndarray:
        return np.abs(C) @ self.z_inf


def disturbance_tube(A: np.ndarray, B: np.ndarray, K: np.ndarray, G: np.ndarray,
                     w_bar: np.ndarray, n_max: int = 200, tail_tol: float = 1e-10) -> Tube:
    """Tube of ``e+ = (A + B K) e + G w`` for ``|w| <= w_bar`` (componentwise)."""
    A_K = A + B @ K
    rho = max(abs(np.linalg.eigvals(A_K)))
    if rho >= 1.0:
        raise ValueError(f"A + B K is not Schur (spectral radius {rho:.4f})")
    w_bar = np.asarray(w_bar, dtype=float)
    n, m = A.shape[0], B.shape[1]
    z = np.zeros((n_max + 1, n))
    kz = np.zeros((n_max + 1, m))
    P = np.asarray(G, dtype=float).copy()  # A_K^l G
    for i in range(1, n_max + 1):
        z[i] = z[i - 1] + np.abs(P) @ w_bar
        kz[i] = kz[i - 1] + np.abs(K @ P) @ w_bar
        P = A_K @ P
    # Tail: |A_K^{n_max + j} G| w <= |A_K^j| |A_K^{n_max} G| w elementwise, so
    # sum_j (...) <= S * max(last) with S = sum_j ||A_K^j||_inf. With p such that
    # ||A_K^p|| < 1: S <= (sum_{j<p} ||A_K^j||) / (1 - ||A_K^p||).
    last = np.abs(P) @ w_bar
    tail = np.zeros(n)
    if np.max(last) > tail_tol:
        powers, Pj = [], np.eye(n)
        for _ in range(10 * n_max):
            powers.append(np.linalg.norm(Pj, np.inf))
            Pj = A_K @ Pj
            if np.linalg.norm(Pj, np.inf) < 1.0:
                break
        q = np.linalg.norm(Pj, np.inf)
        if q >= 1.0:
            raise ValueError("no contracting power of A_K found; tube tail unbounded")
        tail = np.full(n, np.max(last) * sum(powers) / (1.0 - q))
    tail_k = np.abs(K) @ tail
    return Tube(z=z, kz=kz, z_inf=z[-1] + tail, kz_inf=kz[-1] + tail_k, w_bar=w_bar,
                description=f"disturbance tube, |w|<={w_bar.tolist()}, rho(A_K)={rho:.4f}")


def constant_tube(n: int, m: int, eps_state: np.ndarray, n_max: int = 200) -> Tube:
    """NAHS-style constant error box ``eps_state`` for every predicted step ``i >= 1``.

    ``i = 0`` (the plan root) has zero error. No input tightening (``K`` unknown).
    """
    eps = np.asarray(eps_state, dtype=float).reshape(n)
    z = np.tile(eps, (n_max + 1, 1))
    z[0] = 0.0
    return Tube(z=z, kz=np.zeros((n_max + 1, m)), z_inf=eps.copy(), kz_inf=np.zeros(m),
                w_bar=np.zeros(0), description=f"constant tube eps={eps.tolist()}")


def zero_tube(n: int, m: int, n_max: int = 200) -> Tube:
    return constant_tube(n, m, np.zeros(n), n_max)
