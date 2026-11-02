"""CPU energy measurement with Intel RAPL (the counters pyJoules reads).

``EnergyMeter`` reads ``/sys/class/powercap/intel-rapl:*/energy_uj`` directly (same source
as ``pyJoules``' ``RaplPackageDomain``/``RaplCoreDomain``), handles counter wraparound with
``max_energy_range_uj`` and subtracts an idle baseline:

    E_net = E_measured - P_idle * duration.

Recent kernels make ``energy_uj`` root-readable only. Run experiments through
``experiments/run_with_energy.sh``, which makes the counters world-readable once per boot
(``sudo chmod -R a+r /sys/class/powercap/intel-rapl``) and runs Python as the normal user.
Without readable counters the meter still records wall/CPU time and reports energy as NaN
(``backend == "none"``), so experiments never fail because of permissions.

Caveats recorded with every result (:meth:`EnergyMeter.describe`): RAPL measures the whole
package (all processes), the update interval is about 1 ms, so energy of sub-millisecond
calls (policy inference, plan check) is measured by :func:`benchmark` over many repetitions.
"""

from __future__ import annotations

import contextlib
import dataclasses
import glob
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

RAPL_ROOT = Path("/sys/class/powercap")


@dataclasses.dataclass
class Measurement:
    wall_s: float = 0.0
    cpu_s: float = 0.0  # process CPU time
    energy_j: dict[str, float] = dataclasses.field(default_factory=dict)  # gross, per domain
    net_energy_j: dict[str, float] = dataclasses.field(default_factory=dict)  # idle subtracted

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        out = {f"{prefix}wall_s": self.wall_s, f"{prefix}cpu_s": self.cpu_s}
        for k, v in self.energy_j.items():
            out[f"{prefix}energy_{k}_j"] = v
        for k, v in self.net_energy_j.items():
            out[f"{prefix}net_energy_{k}_j"] = v
        return out


def _domains() -> dict[str, tuple[Path, int]]:
    """``{name: (energy_uj path, max range)}`` for readable package/core/dram/psys domains."""
    out = {}
    for d in sorted(glob.glob(str(RAPL_ROOT / "intel-rapl:*"))):
        d = Path(d)
        try:
            name = (d / "name").read_text().strip()
            rng = int((d / "max_energy_range_uj").read_text())
            int((d / "energy_uj").read_text())
        except (OSError, ValueError):
            continue
        out.setdefault(name, (d / "energy_uj", rng))
    return out


def rapl_readable() -> bool:
    return bool(_domains())


class EnergyMeter:
    """Context-manager meter: ``with meter.measure() as m: ...``."""

    def __init__(self, domains: tuple[str, ...] = ("package-0", "core", "psys"),
                 idle_seconds: float = 0.0):
        found = _domains()
        self.domains = {k: v for k, v in found.items() if k in domains}
        self.backend = "rapl" if self.domains else "none"
        self.idle_power_w: dict[str, float] = {}
        if idle_seconds > 0 and self.domains:
            self.calibrate_idle(idle_seconds)

    def _read(self) -> dict[str, int]:
        return {k: int(p.read_text()) for k, (p, _) in self.domains.items()}

    def _diff(self, a: dict[str, int], b: dict[str, int]) -> dict[str, float]:
        out = {}
        for k in a:
            d = b[k] - a[k]
            if d < 0:  # counter wrapped
                d += self.domains[k][1]
            out[k] = d * 1e-6
        return out

    def calibrate_idle(self, seconds: float = 2.0) -> dict[str, float]:
        """Average package power with this process sleeping (other load counts as idle)."""
        a, t0 = self._read(), time.perf_counter()
        time.sleep(seconds)
        b, t1 = self._read(), time.perf_counter()
        self.idle_power_w = {k: e / (t1 - t0) for k, e in self._diff(a, b).items()}
        return self.idle_power_w

    @contextlib.contextmanager
    def measure(self) -> Iterator[Measurement]:
        m = Measurement()
        a = self._read() if self.domains else {}
        c0, t0 = time.process_time(), time.perf_counter()
        try:
            yield m
        finally:
            t1, c1 = time.perf_counter(), time.process_time()
            m.wall_s, m.cpu_s = t1 - t0, c1 - c0
            if self.domains:
                m.energy_j = self._diff(a, self._read())
                m.net_energy_j = {k: e - self.idle_power_w.get(k, 0.0) * m.wall_s
                                  for k, e in m.energy_j.items()}
            else:
                m.energy_j = {"package-0": float("nan")}
                m.net_energy_j = {"package-0": float("nan")}

    def describe(self) -> dict[str, object]:
        cpu = "unknown"
        with contextlib.suppress(OSError):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        return {"backend": self.backend, "domains": sorted(self.domains),
                "idle_power_w": self.idle_power_w, "cpu": cpu, "kernel": os.uname().release,
                "note": "RAPL is package-wide; idle baseline subtracted in net_energy"}


def benchmark(fn: Callable[[], object], meter: EnergyMeter, repeats: int = 1000,
              warmup: int = 10) -> dict[str, float]:
    """Per-call wall time and energy of a short function, averaged over ``repeats``."""
    for _ in range(warmup):
        fn()
    with meter.measure() as m:
        for _ in range(repeats):
            fn()
    out = {"repeats": float(repeats), "wall_s_per_call": m.wall_s / repeats,
           "cpu_s_per_call": m.cpu_s / repeats}
    for k, v in m.net_energy_j.items():
        out[f"net_energy_{k}_j_per_call"] = v / repeats
    for k, v in m.energy_j.items():
        out[f"energy_{k}_j_per_call"] = v / repeats
    return out
