import dataclasses
import json

import pytest

from rl_stl.utils.config import apply_overrides, from_dict, load_config, save_config, to_dict
from rl_stl.utils.device import resolve_device
from rl_stl.utils.provenance import create_run_dir, write_summary


@dataclasses.dataclass(frozen=True)
class Inner:
    gain: float = 1.0
    bounds: tuple[float, ...] = (-1.0, 1.0)


@dataclasses.dataclass(frozen=True)
class Outer:
    name: str = "x"
    horizon: int = 10
    inner: Inner = Inner()
    note: str | None = None


def test_from_dict_nested_and_coercion():
    cfg = from_dict(Outer, {"horizon": 5, "inner": {"gain": 2, "bounds": [0, 3]}})
    assert cfg.horizon == 5
    assert cfg.inner.gain == 2.0 and isinstance(cfg.inner.gain, float)
    assert cfg.inner.bounds == (0.0, 3.0)


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        from_dict(Outer, {"horizn": 5})


def test_overrides_and_roundtrip(tmp_path):
    data = apply_overrides({"inner": {"gain": 1.0}}, ["inner.gain=4.5", "horizon=7", "note=hi"])
    cfg = from_dict(Outer, data)
    assert cfg.inner.gain == 4.5 and cfg.horizon == 7 and cfg.note == "hi"
    path = tmp_path / "c.yaml"
    save_config(cfg, path)
    assert load_config(Outer, path) == cfg


def test_run_dir_provenance(tmp_path):
    run_dir = create_run_dir("unit", Outer(), seed=3, root=tmp_path, tag="t")
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["seed"] == 3
    assert "commit" in meta["git"] and "stlpy" in meta["versions"]
    assert load_config(Outer, run_dir / "config.yaml") == Outer()
    write_summary(run_dir, {"x": to_dict(Inner())})
    assert json.loads((run_dir / "summary.json").read_text())["x"]["gain"] == 1.0


def test_device_resolution(monkeypatch):
    monkeypatch.setenv("RL_STL_DEVICE", "cpu")
    assert resolve_device() == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        resolve_device("tpu")
