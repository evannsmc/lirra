"""YAML configuration files loaded into frozen dataclasses.

Configs live in ``configs/`` as plain YAML. Each experiment declares a frozen
dataclass tree; :func:`load_config` validates the YAML against it, applies
``dotted.key=value`` overrides (e.g. from the command line or a SLURM script),
and returns the typed object. :func:`save_config` writes the *resolved* config
so every run directory records exactly what was executed.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return {} if data is None else dict(data)


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML scalars/lists."""
    out = _deepcopy_dict(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"cannot descend into non-mapping at {part!r} in {key!r}")
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def from_dict[T](cls: type[T], data: Mapping[str, Any]) -> T:
    """Recursively build dataclass ``cls`` from a mapping, rejecting unknown keys."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name])
    return cls(**kwargs)


def to_dict(obj: Any) -> Any:
    """Convert dataclasses/tuples/numpy scalars into YAML-safe builtins."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if hasattr(obj, "tolist"):  # numpy arrays and scalars
        return obj.tolist()
    return obj


def load_config[T](cls: type[T], path: str | Path, overrides: Sequence[str] = ()) -> T:
    return from_dict(cls, apply_overrides(load_yaml(path), overrides))


def save_config(obj: Any, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(obj), f, sort_keys=False)


def _coerce(tp: Any, value: Any) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if value is None:
        return None
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _coerce(non_none[0], value)
        return value
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v) for v in value)
        if args:
            return tuple(_coerce(a, v) for a, v in zip(args, value, strict=True))
        return tuple(value)
    if origin is list:
        return [_coerce(args[0], v) for v in value] if args else list(value)
    if origin in (dict, Mapping):
        return dict(value)
    if tp is float and isinstance(value, (int, float)):
        return float(value)
    if tp is int and isinstance(value, bool):
        raise TypeError(f"expected int, got bool {value!r}")
    return value


def _deepcopy_dict(d: Mapping[str, Any]) -> dict[str, Any]:
    return {k: _deepcopy_dict(v) if isinstance(v, Mapping) else v for k, v in d.items()}
