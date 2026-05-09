"""
configs/loader.py

Central configuration loader for the REAL_DATA PPA framework.
Reads configs/default.yaml and merges with environment variable overrides.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


_CONFIG_CACHE: dict | None = None


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR:-default} and ${VAR} patterns with env values."""
    def _replace(match):
        var_expr = match.group(1)
        if ":-" in var_expr:
            var_name, default = var_expr.split(":-", 1)
            return os.environ.get(var_name, default)
        return os.environ.get(var_expr, match.group(0))
    return re.sub(r'\$\{([^}]+)\}', _replace, value)


def _walk_resolve(obj):
    """Recursively resolve env vars in all string values."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_resolve(v) for v in obj]
    return obj


def load_config(config_path: str | Path | None = None) -> dict:
    """Load and cache the project configuration."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = Path(__file__).parent / "default.yaml"

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    _CONFIG_CACHE = _walk_resolve(raw)
    return _CONFIG_CACHE


def get_data_root() -> Path:
    """Return the resolved DATA_ROOT path."""
    cfg = load_config()
    return Path(cfg.get("data_root", "./data"))


def get_designs() -> list[str]:
    """Return the list of target designs."""
    cfg = load_config()
    return cfg.get("designs", [])


def get_clock_targets() -> list[float]:
    """Generate the clock target sweep list."""
    cfg = load_config()
    ct = cfg.get("clock_targets_ns", {})
    start = float(ct.get("start", 2.0))
    stop = float(ct.get("stop", 6.0))
    step = float(ct.get("step", 0.4))
    targets = []
    val = start
    while val <= stop + 1e-9:
        targets.append(round(val, 2))
        val += step
    return targets
