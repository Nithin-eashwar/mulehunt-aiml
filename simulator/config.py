"""Load simulator/config.yaml.

Every number in the config is a leaf of the form
``{value: ..., source: "...", status: sourced|assumed}`` so the report can cite
where each assumption came from. ``load_config`` strips that wrapper and returns
plain values; ``unsourced_keys`` lists what is still an assumption.
"""

from pathlib import Path

import yaml

DEFAULT_PATH = Path("simulator/config.yaml")
LEAF_KEYS = {"value", "source", "status"}
STATUSES = {"sourced", "assumed"}


def load_raw_config(path: str | Path = DEFAULT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _leaves(node, prefix=""):
    if isinstance(node, dict) and "value" in node:
        yield prefix, node
    elif isinstance(node, dict):
        for key, child in node.items():
            yield from _leaves(child, f"{prefix}.{key}" if prefix else key)
    else:
        raise ValueError(f"{prefix}: expected a {{value, source, status}} leaf, got {node!r}")


def _unwrap(node, prefix=""):
    if "value" in node:
        if set(node) != LEAF_KEYS or node["status"] not in STATUSES:
            raise ValueError(f"{prefix}: leaf needs exactly value/source/status with status in {sorted(STATUSES)}")
        return node["value"]
    return {k: _unwrap(v, f"{prefix}.{k}" if prefix else k) for k, v in node.items()}


def load_config(path: str | Path = DEFAULT_PATH) -> dict:
    raw = load_raw_config(path)
    list(_leaves(raw))  # raises on any value that is not wrapped in a leaf
    return _unwrap(raw)


def unsourced_keys(raw: dict) -> list[str]:
    return [key for key, leaf in _leaves(raw) if leaf.get("status") == "assumed"]
