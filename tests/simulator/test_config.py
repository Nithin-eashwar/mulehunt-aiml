import pytest

from simulator.config import load_config, load_raw_config, unsourced_keys

CONFIG = "simulator/config.yaml"


@pytest.fixture(scope="module")
def raw():
    return load_raw_config(CONFIG)


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


def walk_leaves(node, prefix=""):
    """Yield (dotted key, leaf) for every {value, source, status} leaf."""
    assert isinstance(node, dict), f"{prefix} is not a mapping or a leaf"
    if "value" in node:
        yield prefix, node
        return
    for key, child in node.items():
        yield from walk_leaves(child, f"{prefix}.{key}" if prefix else key)


def test_every_leaf_has_source(raw):
    leaves = list(walk_leaves(raw))
    assert len(leaves) > 20
    for key, leaf in leaves:
        assert set(leaf) == {"value", "source", "status"}, key
        assert leaf["status"] in {"sourced", "assumed"}, key
        assert str(leaf["source"]).strip(), key


def test_load_config_unwraps_values(cfg):
    assert cfg["window"]["days"] == 90
    assert cfg["window"]["start"] == "2025-08-13"
    assert cfg["population"]["n_accounts"] == 50000


@pytest.mark.parametrize("path", [
    ("population", "type_mix"), ("bank_mix",), ("state_mix",), ("channel_mix",),
])
def test_mixes_sum_to_one(cfg, path):
    node = cfg
    for key in path:
        node = node[key]
    assert sum(node.values()) == pytest.approx(1.0, abs=1e-9)


def test_hour_weights_24(cfg):
    weights = cfg["hour_of_day"]
    assert len(weights) == 24
    assert all(w >= 0 for w in weights) and sum(weights) > 0


def test_splits_cover_window(cfg):
    windows = sorted((cfg["splits"][name] for name in ("train", "val", "test")), key=lambda w: w[0])
    assert windows[0][0] == 1
    assert windows[-1][1] == cfg["window"]["days"]
    for (_, end), (start, _) in zip(windows, windows[1:]):
        assert start == end + 1


def test_split_shares_sum_to_one(cfg):
    assert sum(cfg["splits"]["shares"].values()) == pytest.approx(1.0)


def test_unsourced_keys_lists_assumed(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "a:\n  b: {value: 1, source: 'assumed: guess', status: assumed}\n"
        "  c: {value: 2, source: 'NPCI', status: sourced}\n"
    )
    assert unsourced_keys(load_raw_config(path)) == ["a.b"]


def test_leaf_with_bad_status_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("a: {value: 1, source: x, status: maybe}\n")
    with pytest.raises(ValueError, match="a"):
        load_config(path)
