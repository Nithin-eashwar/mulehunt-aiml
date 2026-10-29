"""Helpers shared by Person B's search tests (conftest itself is not importable)."""

import pandas as pd


def assert_time_respecting(path, window=pd.Timedelta(hours=48), start_time_ns=None):
    """Each hop leaves from the previous receiver, strictly later and within the window."""
    w = int(pd.Timedelta(window).value)
    prev_t = start_time_ns
    for i, e in enumerate(path):
        if i:
            assert e.sender == path[i - 1].receiver, f"hop {i} does not chain"
        if prev_t is not None:
            assert prev_t < e.time_ns <= prev_t + w, f"hop {i} breaks the time rule"
        prev_t = e.time_ns
