"""Money-laundering topologies used by the scam scenarios.

Each function returns transaction rows (dicts with timestamp, sender_id,
receiver_id, amount, channel) and never moves money backwards in time: a mule
forwards money after the payment that funded it, keeping a 1-8% commission.
The names match the AMLSim patterns in the IBM AML benchmark (fan-out, fan-in,
scatter-gather, cycle) plus a plain chain.
"""

import numpy as np
import pandas as pd

MIN_HOP_SECONDS = 60


def hop_delay(rng: np.random.Generator, cfg: dict, size=None) -> np.ndarray:
    """Hours a mule waits before forwarding: lognormal, redrawn above the cap."""
    p = cfg["fraud"]["hop_delay"]
    n = 1 if size is None else size
    out = np.exp(rng.normal(np.log(p["median_h"]), p["sigma"], size=n))
    while (bad := out > p["cap_h"]).any():
        out[bad] = np.exp(rng.normal(np.log(p["median_h"]), p["sigma"], size=bad.sum()))
    out = np.maximum(out, MIN_HOP_SECONDS / 3600)
    return out if size is not None else out[:1]


def _after(t0: pd.Timestamp, rng, cfg, patient: bool = False) -> pd.Timestamp:
    """When a mule forwards money received at t0; patient mules wait 1-2 days (plan rev 4)."""
    hours = rng.uniform(*cfg["fraud"]["patient_delay_h"]) if patient else hop_delay(rng, cfg)[0]
    return (t0 + pd.Timedelta(hours=float(hours))).floor("s") + pd.Timedelta(seconds=1)


def _commission(rng, cfg) -> float:
    lo, hi = cfg["fraud"]["mule_commission"]
    return float(rng.uniform(lo, hi))


def _row(t, sender, receiver, amount, channel="P2P") -> dict:
    return {"timestamp": t, "sender_id": sender, "receiver_id": receiver,
            "amount": round(float(amount), 2), "channel": channel}


def fan_out(src, dsts, total, t0, rng, cfg, slow=frozenset()) -> list[dict]:
    """src received `total` at t0 and spreads it over dsts, each shortly after.
    Senders in `slow` are patient mules."""
    kept = total * _commission(rng, cfg)
    shares = rng.dirichlet(np.ones(len(dsts)) * 3)
    return [_row(_after(t0, rng, cfg, src in slow), src, d, np.floor(kept * s * 100) / 100)
            for d, s in zip(dsts, shares)]


def fan_in(srcs, dst, amounts, t0s, rng=None, cfg=None) -> list[dict]:
    """Many payers pay one account at the given times (victim payments)."""
    return [_row(t, s, dst, a) for s, a, t in zip(srcs, amounts, t0s)]


def chain(path, amount, t0, rng, cfg, slow=frozenset()) -> list[dict]:
    """path[0] received `amount` at t0 and the money hops along the path."""
    rows, t = [], t0
    for sender, receiver in zip(path, path[1:]):
        amount, t = amount * _commission(rng, cfg), _after(t, rng, cfg, sender in slow)
        rows.append(_row(t, sender, receiver, amount))
        amount = rows[-1]["amount"]
    return rows


def scatter_gather(src, middles, dst, total, t0, rng, cfg, slow=frozenset()) -> list[dict]:
    """src scatters `total` over middles, which each pass their part on to dst."""
    rows = fan_out(src, middles, total, t0, rng, cfg, slow)
    for r in list(rows):
        rows += chain([r["receiver_id"], dst], r["amount"], r["timestamp"], rng, cfg, slow)
    return rows


def cycle(nodes, amount, t0, rng, cfg, slow=frozenset()) -> list[dict]:
    """Money goes round the nodes and back to the first one."""
    return chain(list(nodes) + [nodes[0]], amount, t0, rng, cfg, slow)


def split_under_threshold(amount: float, thresholds: list[int], rng: np.random.Generator) -> list[float]:
    """Split a payment into parts just under the largest threshold it would cross
    (90-99.9% of it); the last part is the smaller remainder."""
    usable = [t for t in sorted(thresholds) if 0.90 * t <= amount]
    if not usable:
        return [round(float(amount), 2)]
    cap = usable[-1]
    parts, remaining = [], round(float(amount), 2)
    while remaining >= 0.90 * cap:
        part = round(min(float(rng.uniform(0.90, 0.999)) * cap, remaining), 2)
        if remaining - part < 0.01:
            part = remaining
        if part >= cap:
            part = round(0.999 * cap, 2)
        parts.append(part)
        remaining = round(remaining - part, 2)
    if remaining > 0:
        parts.append(remaining)
    return parts
