"""Page 2: trace a flagged victim payment through the mule network, live."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st  # noqa: E402

from app import data, style  # noqa: E402
from app.logic import ring_of, run_trail, trail_rows, trail_svg  # noqa: E402
from mulehunt import b_params as P  # noqa: E402

style.header("Mule trail explorer",
             "Follow stolen money forward from a flagged victim payment. Each hop must happen after the money "
             "arrived and within the window, and the search stops at a known cash-out account.")
data.data_banner()
g, kind = data.graph()
risk, _ = data.risk()
bundle = data.bundle()
threshold = float(bundle["threshold"]) if bundle and kind == "real" else 0.3
demo = data.demo_txns()
victims = demo[demo.ring_id.notna()]
if victims.empty:
    st.markdown(style.empty("No flagged victim payments to trace",
                            "Run python -m app.precompute to build the demo list."), unsafe_allow_html=True)
    data.footer()
    st.stop()

c1, c2, c3 = st.columns([2.2, 1.3, 1.3], gap="medium", vertical_alignment="bottom")
choice = c1.selectbox("Flagged victim payment", victims.txn_id + "  " + victims.kind)
algorithm = c2.segmented_control("Algorithm", ["BFS", "IDDFS", "A*"], default="BFS") or "BFS"
window = c3.segmented_control("Hop window", [12, 24, 48], default=P.HOP_WINDOW_H,
                              format_func=lambda h: f"{h} h") or P.HOP_WINDOW_H
reveal = st.toggle("Reveal ground truth: colour by true tier and mark mules the classifier missed")
txn = choice.split()[0]

with st.spinner(f"Running {algorithm}, budget {P.MAX_EXPANSIONS:,} expansions"):
    res = run_trail(g, txn, algorithm, window, risk)

st.markdown(style.tally([
    ("Cash-out reached", "yes" if res.found else "no"), ("Hops", res.hops if res.found else "none"),
    ("Nodes expanded", f"{res.nodes_expanded:,}"), ("Peak stored", f"{res.peak_stored:,}"),
    ("Time", f"{1000 * res.elapsed_s:.1f} ms")]), unsafe_allow_html=True)
if res.truncated:
    st.warning("Stopped at the expansion budget. That means truncated, not no trail; try a shorter window.")
elif not res.found:
    st.info(f"No time-respecting trail reaches a known cash-out within {window} h. "
            "A mule held the money longer than that; widen the window.")

st.markdown("### The trail, hop by hop")
rows = trail_rows(g, txn, res.path)
rings = data.rings()
head = ["Step", "From", "To", "Amount", "When", "Waited"] + (["Receiver"] if reveal else [])
table = []
for r in rows:
    line = [r["hop"], r["from"], r["to"], f"Rs {r['amount']:,.0f}", f"{r['time']:%d %b %H:%M}",
            "" if r["waited_h"] is None else f"{r['waited_h']:.1f} h"]
    if reveal:
        _, tier = ring_of(r["to"], rings)
        p = risk.get(r["to"]) if risk else None
        missed = tier in ("mule_t1", "mule_t2", "mule_t3") and (p is None or p < threshold)
        line.append(style.TIER_LABEL.get(tier, "genuine") + (", classifier missed" if missed else ""))
    table.append(line)
st.markdown(style.ledger(head, table, num={3, 5}, open_rows={len(rows) - 1} if res.found else set()),
            unsafe_allow_html=True)

if reveal:
    st.markdown(style.legend([("victim", "tier-victim"), ("tier-1 mule", "tier-mule_t1"),
                              ("tier-2 mule", "tier-mule_t2"), ("tier-3 mule", "tier-mule_t3"),
                              ("cash-out", "tier-cashout"), ("genuine", "tier-genuine")]), unsafe_allow_html=True)
else:
    st.markdown(style.legend([("low risk", "risk-low"), ("around 0.5", "risk-mid"), ("near 1", "risk-high"),
                              ("no score", "tier-genuine")]), unsafe_allow_html=True)
st.markdown("<div class='mh-panel mh-scroll'>" + trail_svg(g, txn, res.path, risk, rings, reveal, threshold)
            + "</div>", unsafe_allow_html=True)
st.caption("The trail runs left to right; small dots are each account's other payees in the data. Hover a dot for its id and risk.")
data.footer()
