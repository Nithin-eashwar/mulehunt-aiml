"""Page 2: trace a flagged victim payment through the mule network, live."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from app import data, style  # noqa: E402
from app.logic import run_trail, trail_html  # noqa: E402
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
c1, c2, c3 = st.columns([2.2, 1.3, 1.3], gap="medium")
choice = c1.selectbox("Flagged victim payment", victims.txn_id + "  ·  " + victims.kind)
algorithm = c2.segmented_control("Algorithm", ["BFS", "IDDFS", "A*"], default="BFS") or "BFS"
window = c3.segmented_control("Hop window", [12, 24, 48], default=P.HOP_WINDOW_H,
                              format_func=lambda h: f"{h} h") or P.HOP_WINDOW_H
reveal = st.toggle("Reveal ground truth: colour by true tier and mark mules the classifier missed")
txn = choice.split()[0]

with st.spinner(f"Running {algorithm}, budget {P.MAX_EXPANSIONS:,} expansions"):
    res = run_trail(g, txn, algorithm, window, risk)

m = st.columns(5)
m[0].metric("Cash-out reached", "yes" if res.found else "no")
m[1].metric("Hops", res.hops if res.found else "–")
m[2].metric("Nodes expanded", f"{res.nodes_expanded:,}")
m[3].metric("Peak stored", f"{res.peak_stored:,}")
m[4].metric("Time", f"{1000 * res.elapsed_s:.1f} ms")
if res.truncated:
    st.warning("Stopped at the expansion budget. That means 'truncated', not 'no trail'; try a shorter window.")
elif not res.found:
    st.info(f"No time-respecting trail reaches a known cash-out within {window} h. "
            "A mule held the money longer than that; widen the window.")

if reveal:
    st.markdown(style.legend({"victim": style.TIER["victim"], "tier-1 mule": style.TIER["mule_t1"],
                              "tier-2 mule": style.TIER["mule_t2"], "tier-3 mule": style.TIER["mule_t3"],
                              "cash-out": style.TIER["cashout"], "genuine": style.TIER[None]}), unsafe_allow_html=True)
else:
    st.markdown(style.legend({"risk below 0.25": "#8FA394", "around 0.5": "#E0A85E", "near 1": style.SIGNAL,
                              "no score": style.TIER[None]}), unsafe_allow_html=True)
components.html(trail_html(g, txn, res.path, risk, data.rings(), reveal, threshold, height="520px"), height=540)
st.caption("Dark edges carry the victim payment and the trail; pale edges are one hop of context.")
data.footer()
