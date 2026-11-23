"""Page 3: the fraudster-vs-detector game with minimax and alpha-beta."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st  # noqa: E402

from app import data, style  # noqa: E402
from app.logic import game_run  # noqa: E402
from mulehunt import b_params as P  # noqa: E402

style.header("Evasion game",
             "Each day the bank commits to a rule, then the fraudster sees it and decides how much to move and in "
             "how many parts. Any fixed threshold gets split under; minimax finds the rule that is safest against "
             "the best possible reply.")
fa = data.false_alarms()
c1, c2, c3 = st.columns(3, gap="large")
X = c1.slider("Money to move", 50_000, 500_000, P.GAME_X, step=10_000, format="Rs %d")
R = c2.slider("Days", 1, 4, P.GAME_R)
lam = c3.select_slider("Cost of one false alarm", [0, 25, 100, 400], value=P.ALERT_COST,
                       format_func=lambda v: f"Rs {v}")
options = ["Minimax (worst-case safe)"] + [f"Always Rs {T:,} threshold{', velocity rule on' if v else ''}"
                                           for T in P.GAME_THRESHOLDS for v in (False, True)]
pick = st.selectbox("Bank's policy", options)
if pick.startswith("Minimax"):
    policy = "minimax"
else:
    T = int(pick.split("Rs ")[1].split()[0].replace(",", ""))
    policy = (T, pick.endswith("on"))
with st.spinner("Solving the game"):
    rounds, acc, counts = game_run(X, R, lam, policy, fa)

st.markdown(style.tally([
    ("Moved past the bank", f"Rs {acc['uncaught']:,.0f}"), ("Frozen", f"Rs {acc['caught']:,.0f}"),
    ("Spent on mules", f"Rs {acc['fees']:,.0f}"), ("Fraudster's payoff", f"Rs {acc['payoff']:,.0f}")]),
    unsafe_allow_html=True)

st.markdown("### The fraudster's best reply, day by day")
days = rounds[rounds["round"] != "frozen"]
table = [[int(d.round), f"Rs {d.threshold:,.0f}{' + velocity' if d.velocity else ''}",
          f"{d.fraction:.0%} of Rs {d.remaining_before:,.0f}", int(d.parts), f"Rs {d.moved / d.parts:,.0f}",
          f"Rs {d.caught:,.0f}", f"Rs {d.uncaught:,.0f}"] for d in days.itertuples()]
caught_days = {i for i, d in enumerate(days.itertuples()) if d.caught > 0}
st.markdown(style.ledger(["Day", "Bank's rule", "Moves", "Parts", "Part size", "Frozen", "Got through"], table,
                         num={3, 4, 5, 6}, open_rows=caught_days), unsafe_allow_html=True)
st.caption("Highlighted days are the ones where the bank froze money.")

if counts:
    st.markdown("### Game-tree nodes visited for the same answer")
    st.markdown(style.bars([("Plain minimax", counts["minimax"]), ("Alpha-beta", counts["alpha-beta"]),
                            ("Alpha-beta, ordered", counts["alpha-beta + ordering"])]), unsafe_allow_html=True)
    saving = 1 - counts["alpha-beta + ordering"] / counts["minimax"]
    st.caption(f"Alpha-beta with move ordering skips {saving:.0%} of the tree and returns the same rule and value.")
data.footer()
