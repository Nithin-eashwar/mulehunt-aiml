"""Cached loaders for the dashboard. Everything degrades to toy data / risk only."""

from pathlib import Path

import pandas as pd
import streamlit as st

from mulehunt.b_params import HONEST_FRAMING


@st.cache_resource(show_spinner="Loading the transaction graph...")
def graph():
    from mulehunt.graph import TxGraph
    return TxGraph.load()


@st.cache_resource
def risk():
    from mulehunt.graph import load_risk_scores, placeholder_risk_scores
    g, kind = graph()
    if kind == "toy":
        return placeholder_risk_scores(g), "placeholder (toy data)"
    r = load_risk_scores()
    return (r, "models/risk_scores.csv") if r is not None else (placeholder_risk_scores(g), "placeholder")


@st.cache_resource
def bundle():
    from mulehunt.models import load_model
    try:
        return load_model(open("models/best_model.txt").read().strip())
    except FileNotFoundError:
        return None


@st.cache_resource
def features():
    path = Path("data/processed/features.parquet")
    return pd.read_parquet(path).set_index("account_id") if path.exists() else None


@st.cache_resource
def rings():
    from mulehunt.trace import load_rings
    g, kind = graph()
    return load_rings(g.data_dir)


@st.cache_resource
def demo_txns():
    from app.logic import demo_transactions
    pre = Path("build/b_app/demo_transactions.csv")
    if pre.exists():
        return pd.read_csv(pre)
    g, _ = graph()
    return demo_transactions(g, rings())


@st.cache_resource
def false_alarms():
    from mulehunt.adversarial import measure_false_alarms
    g, _ = graph()
    return measure_false_alarms(g.eval_tx)


@st.cache_resource
def text_model():
    from text.classifier import load_artifact
    return load_artifact()


def footer():
    st.divider()
    st.caption(HONEST_FRAMING + " Flags are prompts for human review, not automatic freezes.")


def data_banner():
    g, kind = graph()
    if kind == "toy":
        st.info("Real data not found, showing the 500-account toy data. Regenerate with "
                "`python -m simulator.generate --config simulator/config.yaml --out data/generated`.")
