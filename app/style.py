"""Shared look for the dashboard: an analyst's workbench. Paper background, ink text,
one signal colour (vermilion) reserved for risk; Plex Sans for UI, Plex Mono for figures."""

import html

import streamlit as st

INK = "#1B2333"
MUTED = "#5B6372"
PAPER = "#F7F5F0"
PANEL = "#FFFFFF"
RULE = "#DDD8CC"
SIGNAL = "#C2410C"   # risk / flagged
CALM = "#3F6B4E"     # below threshold
TIER = {"victim": "#8A8F98", "fraudster": "#7F1D1D", "mule_t1": "#C2410C", "mule_t2": "#D97706",
        "mule_t3": "#A16207", "cashout": "#1B2333", None: "#D6D2C8"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, .stApp, .stMarkdown, p, label, li, h1, h2, h3, button, input, textarea, select {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: 'Material Symbols Rounded' !important; }}
[data-testid="stSidebarContent"] {{ display: flex; flex-direction: column; }}
[data-testid="stSidebarUserContent"] {{ order: -1; padding-bottom: 0; }}
[data-testid="stSidebarHeader"] {{ order: -2; height: 1rem; }}
::selection {{ background: #F3D9C9; color: {INK}; }}
* {{ scrollbar-color: #C9C3B5 transparent; scrollbar-width: thin; }}
:focus-visible {{ outline: 2px solid {SIGNAL}; outline-offset: 2px; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.02em; }}
h1 {{ font-weight: 700; font-size: 2rem !important; margin-bottom: .25rem; }}
h3 {{ font-weight: 600; font-size: 1.05rem !important; margin-top: 1.6rem; }}
[data-testid="stSidebar"] {{ background: {INK}; }}
[data-testid="stSidebar"] * {{ color: #E7E3DA !important; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: rgba(255,255,255,.08); }}
[data-testid="stSidebarNav"] a:hover {{ background: rgba(255,255,255,.05); }}
[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums;
  font-size: 1.55rem; color: {INK}; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: .8rem; }}
[data-testid="stDataFrame"] {{ border: 1px solid {RULE}; border-radius: 10px; }}
.mh-lede {{ color: {MUTED}; max-width: 68ch; font-size: .98rem; line-height: 1.55; margin: 0 0 1.4rem; }}
.mh-mark {{ font-weight: 700; font-size: 1.15rem; letter-spacing: -0.01em; padding: .2rem 0 .1rem; }}
.mh-mark span {{ color: #F0A57E; }}
.mh-sub {{ font-size: .78rem; opacity: .7; line-height: 1.45; margin-bottom: 1.2rem; }}
.mh-panel {{ background: {PANEL}; border-radius: 12px; padding: 1.1rem 1.25rem;
  box-shadow: 0 1px 2px rgba(27,35,51,.06), 0 6px 18px rgba(27,35,51,.05); }}
.mh-verdict {{ display:flex; align-items:baseline; gap: 1rem; flex-wrap: wrap; }}
.mh-verdict .v {{ font-weight: 700; font-size: 1.35rem; letter-spacing: -0.015em; }}
.mh-verdict .n {{ font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; color: {MUTED}; font-size: .86rem; }}
.mh-id {{ font-family: 'IBM Plex Mono', monospace; font-size: .84em; }}
.mh-reasons {{ list-style:none; padding:0; margin:0; }}
.mh-reasons li {{ display:flex; align-items:center; gap:1rem; padding:.65rem 0; border-top:1px solid {RULE}; }}
.mh-reasons li:first-child {{ border-top:0; padding-top:0; }}
.mh-reasons li:last-child {{ padding-bottom:0; }}
.mh-reasons .txt {{ flex:1; min-width:0; }}
.mh-reasons .what {{ display:block; color:{INK}; font-size:.9rem; line-height:1.35; }}
.mh-reasons .id {{ display:block; font-family:'IBM Plex Mono', monospace; font-size:.72rem; color:{MUTED};
  margin-top:.15rem; overflow-wrap:anywhere; }}
.mh-reasons .bar {{ flex:0 0 96px; position:relative; height:26px; }}
.mh-reasons .bar .axis {{ position:absolute; left:50%; top:2px; bottom:2px; width:1px; background:{RULE}; }}
.mh-reasons .bar .b {{ position:absolute; top:13px; height:7px; border-radius:2px; }}
.mh-reasons .bar .v {{ position:absolute; top:0; left:0; right:0; text-align:center; line-height:1;
  font-family:'IBM Plex Mono', monospace; font-size:.68rem; color:{MUTED}; }}
.mh-gauge {{ margin: .35rem 0 .9rem; }}
.mh-gauge .row {{ display:flex; justify-content:space-between; gap:1rem; font-size:.85rem; color:{MUTED}; margin-bottom:.3rem; }}
.mh-gauge .row b {{ color:{INK}; font-weight:600; }}
.mh-gauge .row .p {{ font-family:'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; color:{INK}; }}
.mh-gauge .track {{ position:relative; height:8px; background:#ECE8DF; border-radius:4px; }}
.mh-gauge .fill {{ position:absolute; inset:0 auto 0 0; border-radius:4px; }}
.mh-gauge .tick {{ position:absolute; top:-4px; bottom:-4px; width:2px; background:{INK}; }}
.mh-tag {{ display:inline-block; font-size:.72rem; font-weight:600; padding:.1rem .5rem; border-radius:999px; }}
.mh-msg {{ background:{PANEL}; border-radius:12px; padding:1rem 1.2rem; line-height:1.6; max-width:72ch;
  box-shadow: 0 1px 2px rgba(27,35,51,.06), 0 6px 18px rgba(27,35,51,.05); }}
.mh-msg mark {{ background:#F9E0D2; color:{INK}; padding:0 .15em; border-radius:3px; }}
.mh-foot {{ color:{MUTED}; font-size:.78rem; border-top:1px solid {RULE}; margin-top:2.5rem; padding-top:.9rem; max-width:90ch; }}
.mh-legend {{ display:flex; gap:1rem; flex-wrap:wrap; font-size:.8rem; color:{MUTED}; margin:.4rem 0 1rem; }}
.mh-legend i {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:.35rem; vertical-align:-1px; }}
</style>
"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("<div class='mh-mark'>Mule<span>Hunt</span></div>"
                    "<div class='mh-sub'>Tracing UPI mule networks.<br>Analyst workbench.</div>",
                    unsafe_allow_html=True)


def header(title: str, lede: str):
    st.title(title)
    st.markdown(f"<p class='mh-lede'>{lede}</p>", unsafe_allow_html=True)


def gauge(label: str, prob, threshold: float) -> str:
    if prob is None:
        return (f"<div class='mh-gauge'><div class='row'><b>{html.escape(label)}</b><span class='p'>no score</span>"
                "</div><div class='track'></div></div>")
    colour = SIGNAL if prob >= threshold else CALM
    return (f"<div class='mh-gauge'><div class='row'><b>{html.escape(label)}</b>"
            f"<span class='p'>{prob:.3f}</span></div><div class='track'>"
            f"<div class='fill' style='width:{100 * prob:.1f}%;background:{colour}'></div>"
            f"<div class='tick' style='left:calc({100 * threshold:.1f}% - 1px)' title='threshold {threshold:.3f}'></div>"
            "</div></div>")


def tag(text: str, flagged: bool) -> str:
    bg, fg = ("#FBE3D6", SIGNAL) if flagged else ("#E3EDE5", CALM)
    return f"<span class='mh-tag' style='background:{bg};color:{fg}'>{html.escape(text)}</span>"


def reasons(explain, descriptions) -> str:
    """Top features as a list: plain meaning, feature id, and a signed bar around zero.
    For tree models, + means higher than a typical account, - lower."""
    top = max((abs(e) for _, e in explain), default=1) or 1
    rows = []
    for f, e in explain:
        w = 48 * abs(e) / top
        side = f"left:50%;width:{w:.1f}%" if e >= 0 else f"left:{50 - w:.1f}%;width:{w:.1f}%"
        colour = SIGNAL if e >= 0 else CALM
        rows.append(f"<li><div class='txt'><span class='what'>{html.escape(descriptions.get(f, f))}</span>"
                    f"<span class='id'>{html.escape(f)} · {'above' if e >= 0 else 'below'} typical</span></div>"
                    f"<div class='bar'><span class='v'>{e:+.3f}</span><span class='axis'></span>"
                    f"<span class='b' style='{side};background:{colour}'></span></div></li>")
    return "<ul class='mh-reasons'>" + "".join(rows) + "</ul>"


def legend(items: dict) -> str:
    return "<div class='mh-legend'>" + "".join(
        f"<span><i style='background:{c}'></i>{html.escape(k)}</span>" for k, c in items.items()) + "</div>"


def bars(df, x, y, colour=INK, horizontal=False, height=220, signed=False):
    """Altair bar chart in the house palette."""
    import altair as alt
    enc_x, enc_y = (alt.X(f"{y}:Q", title=None), alt.Y(f"{x}:N", sort="-x", title=None)) if horizontal \
        else (alt.X(f"{x}:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)), alt.Y(f"{y}:Q", title=None))
    col = alt.condition(f"datum.{y} >= 0", alt.value(SIGNAL), alt.value(CALM)) if signed else alt.value(colour)
    chart = alt.Chart(df).mark_bar(cornerRadius=2).encode(x=enc_x, y=enc_y, color=col,
                                                          tooltip=[x, alt.Tooltip(y, format=",.3f")])
    return chart.properties(height=height).configure_axis(
        labelFont="IBM Plex Sans", labelColor=MUTED, labelLimit=240, gridColor="#ECE8DF", domainColor=RULE, tickColor=RULE
    ).configure_view(strokeWidth=0)
