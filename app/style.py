"""Shared look for the dashboard (Hallmark redesign, theme Piste).

Every colour and face lives in app/assets/tokens.css; app/assets/app.css holds the
components. Python only emits HTML and SVG with classes; no colour is set from Python.
"""

import html
import re
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"
FONTS = "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@800&family=Barlow:wght@400;500&display=swap"

TIER_LABEL = {"victim": "victim", "fraudster": "fraudster", "mule_t1": "tier-1 mule", "mule_t2": "tier-2 mule",
              "mule_t3": "tier-3 mule", "cashout": "cash-out", None: "genuine"}


def _css() -> str:
    css = (ASSETS / "app.css").read_text(encoding="utf-8")
    for name in ("tokens.css", "base.css"):  # Streamlit serves no static folder here, so inline the imports
        css = css.replace(f'@import url("./{name}");', (ASSETS / name).read_text(encoding="utf-8"))
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return " ".join(css.split())  # one line: markdown ends an HTML block at a blank line


def setup(pages: list, current) -> None:
    """Stylesheet, fonts and the top bar: the wordmark and one link per page."""
    st.markdown(f'<link rel="stylesheet" href="{FONTS}"><style>{_css()}</style>', unsafe_allow_html=True)
    cols = st.columns([1.2] + [1] * len(pages), vertical_alignment="center")
    cols[0].markdown("<div class='mh-bar'><span class='mh-word'>Mule<span>Hunt</span></span></div>",
                     unsafe_allow_html=True)
    for i, (col, page) in enumerate(zip(cols[1:], pages)):
        with col.container(key="nav_current" if page.url_path == current.url_path else f"nav_{i}"):
            st.page_link(page, label=page.title)
    st.markdown("<div class='mh-rule'></div>", unsafe_allow_html=True)


def header(title: str, lede: str) -> None:
    st.markdown(f"<div class='mh-head'><h1>{html.escape(title)}</h1><p>{html.escape(lede)}</p></div>",
                unsafe_allow_html=True)


def flag_class(flagged: bool) -> str:
    return "is-flag" if flagged else "is-calm"


def verdict(text: str, flagged: bool, detail: str = "") -> str:
    return (f"<div class='mh-verdict'><span class='v {flag_class(flagged)}'>{html.escape(text)}</span>"
            f"<span class='n'>{html.escape(detail)}</span></div>")


def gauge(label: str, prob, threshold: float) -> str:
    if prob is None:
        return (f"<div class='mh-gauge'><div class='row'><b>{html.escape(label)}</b><span class='p'>no score</span>"
                "</div><div class='track'></div></div>")
    return (f"<div class='mh-gauge'><div class='row'><b>{html.escape(label)}</b><span class='p'>{prob:.3f}</span>"
            f"</div><div class='track'><div class='fill {flag_class(prob >= threshold)}' "
            f"style='width:{100 * prob:.1f}%'></div><div class='tick' style='left:calc({100 * threshold:.1f}% - 1px)'"
            f" title='threshold {threshold:.3f}'></div></div></div>")


def tally(items: list) -> str:
    """Figures in one ruled row, never stat tiles."""
    return "<dl class='mh-tally'>" + "".join(
        f"<div><dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd></div>" for k, v in items) + "</dl>"


def ledger(head: list, rows: list, num=frozenset(), open_rows=frozenset()) -> str:
    """A ruled table; `num` = right-aligned column indexes, `open_rows` = rows marked in apricot."""
    th = "".join(f"<th class='{'num' if i in num else ''}'>{html.escape(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        f"<tr class='{'is-open' if r in open_rows else ''}'>"
        + "".join(f"<td class='{'num' if i in num else ('id' if i == 0 else '')}'>{html.escape(str(c))}</td>"
                  for i, c in enumerate(row)) + "</tr>"
        for r, row in enumerate(rows))
    return f"<div class='mh-scroll'><table class='mh-ledger'><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


def reasons(explain, descriptions) -> str:
    """Top features: plain meaning, feature id, and a signed bar around zero. For tree
    models, + means higher than a typical account, - lower."""
    top = max((abs(e) for _, e in explain), default=1) or 1
    rows = []
    for f, e in explain:
        w = 48 * abs(e) / top
        side = f"left:50%;width:{w:.1f}%" if e >= 0 else f"left:{50 - w:.1f}%;width:{w:.1f}%"
        rows.append(f"<li><div class='txt'><span class='what'>{html.escape(descriptions.get(f, f))}</span>"
                    f"<span class='id'>{html.escape(f)}, {'above' if e >= 0 else 'below'} typical</span></div>"
                    f"<div class='bar'><span class='v'>{e:+.3f}</span><span class='axis'></span>"
                    f"<span class='b {flag_class(e >= 0)}' style='{side}'></span></div></li>")
    return "<ul class='mh-reasons'>" + "".join(rows) + "</ul>"


def bars(items: list) -> str:
    top = max((v for _, v in items), default=1) or 1
    return "<div class='mh-bars'>" + "".join(
        f"<div class='r'><span class='l'>{html.escape(k)}</span><span class='t'><span class='f' "
        f"style='width:{100 * v / top:.1f}%'></span></span><span class='n'>{v:,}</span></div>"
        for k, v in items) + "</div>"


def legend(items: list) -> str:
    """(label, css class) pairs; the classes are defined in app.css from tokens."""
    return "<div class='mh-legend'>" + "".join(
        f"<span><i class='{c}'></i>{html.escape(k)}</span>" for k, c in items) + "</div>"


def empty(title: str, body: str) -> str:
    return f"<div class='mh-empty'><b>{html.escape(title)}</b><br>{html.escape(body)}</div>"


def skeleton() -> str:
    return "".join(f"<div class='mh-skel' style='width:{w}%'></div>" for w in (92, 76, 58))
