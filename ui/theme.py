"""Figma Indigo design system + safe HTML renderers for the ELC3008 RAG chatbot.

Pure functions (clean_snippet, render_citation_box_html, render_pastel_bar_chart_html)
are unit-testable without a Streamlit runtime. inject_global_css() imports streamlit
lazily so importing this module never requires Streamlit to be installed.
"""

import html
import math
import re
from string import Template

PALETTE = {
    "canvas": "#F4F6FB",
    "sidebar": "#FFFFFF",
    "card": "#FFFFFF",
    "primary": "#5B6EF5",
    "primary_end": "#6A5AE0",
    "active_pill": "#EBF0FF",
    "callout": "#EEF3FF",
    "ink": "#1E293B",
    "muted": "#64748B",
    "border": "#E2E8F0",
}

PASTELS = ["#4FD1C5", "#F6AD55", "#63B3ED", "#9F7AEA", "#FC8181", "#68D391"]

_PAGE_COMMENT = re.compile(r"<!--\s*page:\s*\d+\s*-->")
_WHITESPACE = re.compile(r"\s+")


def clean_snippet(text, max_len=240):
    """Strip indexed `<!-- page: N -->` markers, collapse whitespace, truncate,
    then HTML-escape so raw chunk text can never break out of a custom card."""
    if not text:
        return ""
    text = _PAGE_COMMENT.sub(" ", str(text))
    text = _WHITESPACE.sub(" ", text).strip()
    truncated = len(text) > max_len
    if truncated:
        text = text[:max_len].rstrip()
    escaped = html.escape(text)
    return (escaped + "…") if truncated else escaped


def _fmt_score(score):
    if isinstance(score, bool):
        return None
    if isinstance(score, (int, float)) and math.isfinite(score):
        return f"{float(score):.3f}"
    return None


def render_citation_box_html(sources):
    """Figma `#EEF3FF` callout listing cited sources. Only `https://` URLs become
    links; every text field is escaped. Missing url/score/page degrade gracefully."""
    if not sources:
        return ""
    items = []
    for i, s in enumerate(sources, 1):
        doc_id = str(s.get("doc_id") or "")
        title = str(s.get("doc_title") or doc_id or "Tài liệu")
        title_esc = html.escape(title)
        doc_id_esc = html.escape(doc_id)
        trang = str(s.get("trang") or "").strip()
        url = s.get("url")
        if isinstance(url, str) and url.startswith("https://"):
            url_esc = html.escape(url, quote=True)
            label = (f'<a href="{url_esc}" target="_blank" '
                     f'rel="noopener noreferrer">{title_esc}</a>')
        else:
            label = title_esc
        score = _fmt_score(s.get("rerank_score"))
        score_html = f'<span class="rag-cite-score">↗ {score}</span>' if score else ""
        page_html = (f'<span class="rag-cite-page">· Trang {html.escape(trang)}</span>'
                     if trang else "")
        snippet = clean_snippet(s.get("snippet") or "", max_len=200)
        snippet_html = f'<div class="rag-cite-snippet">{snippet}</div>' if snippet else ""
        items.append(
            f'<li class="rag-cite-item">'
            f'<span class="rag-cite-arrow">→</span>'
            f'<span class="rag-cite-num">{i}.</span> '
            f'<span class="rag-cite-title">{label}</span> '
            f'<span class="rag-cite-id">({doc_id_esc})</span>'
            f'{page_html}{score_html}{snippet_html}'
            f'</li>'
        )
    return (
        f'<div class="rag-citation" style="background:{PALETTE["callout"]};'
        f'border:1px solid {PALETTE["border"]};border-radius:12px;'
        f'padding:14px 16px;margin-top:10px;">'
        f'<div class="rag-citation-head" style="color:{PALETTE["primary"]};'
        f'font-weight:600;font-size:0.82rem;letter-spacing:0.03em;margin-bottom:6px;">'
        f'NGUỒN TRÍCH DẪN ({len(sources)})</div>'
        f'<ol class="rag-cite-list" style="list-style:none;margin:0;padding:0;">'
        f'{"".join(items)}</ol></div>'
    )


def _metric_pct(val):
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)) and math.isfinite(val):
        return float(val) * 100.0
    return None


def render_pastel_bar_chart_html(runs, thresholds=None, metric_keys=None):
    """Pastel pill-topped columns comparing one or more eval runs per metric.
    NaN/None/non-numeric metrics render `N/A` with a 0% bar instead of raising."""
    if not runs:
        return ""
    run_ids = list(runs.keys())
    if metric_keys is None:
        metric_keys = []
        for rid in run_ids:
            for k in (runs[rid].get("metrics") or {}):
                if k not in metric_keys:
                    metric_keys.append(k)
    thresholds = thresholds or {}
    groups = []
    for metric in metric_keys:
        bars = []
        for j, rid in enumerate(run_ids):
            run = runs[rid] or {}
            label = html.escape(str(run.get("label") or rid))
            pct = _metric_pct((run.get("metrics") or {}).get(metric))
            color = PASTELS[j % len(PASTELS)]
            if pct is None:
                value_label, height = "N/A", 0.0
            else:
                value_label, height = f"{pct:.1f}%", max(0.0, min(100.0, pct))
            bars.append(
                f'<div class="rag-bar-col" style="display:flex;flex-direction:column;'
                f'align-items:center;gap:6px;flex:1;">'
                f'<div class="rag-bar-value" style="font-size:0.8rem;font-weight:600;'
                f'color:{PALETTE["ink"]};">{value_label}</div>'
                f'<div class="rag-bar-track" style="width:38px;height:140px;'
                f'background:{PALETTE["active_pill"]};border-radius:10px;'
                f'display:flex;align-items:flex-end;overflow:hidden;">'
                f'<div class="rag-bar-fill" style="width:100%;height:{height:.1f}%;'
                f'background:{color};border-radius:10px 10px 0 0;"></div></div>'
                f'<div class="rag-bar-label" style="font-size:0.72rem;'
                f'color:{PALETTE["muted"]};text-align:center;">{label}</div></div>'
            )
        thr = _metric_pct(thresholds.get(metric))
        thr_html = (f'<div class="rag-bar-thr" style="font-size:0.7rem;'
                    f'color:{PALETTE["muted"]};text-align:center;margin-top:6px;">'
                    f'Ngưỡng {thr:.0f}%</div>') if thr is not None else ""
        metric_title = html.escape(str(metric).replace("_", " "))
        groups.append(
            f'<div class="rag-bar-group" style="flex:1;min-width:150px;">'
            f'<div class="rag-bar-metric" style="font-size:0.8rem;font-weight:600;'
            f'color:{PALETTE["ink"]};margin-bottom:8px;text-align:center;">{metric_title}</div>'
            f'<div class="rag-bar-row" style="display:flex;gap:10px;align-items:flex-end;'
            f'justify-content:center;">{"".join(bars)}</div>{thr_html}</div>'
        )
    return (
        f'<div class="rag-chart" style="display:flex;flex-wrap:wrap;gap:20px;'
        f'background:{PALETTE["card"]};border:1px solid {PALETTE["border"]};'
        f'border-radius:14px;padding:18px;">{"".join(groups)}</div>'
    )


_CSS_TEMPLATE = Template("""
:root { color-scheme: light;
  --canvas:$canvas; --sidebar:$sidebar; --card:$card; --ink:$ink; --muted:$muted;
  --border:$border; --primary:$primary; --primary-end:$primary_end;
  --active-pill:$active_pill; --callout:$callout; }
html, body, [data-testid="stApp"] { background:var(--canvas); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; }
[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
[data-testid="stToolbarActions"] { display:none; }
[data-testid="stMainBlockContainer"] { max-width:1160px; margin:0 auto; padding-top:2rem; }
h1 { font-weight:700; letter-spacing:-0.02em; color:var(--ink); }
h2, h3 { color:var(--ink); letter-spacing:-0.01em; }
[data-testid="stMarkdownContainer"] { color:var(--ink); }
[data-testid="stCaptionContainer"] { color:var(--muted); }
a, a:visited { color:var(--primary); }
[data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { color:var(--ink); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:var(--muted); }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
  background:var(--active-pill); color:var(--ink); border:1px solid var(--border);
  border-radius:999px; text-align:left; font-weight:500; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
  border-color:var(--primary); color:var(--primary); }
[data-testid="stChatMessage"] { background:var(--card); border:1px solid var(--border);
  border-radius:14px; padding:1rem 1.25rem; box-shadow:0 1px 2px rgba(15,23,42,0.04); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background:var(--active-pill); border-color:#C7D2FE; }
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--card);
  border:1px solid var(--border); border-radius:14px; }
[data-testid="stBaseButton-secondary"] { background:var(--card); color:var(--ink);
  border:1px solid var(--border); border-radius:10px; }
[data-testid="stBaseButton-primary"] {
  background:linear-gradient(135deg,var(--primary),var(--primary-end));
  border:none; color:#fff; border-radius:10px; font-weight:600; }
[data-testid="stChatInput"] { background:var(--card); border:1px solid var(--border);
  border-radius:14px; }
[data-testid="stChatInput"] textarea, [data-testid="stChatInputTextArea"] { color:var(--ink); }
.rag-cite-item { padding:6px 0; border-bottom:1px dashed var(--border); font-size:0.9rem; }
.rag-cite-item:last-child { border-bottom:none; }
.rag-cite-arrow { color:var(--primary); margin-right:6px; }
.rag-cite-num { color:var(--muted); font-weight:600; }
.rag-cite-title a { color:var(--primary); font-weight:600; text-decoration:none; }
.rag-cite-title a:hover { text-decoration:underline; }
.rag-cite-id { color:var(--muted); font-size:0.78rem; }
.rag-cite-page { color:var(--muted); font-size:0.8rem; margin-left:4px; }
.rag-cite-score { background:var(--active-pill); color:var(--primary);
  border-radius:999px; padding:1px 8px; font-size:0.74rem; font-weight:600; margin-left:6px; }
.rag-cite-snippet { color:var(--muted); font-size:0.82rem; margin-top:4px; line-height:1.45; }
""")


def _global_css():
    return _CSS_TEMPLATE.substitute(**PALETTE)


def inject_global_css():
    import streamlit as st
    st.markdown(f"<style>{_global_css()}</style>", unsafe_allow_html=True)