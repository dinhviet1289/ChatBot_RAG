"""Figma Indigo design system + safe HTML renderers for the ELC3008 RAG chatbot.

Pure functions (clean_snippet, render_citation_box_html) are unit-testable
without a Streamlit runtime. inject_global_css() imports streamlit
lazily so importing this module never requires Streamlit to be installed.
"""

import html
import math
import re
from string import Template

PALETTE = {
    "canvas": "#0E1117",
    "sidebar": "#151922",
    "card": "#1C2230",
    "primary": "#5A5BE0",
    "primary_end": "#6A5AE0",
    "active_pill": "#252D46",
    "callout": "#1B243B",
    "ink": "#E6EDF3",
    "muted": "#A6B2C5",
    "border": "#4A566E",
    "link": "#B4BFFE",
    "success": "#3FB950",
    "warning": "#D29922",
    "danger": "#F85149",
    "info": "#58A6FF",
}

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
        snippet_html = (f'<details class="rag-cite-details">'
                        f'<summary>Trích đoạn</summary>'
                        f'<div class="rag-cite-snippet">{snippet}</div></details>'
                        if snippet else "")
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
        f'<div class="rag-citation-head" style="color:{PALETTE["link"]};'
        f'font-weight:600;font-size:0.82rem;letter-spacing:0.03em;margin-bottom:6px;">'
        f'NGUỒN TRÍCH DẪN ({len(sources)})</div>'
        f'<ol class="rag-cite-list" style="list-style:none;margin:0;padding:0;">'
        f'{"".join(items)}</ol></div>'
    )


_CSS_TEMPLATE = Template("""
:root { color-scheme: dark;
  --canvas:$canvas; --sidebar:$sidebar; --card:$card; --ink:$ink; --muted:$muted;
  --border:$border; --primary:$primary; --primary-end:$primary_end; --link:$link;
  --active-pill:$active_pill; --callout:$callout;
  --success:$success; --warning:$warning; --danger:$danger; --info:$info;
  --ease:cubic-bezier(0.22,1,0.36,1); --speed:180ms; }
html { scroll-behavior:smooth; }
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
a, a:visited { color:var(--link); text-underline-offset:3px;
  transition:color var(--speed) var(--ease); }
a:hover { color:var(--ink); }
::selection { background:var(--primary); color:#fff; }
@keyframes rag-rise { from { opacity:0; transform:translateY(10px); }
  to { opacity:1; transform:translateY(0); } }
@keyframes rag-fade { from { opacity:0; } to { opacity:1; } }
* { scrollbar-width:thin; scrollbar-color:var(--border) var(--sidebar); }
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-track { background:var(--sidebar); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:999px;
  border:2px solid var(--sidebar); }
::-webkit-scrollbar-thumb:hover { background:var(--primary); }
::-webkit-scrollbar-corner { background:var(--sidebar); }
[data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { color:var(--ink); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:var(--muted); }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
  background:var(--active-pill); color:var(--ink); border:1px solid var(--border);
  border-radius:999px; text-align:left; font-weight:500; line-height:1.35;
  transition:border-color var(--speed) var(--ease), color var(--speed) var(--ease),
    background var(--speed) var(--ease); }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
  border-color:var(--link); color:var(--link); background:var(--card); }
[data-testid="stChatMessage"] { animation:rag-rise 280ms var(--ease) backwards;
  background:var(--card); border:1px solid var(--border);
  border-radius:14px; padding:1rem 1.25rem; box-shadow:0 1px 2px rgba(15,23,42,0.04); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background:var(--active-pill); border-color:var(--border); }
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--card);
  border:1px solid var(--border); border-radius:14px;
  transition:border-color var(--speed) var(--ease), box-shadow var(--speed) var(--ease); }
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color:color-mix(in srgb, var(--link) 55%, var(--border));
  box-shadow:0 4px 16px rgba(0,0,0,0.25); }
[data-testid="stBaseButton-secondary"] { background:var(--card); color:var(--ink);
  border:1px solid var(--border); border-radius:10px;
  transition:border-color var(--speed) var(--ease), color var(--speed) var(--ease),
    transform var(--speed) var(--ease), box-shadow var(--speed) var(--ease); }
[data-testid="stBaseButton-secondary"]:hover { border-color:var(--link); color:var(--link); }
[data-testid="stBaseButton-primary"] {
  background:linear-gradient(135deg,var(--primary),var(--primary-end),var(--primary));
  background-size:200% 200%; background-position:0% 50%;
  border:none; color:#fff; border-radius:10px; font-weight:600;
  transition:transform var(--speed) var(--ease), box-shadow var(--speed) var(--ease),
    filter var(--speed) var(--ease), background-position 400ms var(--ease); }
[data-testid="stBaseButton-primary"]:hover { transform:translateY(-1px); filter:brightness(1.06);
  background-position:100% 50%;
  box-shadow:0 6px 18px color-mix(in srgb, var(--primary) 45%, transparent); }
[data-testid="stBaseButton-primary"]:active,
[data-testid="stBaseButton-secondary"]:active { transform:translateY(0) scale(0.98); }
button:focus-visible, a:focus-visible, summary:focus-visible, input:focus-visible {
  outline:2px solid var(--link); outline-offset:2px; }
[data-testid="stTextInput"] input { background:var(--card); color:var(--ink);
  border:1px solid var(--border); border-radius:10px;
  transition:border-color var(--speed) var(--ease), box-shadow var(--speed) var(--ease); }
[data-testid="stTextInput"] input:focus { border-color:var(--primary);
  box-shadow:0 0 0 3px color-mix(in srgb, var(--primary) 30%, transparent); }
[data-testid="stChatInput"] { background:var(--card); border:1px solid var(--border);
  border-radius:14px;
  transition:border-color var(--speed) var(--ease), box-shadow var(--speed) var(--ease); }
[data-testid="stChatInput"]:focus-within { border-color:var(--primary);
  box-shadow:0 0 0 3px color-mix(in srgb, var(--primary) 25%, transparent); }
[data-testid="stChatInput"] textarea, [data-testid="stChatInputTextArea"] { color:var(--ink); }
[data-testid="stCheckbox"] input { accent-color:var(--primary); }
[data-testid="stToggle"] input { accent-color:var(--primary); }
[data-testid="stSpinner"] { color:var(--link); }
[data-testid="stMetric"] { animation:rag-rise 240ms var(--ease) backwards; }
[data-testid="stPills"] button { border-radius:999px;
  transition:background var(--speed) var(--ease), color var(--speed) var(--ease),
    border-color var(--speed) var(--ease); }
[data-testid="stMetricValue"] { font-variant-numeric:tabular-nums; letter-spacing:-0.01em; }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stAlert"] [kind="error"] { border-left:3px solid var(--danger); }
[data-testid="stAlert"] [kind="warning"] { border-left:3px solid var(--warning); }
[data-testid="stAlert"] [kind="success"] { border-left:3px solid var(--success); }
[data-testid="stAlert"] [kind="info"] { border-left:3px solid var(--info); }
[data-testid="stExpander"] details { border:1px solid var(--border); border-radius:12px;
  background:var(--card); }
[data-testid="stExpander"] summary { transition:color var(--speed) var(--ease); }
[data-testid="stExpander"] summary:hover { color:var(--link); }
[data-testid="stExpander"] [data-testid="stVerticalBlock"],
[data-testid="stExpanderDetails"] [data-testid="stVerticalBlock"] {
  animation:rag-fade 220ms var(--ease) backwards; }
.rag-citation { animation:rag-rise 240ms var(--ease) backwards; }
[data-testid="stDivider"] hr { border-color:var(--border); }
.rag-cite-item { padding:6px 0; border-bottom:1px dashed var(--border); font-size:0.9rem; }
.rag-cite-item:last-child { border-bottom:none; }
.rag-cite-arrow { color:var(--link); margin-right:6px; }
.rag-cite-num { color:var(--muted); font-weight:600; }
.rag-cite-title a { color:var(--link); font-weight:600; text-decoration:none; }
.rag-cite-title a:hover { text-decoration:underline; }
.rag-cite-id { color:var(--muted); font-size:0.78rem; }
.rag-cite-page { color:var(--muted); font-size:0.8rem; margin-left:4px; }
.rag-cite-score { background:var(--active-pill); color:var(--link);
  border-radius:999px; padding:1px 8px; font-size:0.74rem; font-weight:600; margin-left:6px; }
.rag-cite-details { margin-top:4px; }
.rag-cite-details > summary { color:var(--muted); font-size:0.78rem; cursor:pointer;
  list-style:none; display:inline-block; transition:color var(--speed) var(--ease); }
.rag-cite-details > summary::before { content:"▸ "; }
.rag-cite-details[open] > summary::before { content:"▾ "; }
.rag-cite-details > summary:hover { color:var(--ink); }
.rag-cite-details > summary:focus-visible { outline:2px solid var(--link); outline-offset:2px; }
.rag-cite-snippet { color:var(--muted); font-size:0.82rem; margin-top:4px; line-height:1.45;
  border-left:2px solid var(--border); padding-left:8px; }
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior:auto; }
  *, *::before, *::after { transition-duration:0.01ms !important;
    animation-duration:0.01ms !important; }
}
""")


def _global_css():
    return _CSS_TEMPLATE.substitute(**PALETTE)


def inject_global_css():
    import streamlit as st
    st.markdown(f"<style>{_global_css()}</style>", unsafe_allow_html=True)