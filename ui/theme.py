"""Scientific Observatory design system & safe HTML renderers for ELC3008.

Pure functions (clean_snippet, render_citation_box_html, render_system_card_html,
render_turn_telemetry_html, render_retrieval_panel_html,
render_corpus_overview_html, render_doc_card_html) are unit-testable without a
Streamlit runtime. inject_global_css() imports streamlit lazily so importing
this module never requires Streamlit to be installed.
"""

import html
import math
import re
from string import Template

PALETTE = {
    "canvas": "#0B0F17",
    "sidebar": "#101622",
    "card": "#151D2C",
    "primary": "#6366F1",
    "primary_end": "#4F46E5",
    "active_pill": "#1E2942",
    "callout": "#182236",
    "ink": "#EAF0F8",
    "muted": "#94A3B8",
    "border": "#263248",
    "link": "#A5B4FC",
    "success": "#34D399",
    "warning": "#FBBF24",
    "danger": "#F87171",
    "info": "#38BDF8",
}

_CATEGORY_COLORS = (
    "#6366F1",
    "#38BDF8",
    "#34D399",
    "#A78BFA",
    "#FBBF24",
    "#F472B6",
    "#2DD4BF",
    "#94A3B8",
)

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


def _fmt_score(score, decimals=3):
    if isinstance(score, bool):
        return None
    if isinstance(score, (int, float)) and math.isfinite(score):
        return f"{float(score):.{decimals}f}"
    return None


def _score_pct(score):
    """Map a rerank relevance score or logit onto [0, 100] for visual meter width."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    val = float(score)
    if not math.isfinite(val):
        return None
    if 0.0 <= val <= 1.0:
        return round(val * 100.0, 1)
    clamped = max(-20.0, min(20.0, val))
    return round(100.0 / (1.0 + math.exp(-clamped)), 1)


def _safe_url_label(label_esc, url):
    if isinstance(url, str) and url.startswith("https://"):
        url_esc = html.escape(url, quote=True)
        return (
            f'<a href="{url_esc}" target="_blank" '
            f'rel="noopener noreferrer">{label_esc}</a>'
        )
    return label_esc


def _fmt_pages(page_min, page_max):
    if page_min is None and page_max is None:
        return "—"
    if page_min is None:
        return str(page_max)
    if page_max is None or page_min == page_max:
        return str(page_min)
    return f"{page_min}-{page_max}"


def render_citation_box_html(sources):
    """Render cited chunks with visual confidence meter and collapsible verbatim snippet."""
    if not sources:
        return ""
    items = []
    for i, s in enumerate(sources, 1):
        doc_id = str(s.get("doc_id") or "")
        title = str(s.get("doc_title") or doc_id or "Tài liệu")
        title_esc = html.escape(title)
        doc_id_esc = html.escape(doc_id)
        trang = str(s.get("trang") or "").strip()
        label = _safe_url_label(title_esc, s.get("url"))

        raw_score = s.get("rerank_score")
        score = _fmt_score(raw_score, decimals=3)
        pct = _score_pct(raw_score)
        meter_html = ""
        if pct is not None:
            meter_html = (
                f'<span class="rag-cite-meter" title="Độ liên quan">'
                f'<span class="rag-cite-fill" style="width:{pct:.1f}%;"></span>'
                f'</span>'
            )
        score_html = (
            f'<span class="rag-cite-score">↗ {score}</span>' if score else ""
        )
        page_html = (
            f'<span class="rag-cite-page">Trang {html.escape(trang)}</span>'
            if trang else ""
        )
        snippet = clean_snippet(s.get("snippet") or "", max_len=240)
        snippet_html = (
            f'<details class="rag-cite-details">'
            f'<summary>Trích đoạn</summary>'
            f'<div class="rag-cite-snippet">{snippet}</div></details>'
            if snippet else ""
        )
        items.append(
            f'<li class="rag-cite-item">'
            f'<div class="rag-cite-row">'
            f'<span class="rag-cite-num">[{i}]</span>'
            f'<span class="rag-cite-title">{label}</span>'
            f'<span class="rag-cite-id">{doc_id_esc}</span>'
            f'{page_html}{meter_html}{score_html}'
            f'</div>'
            f'{snippet_html}'
            f'</li>'
        )
    return (
        f'<div class="rag-citation" style="background:{PALETTE["callout"]};'
        f'border:1px solid {PALETTE["border"]};border-radius:12px;'
        f'padding:14px 16px;margin-top:12px;">'
        f'<div class="rag-citation-head" style="color:{PALETTE["link"]};'
        f'font-weight:600;font-size:0.78rem;letter-spacing:0.06em;margin-bottom:8px;">'
        f'<span class="rag-mono">NGUỒN TRÍCH DẪN · {len(sources)}</span></div>'
        f'<ol class="rag-cite-list" style="list-style:none;margin:0;padding:0;">'
        f'{"".join(items)}</ol></div>'
    )


def render_system_card_html(cfg):
    """Render sidebar RAG pipeline architecture console without ever leaking API keys."""
    cfg = cfg or {}
    provider = str(cfg.get("rerank_provider") or "off").strip().lower()
    rerank_active = provider not in ("off", "none", "")
    rerank_model = str(cfg.get("rerank_model") or "").strip()
    top_k = html.escape(str(cfg.get("top_k") or 5))
    retrieve_n = html.escape(str(cfg.get("retrieve_n") or 20))
    llm_model = html.escape(str(cfg.get("llm_model") or "—"))
    embed_raw = str(cfg.get("embed_model") or "MiniLM-L12-v2")
    embed_short = html.escape(embed_raw.split("/")[-1])
    key_count = len(cfg.get("groq_keys") or [])

    dot_cls = "rag-sys-dot is-on" if rerank_active else "rag-sys-dot is-off"
    if rerank_active:
        rerank_label = f'Rerank {html.escape(provider.upper())}'
        rerank_sub = html.escape(rerank_model or "active")
    else:
        rerank_label = "Rerank TẮT"
        rerank_sub = "Chỉ Stage-1 Hybrid"

    return (
        f'<div class="rag-sys-card">'
        f'<div class="rag-sys-head">'
        f'<span class="{dot_cls}"></span>'
        f'<span class="rag-sys-title">{rerank_label}</span>'
        f'</div>'
        f'<div class="rag-sys-sub rag-mono">{rerank_sub}</div>'
        f'<div class="rag-sys-grid">'
        f'<div class="rag-sys-row"><span>Stage-1</span><span class="rag-mono">BM25 0.3 + FAISS 0.7 · N={retrieve_n}</span></div>'
        f'<div class="rag-sys-row"><span>Stage-2</span><span class="rag-mono">Top-{top_k}</span></div>'
        f'<div class="rag-sys-row"><span>LLM</span><span class="rag-mono">{llm_model} ({key_count} key)</span></div>'
        f'<div class="rag-sys-row"><span>Embed</span><span class="rag-mono">384d · {embed_short}</span></div>'
        f'</div></div>'
    )


def render_turn_telemetry_html(telemetry):
    """Render compact per-turn scientific telemetry strip below an assistant response."""
    if not telemetry:
        return ""
    chips = []
    latency = telemetry.get("latency_s")
    if isinstance(latency, (int, float)) and math.isfinite(latency):
        chips.append(f'<span class="rag-tel-chip">⏱ {float(latency):.2f}s</span>')

    ret_ms = telemetry.get("retrieve_ms")
    if isinstance(ret_ms, (int, float)) and math.isfinite(ret_ms):
        chips.append(f'<span class="rag-tel-chip">Stage-1 {float(ret_ms):.0f}ms</span>')

    if telemetry.get("rerank_enabled"):
        rer_ms = telemetry.get("rerank_ms")
        if isinstance(rer_ms, (int, float)) and math.isfinite(rer_ms):
            chips.append(f'<span class="rag-tel-chip">Rerank {float(rer_ms):.0f}ms</span>')
    else:
        chips.append('<span class="rag-tel-chip">Rerank TẮT</span>')

    sel_cnt = telemetry.get("selected_count")
    tot_cnt = telemetry.get("total_candidates")
    if sel_cnt is not None and tot_cnt is not None:
        chips.append(f'<span class="rag-tel-chip">{int(sel_cnt)}/{int(tot_cnt)} chunks</span>')

    model = str(telemetry.get("model") or "").strip()
    if model:
        slot = str(telemetry.get("key_slot") or "").strip()
        slot_part = f" · key {html.escape(slot)}" if slot else ""
        chips.append(f'<span class="rag-tel-chip rag-tel-model">{html.escape(model)}{slot_part}</span>')

    if telemetry.get("fell_back"):
        cfg_model = html.escape(str(telemetry.get("configured_model") or "model chính"))
        chips.append(
            f'<span class="rag-tel-chip rag-tel-warn">'
            f'⇄ Dự phòng (hết hạn mức {cfg_model})</span>'
        )

    return f'<div class="rag-telemetry-strip rag-mono">{"".join(chips)}</div>'


def _delta_badge(delta):
    d = int(delta or 0)
    if d > 0:
        return f'<span class="rag-delta-up">▲ +{d}</span>'
    if d < 0:
        return f'<span class="rag-delta-down">▼ {d}</span>'
    return '<span class="rag-delta-flat">● 0</span>'


def render_retrieval_panel_html(res):
    """Render 2-stage retrieval diagnostics: latency split, Top-K cards, Stage-1 table."""
    if not res:
        return '<div class="rag-lab-empty">Chưa có dữ liệu truy hồi.</div>'

    ret_ms = float(res.get("retrieve_ms") or 0.0)
    rer_ms = float(res.get("rerank_ms") or 0.0)
    rer_on = bool(res.get("rerank_enabled"))
    candidates = list(res.get("candidates") or [])
    selected = list(res.get("selected") or [])
    total_cand = int(res.get("total_candidates") if res.get("total_candidates") is not None else len(candidates))

    sum_ms = ret_ms + (rer_ms if rer_on else 0.0)
    if sum_ms <= 0.0:
        ret_w, rer_w = (100.0, 0.0)
    elif not rer_on or rer_ms <= 0.0:
        ret_w, rer_w = (100.0, 0.0)
    else:
        ret_w = max(8.0, min(92.0, (ret_ms / sum_ms) * 100.0))
        rer_w = round(100.0 - ret_w, 1)
        ret_w = round(ret_w, 1)

    rerank_val = f"{rer_ms:.0f} ms" if rer_on else "TẮT"
    summary_html = (
        f'<div class="rag-lab-summary">'
        f'<div class="rag-lab-stat">'
        f'<div class="rag-lab-k">STAGE-1 HYBRID</div>'
        f'<div class="rag-lab-v rag-mono">{ret_ms:.0f} ms</div>'
        f'<div class="rag-lab-s">BM25 0.30 + FAISS 0.70</div>'
        f'</div>'
        f'<div class="rag-lab-stat">'
        f'<div class="rag-lab-k">STAGE-2 RERANK</div>'
        f'<div class="rag-lab-v rag-mono">{rerank_val}</div>'
        f'<div class="rag-lab-s">{"Cohere Cross-Encoder" if rer_on else "Bỏ qua rerank"}</div>'
        f'</div>'
        f'<div class="rag-lab-stat">'
        f'<div class="rag-lab-k">MẬT ĐỘ LỌC</div>'
        f'<div class="rag-lab-v rag-mono">{len(selected)} / {total_cand}</div>'
        f'<div class="rag-lab-s">Top-K giữ lại / ứng viên</div>'
        f'</div>'
        f'</div>'
        f'<div class="rag-latency-bar" title="Tỷ lệ thời gian Stage-1 vs Stage-2">'
        f'<span class="rag-lat-ret" style="width:{ret_w:.1f}%;"></span>'
        f'<span class="rag-lat-rer" style="width:{rer_w:.1f}%;"></span>'
        f'</div>'
    )

    if not selected:
        return (
            f'<div class="rag-lab-panel">{summary_html}'
            f'<div class="rag-lab-empty">Không tìm thấy ứng viên phù hợp cho truy vấn này.</div>'
            f'</div>'
        )

    topk_cards = []
    for pos, item in enumerate(selected, 1):
        final_rank = int(item.get("final_rank") or pos)
        init_rank = int(item.get("initial_rank") or pos)
        title_esc = html.escape(str(item.get("doc_title") or item.get("doc_id") or "Tài liệu"))
        doc_id_esc = html.escape(str(item.get("doc_id") or "?"))
        trang_esc = html.escape(str(item.get("trang") or "?"))
        label_html = _safe_url_label(title_esc, item.get("url"))
        delta_html = _delta_badge(item.get("rank_delta"))

        raw_score = item.get("rerank_score")
        score_txt = _fmt_score(raw_score, decimals=4)
        pct = _score_pct(raw_score)
        score_block = ""
        if score_txt is not None and pct is not None:
            score_block = (
                f'<span class="rag-rerank-score-wrap">'
                f'<span class="rag-cite-meter"><span class="rag-cite-fill" style="width:{pct:.1f}%;"></span></span>'
                f'<span class="rag-mono rag-rerank-val">{score_txt}</span>'
                f'</span>'
            )

        snip = clean_snippet(item.get("snippet") or "", max_len=220)
        snip_html = f'<div class="rag-rerank-snip">{snip}</div>' if snip else ""
        topk_cards.append(
            f'<div class="rag-rerank-card">'
            f'<div class="rag-rerank-head">'
            f'<span class="rag-rank-badge rag-mono">#{final_rank}</span>'
            f'<span class="rag-rerank-title">{label_html}</span>'
            f'<span class="rag-cite-id">{doc_id_esc}</span>'
            f'<span class="rag-cite-page">Trang {trang_esc}</span>'
            f'<span class="rag-rank-flow rag-mono">#{init_rank} → #{final_rank}</span>'
            f'{delta_html}{score_block}'
            f'</div>'
            f'{snip_html}'
            f'</div>'
        )

    selected_keys = {
        (it.get("doc_id"), it.get("trang"), it.get("initial_rank"))
        for it in selected
    }
    ordered_cands = sorted(candidates, key=lambda c: int(c.get("initial_rank") or 9999))
    rows_html = []
    for c in ordered_cands:
        init_r = int(c.get("initial_rank") or 0)
        fin_r = int(c.get("final_rank") or init_r)
        is_kept = (c.get("doc_id"), c.get("trang"), c.get("initial_rank")) in selected_keys
        row_cls = "rag-s1-kept" if is_kept else ""
        status_pill = (
            '<span class="rag-badge rag-badge-ok">TOP-K</span>'
            if is_kept else '<span class="rag-badge rag-badge-dim">—</span>'
        )
        c_title = html.escape(str(c.get("doc_title") or c.get("doc_id") or "?"))
        c_trang = html.escape(str(c.get("trang") or "?"))
        c_score = _fmt_score(c.get("rerank_score"), decimals=4) or "—"
        rows_html.append(
            f'<tr class="{row_cls}">'
            f'<td class="rag-mono">#{init_r}</td>'
            f'<td class="rag-mono">#{fin_r}</td>'
            f'<td class="rag-mono">{_delta_badge(c.get("rank_delta"))}</td>'
            f'<td class="rag-mono">{c_score}</td>'
            f'<td>{c_title}</td>'
            f'<td class="rag-mono">{c_trang}</td>'
            f'<td>{status_pill}</td>'
            f'</tr>'
        )

    stage1_html = ""
    if rows_html:
        stage1_html = (
            f'<details class="rag-stage1-details">'
            f'<summary>Toàn bộ {total_cand} ứng viên stage-1 (BM25 + FAISS)</summary>'
            f'<div class="rag-stage1-wrap">'
            f'<table class="rag-stage1-table">'
            f'<thead><tr>'
            f'<th>S1</th><th>S2</th><th>Δ Hạng</th><th>Điểm</th>'
            f'<th>Tài liệu</th><th>Trang</th><th>Chọn</th>'
            f'</tr></thead>'
            f'<tbody>{"".join(rows_html)}</tbody>'
            f'</table></div></details>'
        )

    return (
        f'<div class="rag-lab-panel">'
        f'{summary_html}'
        f'<div class="rag-lab-subhead">TOP-{len(selected)} SAU RERANK</div>'
        f'{"".join(topk_cards)}'
        f'{stage1_html}'
        f'</div>'
    )


def render_corpus_overview_html(rows, embed_dim=384,
                                 embed_model="paraphrase-multilingual-MiniLM-L12-v2"):
    """Render corpus-level KPI cards and 8-category chunk distribution bar."""
    rows = list(rows or [])
    embed_dim = int(embed_dim or 384)
    total_docs = len(rows)
    total_chunks = sum(int(r.get("chunk_count") or 0) for r in rows)
    en_docs = sum(1 for r in rows if r.get("lang") == "EN")
    vi_docs = total_docs - en_docs

    by_cat = {}
    for r in rows:
        cat = str(r.get("category") or "Khác")
        entry = by_cat.setdefault(cat, {"docs": 0, "chunks": 0})
        entry["docs"] += 1
        entry["chunks"] += int(r.get("chunk_count") or 0)

    try:
        from ui.catalog import CATEGORY_ORDER
        ordered_cats = [c for c in CATEGORY_ORDER if c in by_cat]
        ordered_cats += [c for c in by_cat if c not in ordered_cats]
    except Exception:
        ordered_cats = list(by_cat.keys())

    kpis_html = (
        f'<div class="rag-kpi-grid">'
        f'<div class="rag-kpi-card">'
        f'<div class="rag-kpi-lbl">HỌC LIỆU CHỈ MỤC</div>'
        f'<div class="rag-kpi-val rag-mono">{total_docs}</div>'
        f'<div class="rag-kpi-sub">{vi_docs} Tiếng Việt · {en_docs} English</div>'
        f'</div>'
        f'<div class="rag-kpi-card">'
        f'<div class="rag-kpi-lbl">TỔNG VECTOR CHUNKS</div>'
        f'<div class="rag-kpi-val rag-mono">{total_chunks:,}</div>'
        f'<div class="rag-kpi-sub">Chunk 1,500 ký tự · overlap 300</div>'
        f'</div>'
        f'<div class="rag-kpi-card">'
        f'<div class="rag-kpi-lbl">KHÔNG GIAN EMBEDDING</div>'
        f'<div class="rag-kpi-val rag-mono">{embed_dim}d</div>'
        f'<div class="rag-kpi-sub rag-mono">{html.escape(str(embed_model))}</div>'
        f'</div>'
        f'</div>'
    )

    segs = []
    legends = []
    for idx, cat in enumerate(ordered_cats):
        color = _CATEGORY_COLORS[idx % len(_CATEGORY_COLORS)]
        stats = by_cat[cat]
        pct = (stats["chunks"] / total_chunks * 100.0) if total_chunks > 0 else 0.0
        cat_esc = html.escape(cat)
        if pct > 0:
            segs.append(
                f'<span class="rag-dist-seg" '
                f'style="width:{pct:.2f}%;background:{color};" '
                f'title="{cat_esc}: {stats["docs"]} tài liệu · {stats["chunks"]:,} chunks ({pct:.1f}%)"></span>'
            )
        legends.append(
            f'<span class="rag-dist-leg">'
            f'<span class="rag-dist-dot" style="background:{color};"></span>'
            f'<span>{cat_esc}</span>'
            f'<span class="rag-mono rag-dist-num">{stats["docs"]} · {pct:.1f}%</span>'
            f'</span>'
        )

    dist_html = ""
    if ordered_cats:
        dist_html = (
            f'<div class="rag-dist-box">'
            f'<div class="rag-dist-head">'
            f'<span>PHÂN BỐ CHUNK THEO HỌC PHẦN</span>'
            f'<span class="rag-mono">{len(ordered_cats)} nhóm</span>'
            f'</div>'
            f'<div class="rag-dist-bar">{"".join(segs)}</div>'
            f'<div class="rag-dist-legend">{"".join(legends)}</div>'
            f'</div>'
        )

    return f'<div class="rag-corpus-overview">{kpis_html}{dist_html}</div>'


def render_doc_card_html(row, max_chunks, total_chunks):
    """Render metadata body and relative density bar inside a document card."""
    title_esc = html.escape(str(row.get("title") or row.get("doc_id") or "Tài liệu"))
    doc_id_esc = html.escape(str(row.get("doc_id") or ""))
    cat_esc = html.escape(str(row.get("category") or ""))
    lang = str(row.get("lang") or "VI").upper()
    lang_esc = html.escape(lang)
    lang_cls = "rag-badge rag-badge-en" if lang == "EN" else "rag-badge rag-badge-vi"

    has_url = isinstance(row.get("url"), str) and row["url"].startswith("https://")
    src_badge = (
        '<span class="rag-badge rag-badge-ok">PDF Drive</span>'
        if has_url else '<span class="rag-badge rag-badge-dim">Vector Index</span>'
    )

    chunks = int(row.get("chunk_count") or 0)
    pages = html.escape(_fmt_pages(row.get("page_min"), row.get("page_max")))
    share_pct = (chunks / total_chunks * 100.0) if total_chunks > 0 else 0.0
    rel_pct = (chunks / max_chunks * 100.0) if max_chunks > 0 else 0.0
    rel_pct = max(2.0, min(100.0, rel_pct)) if chunks > 0 else 0.0

    return (
        f'<div class="rag-doc-card">'
        f'<div class="rag-doc-badges">'
        f'<span class="rag-badge rag-badge-cat">{cat_esc}</span>'
        f'<span class="{lang_cls}">{lang_esc}</span>'
        f'{src_badge}'
        f'</div>'
        f'<div class="rag-doc-title">{title_esc}</div>'
        f'<div class="rag-doc-id rag-mono">{doc_id_esc}</div>'
        f'<div class="rag-doc-stats rag-mono">'
        f'<span>{chunks:,} chunks ({share_pct:.1f}%)</span>'
        f'<span>Trang {pages}</span>'
        f'</div>'
        f'<div class="rag-doc-bar"><span style="width:{rel_pct:.1f}%;"></span></div>'
        f'</div>'
    )


_CSS_TEMPLATE = Template("""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root { color-scheme: dark;
  --canvas:$canvas; --sidebar:$sidebar; --card:$card; --ink:$ink; --muted:$muted;
  --border:$border; --primary:$primary; --primary-end:$primary_end; --link:$link;
  --active-pill:$active_pill; --callout:$callout;
  --success:$success; --warning:$warning; --danger:$danger; --info:$info;
  --font-sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  --font-mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Consolas,monospace;
  --ease:cubic-bezier(0.16,1,0.3,1); --speed:170ms; }
html, body, [data-testid="stApp"] {
  background:radial-gradient(ellipse 85% 45% at 50% -10%, rgba(99,102,241,0.11), transparent 70%), var(--canvas);
  color:var(--ink); font-family:var(--font-sans); overflow-anchor:auto; }
[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
[data-testid="stToolbarActions"] { display:none; }
[data-testid="stMainBlockContainer"] { max-width:1120px; margin:0 auto; padding-top:1.75rem; }
[data-testid="stElementContainer"]:has(style) { margin:0 !important; padding:0 !important; }
h1, h2, h3 { color:var(--ink); letter-spacing:-0.02em; font-weight:700; }
[data-testid="stMarkdownContainer"] { color:var(--ink); line-height:1.62; }
[data-testid="stCaptionContainer"] { color:var(--muted); }
.rag-mono { font-family:var(--font-mono); font-variant-numeric:tabular-nums; }
a, a:visited { color:var(--link); text-underline-offset:3px;
  transition:color var(--speed) var(--ease); }
a:hover { color:var(--ink); }
::selection { background:var(--primary); color:#fff; }
@keyframes rag-rise { from { opacity:0; transform:translate3d(0,4px,0); }
  to { opacity:1; transform:translate3d(0,0,0); } }
@keyframes rag-fade { from { opacity:0; } to { opacity:1; } }
@keyframes rag-pulse { 0%, 100% { opacity:0.55; } 50% { opacity:1; } }
* { scrollbar-width:thin; scrollbar-color:var(--border) var(--sidebar); }
::-webkit-scrollbar { width:9px; height:9px; }
::-webkit-scrollbar-track { background:var(--sidebar); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:999px;
  border:2px solid var(--sidebar); }
::-webkit-scrollbar-thumb:hover { background:var(--primary); }
::-webkit-scrollbar-corner { background:var(--sidebar); }
[data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { color:var(--ink); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:var(--muted); }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
  background:rgba(30,41,66,0.55); color:var(--ink); border:1px solid var(--border);
  border-radius:10px; text-align:left; font-weight:500; line-height:1.35;
  transition:border-color var(--speed) var(--ease), color var(--speed) var(--ease),
    background-color var(--speed) var(--ease); }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
  border-color:var(--link); color:var(--link); background:var(--active-pill); }
.rag-brand { padding:4px 0 10px; }
.rag-brand-kicker { display:inline-block; font-family:var(--font-mono); font-size:0.68rem;
  font-weight:600; letter-spacing:0.08em; color:var(--link);
  background:rgba(99,102,241,0.14); border:1px solid rgba(99,102,241,0.35);
  border-radius:999px; padding:2px 9px; margin-bottom:6px; }
.rag-brand-title { font-size:1.25rem; font-weight:700; letter-spacing:-0.02em; line-height:1.2; }
.rag-brand-sub { color:var(--muted); font-size:0.78rem; margin-top:3px; }
.rag-sys-card { background:var(--card); border:1px solid var(--border); border-radius:12px;
  padding:11px 12px; }
.rag-sys-head { display:flex; align-items:center; gap:7px; font-size:0.82rem; font-weight:600; }
.rag-sys-dot { width:8px; height:8px; border-radius:50%; display:inline-block; flex-shrink:0; }
.rag-sys-dot.is-on { background:var(--success); box-shadow:0 0 8px rgba(52,211,153,0.6);
  animation:rag-pulse 2.2s ease-in-out infinite; }
.rag-sys-dot.is-off { background:var(--warning); }
.rag-sys-sub { color:var(--muted); font-size:0.72rem; margin:3px 0 8px 15px; }
.rag-sys-grid { display:flex; flex-direction:column; gap:4px; border-top:1px solid var(--border);
  padding-top:8px; }
.rag-sys-row { display:flex; justify-content:space-between; gap:8px; font-size:0.72rem;
  color:var(--muted); }
.rag-sys-row .rag-mono { color:var(--ink); text-align:right; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap; }
[data-testid="stChatMessage"] { animation:rag-rise 180ms var(--ease) backwards;
  background:var(--card); border:1px solid var(--border);
  border-radius:14px; padding:1rem 1.25rem; box-shadow:0 2px 10px rgba(0,0,0,0.18);
  will-change:opacity,transform; }
[data-testid="stChatMessage"]:not(:last-of-type) { animation:none; will-change:auto; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background:var(--active-pill); border-color:rgba(165,180,252,0.22); }
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--card);
  border:1px solid var(--border); border-radius:14px;
  transition:border-color var(--speed) var(--ease), box-shadow var(--speed) var(--ease); }
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color:color-mix(in srgb, var(--link) 50%, var(--border));
  box-shadow:0 6px 20px rgba(0,0,0,0.28); }
[data-testid="stBaseButton-secondary"] { background:var(--card); color:var(--ink);
  border:1px solid var(--border); border-radius:10px;
  transition:border-color var(--speed) var(--ease), color var(--speed) var(--ease),
    background-color var(--speed) var(--ease), box-shadow var(--speed) var(--ease); }
[data-testid="stBaseButton-secondary"]:hover { border-color:var(--link); color:var(--link);
  background:var(--active-pill); }
[data-testid="stBaseButton-primary"] {
  background:linear-gradient(135deg,var(--primary),var(--primary-end));
  border:1px solid rgba(165,180,252,0.28); color:#fff; border-radius:10px; font-weight:600;
  transition:transform var(--speed) var(--ease), box-shadow var(--speed) var(--ease),
    filter var(--speed) var(--ease); }
[data-testid="stBaseButton-primary"]:hover { transform:translate3d(0,-1px,0); filter:brightness(1.07);
  box-shadow:0 6px 18px color-mix(in srgb, var(--primary) 42%, transparent); }
[data-testid="stBaseButton-primary"]:active,
[data-testid="stBaseButton-secondary"]:active { transform:translate3d(0,0,0); filter:brightness(0.95); }
button:focus-visible, a:focus-visible, summary:focus-visible, input:focus-visible {
  outline:2px solid var(--link); outline-offset:2px; }
[data-testid="stTextInput"] input { background:var(--card); color:var(--ink);
  border:1px solid var(--border); border-radius:10px;
  transition:border-color var(--speed) var(--ease), box-shadow var(--speed) var(--ease); }
[data-testid="stTextInput"] input:focus { border-color:var(--primary);
  box-shadow:0 0 0 3px color-mix(in srgb, var(--primary) 28%, transparent); }
[data-testid="stChatInput"] { background:var(--card); border:1px solid var(--border);
  border-radius:14px;
  transition:border-color var(--speed) var(--ease), box-shadow var(--speed) var(--ease); }
[data-testid="stChatInput"]:focus-within { border-color:var(--primary);
  box-shadow:0 0 0 3px color-mix(in srgb, var(--primary) 24%, transparent); }
[data-testid="stChatInput"] textarea, [data-testid="stChatInputTextArea"] { color:var(--ink); }
[data-testid="stCheckbox"] input, [data-testid="stToggle"] input { accent-color:var(--primary); }
[data-testid="stSpinner"] { color:var(--link); }
[data-testid="stMetric"] { animation:rag-fade 180ms var(--ease) backwards; }
[data-testid="stPills"] button { border-radius:999px;
  transition:background-color var(--speed) var(--ease), color var(--speed) var(--ease),
    border-color var(--speed) var(--ease); }
[data-testid="stMetricValue"] { font-variant-numeric:tabular-nums; letter-spacing:-0.02em; }
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
  animation:rag-fade 180ms var(--ease) backwards; }
[data-testid="stDivider"] hr { border-color:var(--border); }
.rag-hero { text-align:center; padding:1.6rem 0 1.2rem; animation:rag-fade 220ms var(--ease) backwards; }
.rag-hero-badge { display:inline-block; font-family:var(--font-mono); font-size:0.72rem;
  font-weight:600; letter-spacing:0.08em; color:var(--link);
  background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.32);
  border-radius:999px; padding:4px 12px; margin-bottom:10px; }
.rag-hero-title { font-size:2rem; font-weight:700; letter-spacing:-0.025em; margin:0; }
.rag-hero-sub { color:var(--muted); margin-top:6px; font-size:0.95rem; }
.rag-hero-pills { display:flex; flex-wrap:wrap; justify-content:center; gap:8px; margin-top:14px; }
.rag-hero-pill { font-family:var(--font-mono); font-size:0.74rem; color:var(--ink);
  background:var(--card); border:1px solid var(--border); border-radius:999px; padding:4px 11px; }
.rag-sug-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:4px; }
.rag-sug-idx { font-family:var(--font-mono); font-size:0.72rem; color:var(--link);
  background:var(--active-pill); border-radius:6px; padding:2px 7px; }
.rag-stage-live { display:flex; align-items:center; gap:10px; height:38px;
  padding:0 12px; border-radius:10px; background:var(--callout); border:1px solid var(--border);
  color:var(--link); font-size:0.84rem; font-family:var(--font-mono); }
.rag-stage-dot { width:8px; height:8px; border-radius:50%; background:var(--info);
  animation:rag-pulse 1.2s ease-in-out infinite; flex-shrink:0; }
.rag-telemetry-strip { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
.rag-tel-chip { font-size:0.74rem; color:var(--muted); background:rgba(16,22,34,0.75);
  border:1px solid var(--border); border-radius:999px; padding:2px 9px; }
.rag-tel-model { color:var(--link); }
.rag-tel-warn { color:var(--warning); border-color:rgba(251,191,36,0.4); }
.rag-citation { animation:rag-fade 180ms var(--ease) backwards; }
.rag-cite-item { padding:8px 0; border-bottom:1px dashed var(--border); font-size:0.9rem; }
.rag-cite-item:last-child { border-bottom:none; padding-bottom:2px; }
.rag-cite-row { display:flex; flex-wrap:wrap; align-items:center; gap:6px; }
.rag-cite-num { font-family:var(--font-mono); color:var(--link); font-weight:600; font-size:0.8rem; }
.rag-cite-title, .rag-cite-title a { color:var(--link); font-weight:600; text-decoration:none; }
.rag-cite-title a:hover { text-decoration:underline; }
.rag-cite-id { font-family:var(--font-mono); color:var(--muted); font-size:0.75rem;
  background:rgba(16,22,34,0.65); border:1px solid var(--border); border-radius:6px; padding:1px 6px; }
.rag-cite-page { color:var(--muted); font-size:0.78rem; }
.rag-cite-meter { display:inline-block; width:54px; height:5px; background:rgba(148,163,184,0.18);
  border-radius:999px; overflow:hidden; vertical-align:middle; margin-left:auto; }
.rag-cite-fill { display:block; height:100%; border-radius:999px;
  background:linear-gradient(90deg, var(--primary), var(--info)); }
.rag-cite-score { font-family:var(--font-mono); background:var(--active-pill); color:var(--link);
  border-radius:999px; padding:1px 8px; font-size:0.74rem; font-weight:600; }
.rag-cite-details { margin-top:5px; }
.rag-cite-details > summary { color:var(--muted); font-size:0.78rem; cursor:pointer;
  list-style:none; display:inline-block; transition:color var(--speed) var(--ease); }
.rag-cite-details > summary::before { content:"▸ "; }
.rag-cite-details[open] > summary::before { content:"▾ "; }
.rag-cite-details > summary:hover { color:var(--ink); }
.rag-cite-details[open] .rag-cite-snippet { animation:rag-fade 160ms var(--ease); }
.rag-cite-snippet { color:var(--muted); font-size:0.82rem; margin-top:5px; line-height:1.5;
  border-left:2px solid var(--primary); padding-left:10px; }
.rag-lab-panel { display:flex; flex-direction:column; gap:10px; }
.rag-lab-summary { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:10px; }
.rag-lab-stat { background:var(--callout); border:1px solid var(--border); border-radius:10px;
  padding:10px 12px; }
.rag-lab-k { font-family:var(--font-mono); font-size:0.68rem; color:var(--muted); letter-spacing:0.06em; }
.rag-lab-v { font-size:1.25rem; font-weight:700; color:var(--ink); margin:2px 0; }
.rag-lab-s { font-size:0.74rem; color:var(--muted); }
.rag-latency-bar { display:flex; height:6px; border-radius:999px; overflow:hidden;
  background:var(--border); gap:2px; }
.rag-lat-ret { display:block; height:100%; background:var(--info); }
.rag-lat-rer { display:block; height:100%; background:var(--primary); }
.rag-lab-subhead { font-family:var(--font-mono); font-size:0.72rem; letter-spacing:0.06em;
  color:var(--link); font-weight:600; margin-top:2px; }
.rag-rerank-card { background:var(--callout); border:1px solid var(--border); border-radius:10px;
  padding:10px 12px; }
.rag-rerank-head { display:flex; flex-wrap:wrap; align-items:center; gap:7px; }
.rag-rank-badge { font-size:0.78rem; font-weight:700; color:#fff; background:var(--primary);
  border-radius:6px; padding:1px 7px; }
.rag-rerank-title, .rag-rerank-title a { font-weight:600; color:var(--ink); text-decoration:none; }
.rag-rerank-title a:hover { color:var(--link); text-decoration:underline; }
.rag-rank-flow { font-size:0.74rem; color:var(--muted); }
.rag-rerank-score-wrap { display:inline-flex; align-items:center; gap:6px; margin-left:auto; }
.rag-rerank-val { font-size:0.76rem; font-weight:600; color:var(--link); }
.rag-delta-up { font-family:var(--font-mono); font-size:0.74rem; font-weight:600; color:var(--success); }
.rag-delta-down { font-family:var(--font-mono); font-size:0.74rem; font-weight:600; color:var(--danger); }
.rag-delta-flat { font-family:var(--font-mono); font-size:0.74rem; color:var(--muted); }
.rag-rerank-snip { color:var(--muted); font-size:0.81rem; line-height:1.45; margin-top:6px;
  border-left:2px solid var(--border); padding-left:9px; }
.rag-stage1-details { border:1px solid var(--border); border-radius:10px; padding:8px 12px;
  background:rgba(16,22,34,0.65); }
.rag-stage1-details > summary { cursor:pointer; font-size:0.82rem; font-weight:600; color:var(--link);
  list-style:none; }
.rag-stage1-details > summary::before { content:"▸ "; }
.rag-stage1-details[open] > summary::before { content:"▾ "; }
.rag-stage1-wrap { overflow-x:auto; margin-top:8px; animation:rag-fade 160ms var(--ease); }
.rag-stage1-table { width:100%; border-collapse:collapse; font-size:0.79rem;
  font-variant-numeric:tabular-nums; }
.rag-stage1-table th { text-align:left; color:var(--muted); font-weight:600; padding:6px 8px;
  border-bottom:1px solid var(--border); font-family:var(--font-mono); font-size:0.72rem; }
.rag-stage1-table td { padding:6px 8px; border-bottom:1px solid rgba(38,50,72,0.55); }
.rag-s1-kept { background:rgba(99,102,241,0.08); }
.rag-lab-empty { color:var(--muted); font-size:0.86rem; padding:10px 0; }
.rag-corpus-overview { display:flex; flex-direction:column; gap:12px; margin-bottom:14px; }
.rag-kpi-grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; }
.rag-kpi-card { background:var(--card); border:1px solid var(--border); border-radius:14px;
  padding:14px 16px; }
.rag-kpi-lbl { font-family:var(--font-mono); font-size:0.7rem; letter-spacing:0.06em; color:var(--muted); }
.rag-kpi-val { font-size:1.7rem; font-weight:700; color:var(--ink); margin:4px 0 2px; }
.rag-kpi-sub { font-size:0.78rem; color:var(--muted); overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; }
.rag-dist-box { background:var(--card); border:1px solid var(--border); border-radius:14px;
  padding:12px 16px; }
.rag-dist-head { display:flex; justify-content:space-between; font-family:var(--font-mono);
  font-size:0.72rem; letter-spacing:0.05em; color:var(--muted); margin-bottom:8px; }
.rag-dist-bar { display:flex; height:10px; border-radius:999px; overflow:hidden; gap:2px;
  background:var(--sidebar); }
.rag-dist-seg { display:block; height:100%; }
.rag-dist-legend { display:flex; flex-wrap:wrap; gap:10px 16px; margin-top:10px; }
.rag-dist-leg { display:inline-flex; align-items:center; gap:6px; font-size:0.76rem; color:var(--ink); }
.rag-dist-dot { width:8px; height:8px; border-radius:2px; display:inline-block; }
.rag-dist-num { color:var(--muted); font-size:0.72rem; }
.rag-doc-card { margin-bottom:10px; }
.rag-doc-badges { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }
.rag-badge { display:inline-block; font-family:var(--font-mono); font-size:0.69rem; font-weight:600;
  border-radius:999px; padding:2px 8px; border:1px solid var(--border); }
.rag-badge-cat { color:var(--link); background:var(--active-pill); border-color:rgba(165,180,252,0.25); }
.rag-badge-en { color:var(--info); background:rgba(56,189,248,0.12); border-color:rgba(56,189,248,0.35); }
.rag-badge-vi { color:#C4B5FD; background:rgba(167,139,250,0.12); border-color:rgba(167,139,250,0.35); }
.rag-badge-ok { color:var(--success); background:rgba(52,211,153,0.12); border-color:rgba(52,211,153,0.35); }
.rag-badge-dim { color:var(--muted); background:rgba(148,163,184,0.08); }
.rag-doc-title { font-weight:600; font-size:0.98rem; color:var(--ink); line-height:1.35; }
.rag-doc-id { color:var(--muted); font-size:0.75rem; margin-top:3px; }
.rag-doc-stats { display:flex; justify-content:space-between; color:var(--muted); font-size:0.76rem;
  margin-top:8px; }
.rag-doc-bar { height:4px; background:var(--sidebar); border-radius:999px; overflow:hidden;
  margin-top:6px; }
.rag-doc-bar > span { display:block; height:100%; border-radius:999px;
  background:linear-gradient(90deg, var(--primary), var(--info)); }
@media (max-width: 768px) {
  .rag-lab-summary, .rag-kpi-grid { grid-template-columns:1fr; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition-duration:0.01ms !important;
    animation-duration:0.01ms !important; animation-iteration-count:1 !important; }
}
""")


def _global_css():
    return _CSS_TEMPLATE.substitute(**PALETTE)


def inject_global_css():
    import streamlit as st
    st.markdown(f"<style>{_global_css()}</style>", unsafe_allow_html=True)
