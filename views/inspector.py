"""Kiểm tra truy hồi — trang nav riêng: mổ xẻ tầng hybrid retrieve + rerank cho
một câu hỏi bất kỳ, không gọi LLM nên không tốn token hạn mức. Tách khỏi màn
hình hội thoại để cuộc hội thoại mới luôn mở ra bằng ô chat trống.
"""

import streamlit as st

from rag_engine_web import inspect_retrieval
from ui.theme import (
    render_page_header_html,
    render_retrieval_empty_html,
    render_retrieval_panel_html,
    render_retrieval_stage1_html,
    render_retrieval_stage2_html,
    render_retrieval_summary_html,
)

SAMPLE_PROBES = [
    ("PEAS", "Mô hình PEAS gồm những thành phần nào? Cho ví dụ."),
    ("DFS vs BFS", "So sánh thuật toán tìm kiếm DFS và BFS trong AI."),
    ("Minimax", "Thuật toán Minimax và cắt tỉa Alpha-Beta hoạt động ra sao?"),
]


def _queue_probe(sample_q):
    st.session_state["dbg_q"] = sample_q
    st.session_state["dbg_auto"] = True


def render_retrieval_panel(res):
    st.markdown(render_retrieval_panel_html(res), unsafe_allow_html=True)


def render_inspector_view(bundle):
    cfg = (bundle or {}).get("config") or {}
    embed_dim = int(cfg.get("embed_dim") or 384)
    provider = str(cfg.get("rerank_provider") or "off").strip().lower()
    rerank_chip = ("Cohere Rerank", "success") if provider == "cohere" else (
        ("Local Rerank", "success") if provider == "local" else ("Stage-1 Only", "warn")
    )

    st.markdown(
        render_page_header_html(
            "Kiểm tra truy hồi (Retrieval Lab)",
            f"Mổ xẻ pipeline 2 tầng: Stage 1 Hybrid (0.3·BM25_df + 0.7·FAISS_{embed_dim}d) "
            "→ Stage 2 Cross-Encoder Rerank. Không gọi LLM — không tốn token hạn mức.",
            [
                ("BM25 0.3 + FAISS 0.7", "info"),
                rerank_chip,
                ("0 token LLM", "primary"),
            ],
        ),
        unsafe_allow_html=True,
    )

    auto_run = bool(st.session_state.pop("dbg_auto", False))
    with st.container(border=True):
        with st.form("dbg_form", border=False):
            dq = st.text_area(
                "Câu hỏi kiểm tra",
                key="dbg_q",
                height=78,
                placeholder="Nhập câu hỏi để xem tầng truy hồi chọn tài liệu nào...",
            )
            clicked = st.form_submit_button(":material/science: Chạy kiểm tra",
                                            type="primary", width="stretch")
        st.caption("Thử nhanh mẫu chuẩn")
        p_cols = st.columns(len(SAMPLE_PROBES))
        for idx, (label, sample_q) in enumerate(SAMPLE_PROBES):
            with p_cols[idx]:
                st.button(
                    f":material/bolt: {label}",
                    key=f"probe_{label}",
                    width="stretch",
                    on_click=_queue_probe,
                    args=(sample_q,),
                )

    dq = (dq or "").strip()
    res = None
    if (clicked or auto_run) and dq:
        with st.spinner("Đang truy hồi tài liệu..."):
            try:
                res = inspect_retrieval(bundle, dq)
            except Exception as exc:
                st.error(f"Không truy hồi được ({type(exc).__name__}).")
                res = st.session_state.get("dbg_res")
            else:
                st.session_state["dbg_res"] = res
    elif st.session_state.get("dbg_res"):
        res = st.session_state["dbg_res"]

    if not res:
        st.markdown(render_retrieval_empty_html(embed_dim), unsafe_allow_html=True)
    else:
        st.markdown(render_retrieval_summary_html(res), unsafe_allow_html=True)
        tab_s2, tab_s1 = st.tabs(["Stage 2 · Rerank", "Stage 1 · Ứng viên"])
        with tab_s2:
            st.markdown(render_retrieval_stage2_html(res), unsafe_allow_html=True)
        with tab_s1:
            st.markdown(render_retrieval_stage1_html(res, collapsible=False),
                        unsafe_allow_html=True)
