"""Kiểm tra truy hồi — trang nav riêng: mổ xẻ tầng hybrid retrieve + rerank cho
một câu hỏi bất kỳ, không gọi LLM nên không tốn token hạn mức. Tách khỏi màn
hình hội thoại để cuộc hội thoại mới luôn mở ra bằng ô chat trống.
"""

import streamlit as st

from rag_engine_web import inspect_retrieval
from ui.theme import render_retrieval_panel_html

SAMPLE_PROBES = [
    ("PEAS", "Mô hình PEAS gồm những thành phần nào? Cho ví dụ."),
    ("DFS vs BFS", "So sánh thuật toán tìm kiếm DFS và BFS trong AI."),
    ("Minimax", "Thuật toán Minimax và cắt tỉa Alpha-Beta hoạt động ra sao?"),
]


def render_retrieval_panel(res):
    st.markdown(render_retrieval_panel_html(res), unsafe_allow_html=True)


def render_inspector_view(bundle):
    embed_dim = int(((bundle or {}).get("config") or {}).get("embed_dim") or 384)
    st.markdown("### Kiểm tra truy hồi (Retrieval Lab)")
    st.caption(f"Mổ xẻ pipeline 2 tầng: Stage 1 Hybrid (0.3·BM25_df + 0.7·FAISS_{embed_dim}d) "
               "→ Stage 2 Cross-Encoder Rerank. Không gọi LLM — không tốn token hạn mức.")

    with st.container(border=True):
        st.caption("Thử nhanh mẫu chuẩn")
        pcols = st.columns(len(SAMPLE_PROBES))
        auto_run = False
        for col, (label, sample_q) in zip(pcols, SAMPLE_PROBES):
            with col:
                if st.button(f":material/bolt: {label}", key=f"probe_{label}", width="stretch"):
                    st.session_state["dbg_q"] = sample_q
                    auto_run = True

        dq = st.text_input("Câu hỏi kiểm tra", key="dbg_q",
                           placeholder="Nhập câu hỏi để xem tầng truy hồi chọn tài liệu nào...")
        dq = (dq or "").strip()
        clicked = st.button(":material/science: Chạy kiểm tra", key="dbg_run", type="primary")
        if (clicked or auto_run) and dq:
            with st.spinner("Đang truy hồi tài liệu..."):
                res = inspect_retrieval(bundle, dq)
            render_retrieval_panel(res)
