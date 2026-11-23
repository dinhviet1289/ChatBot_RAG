"""Documents view: header, corpus KPIs, distribution bar, compact 1-row filter, 3-col cards."""

import streamlit as st

from ui.catalog import CATEGORY_ORDER, summarize_docstore
from ui.state import ask_about_document
from ui.theme import (
    render_corpus_overview_html,
    render_doc_card_html,
    render_page_header_html,
)


def render_documents_view(bundle):
    state = st.session_state
    rows = summarize_docstore(bundle["db"].docstore._dict)
    total_chunks = sum(r["chunk_count"] for r in rows)
    max_chunks = max((r["chunk_count"] for r in rows), default=0)

    cfg = bundle.get("config", {})
    embed_dim = int(cfg.get("embed_dim") or 384)
    embed_model = cfg.get("embed_model") or "paraphrase-multilingual-MiniLM-L12-v2"

    st.markdown(
        render_page_header_html(
            "Kho tài liệu đã nạp",
            "Toàn bộ học liệu được đánh chỉ mục FAISS + BM25 cho trợ lý học tập ELC3008.",
            [
                (f"{len(rows)} học liệu", "primary"),
                (f"{total_chunks:,} chunks", "info"),
                (f"FAISS {embed_dim}d", "neutral"),
            ],
        ),
        unsafe_allow_html=True,
    )

    st.markdown(render_corpus_overview_html(
        rows,
        embed_dim=embed_dim,
        embed_model=embed_model,
    ), unsafe_allow_html=True)

    f1, f2, f3 = st.columns([2.4, 2.1, 1.0], vertical_alignment="bottom")
    with f1:
        query = st.text_input("Tìm tài liệu",
                              placeholder="Tìm theo tên hoặc doc_id...",
                              key="doc_q")
    with f2:
        sel = st.selectbox("Nhóm học phần", ["Tất cả"] + list(CATEGORY_ORDER),
                           key="doc_cat")
    with f3:
        lang_sel = st.selectbox("Ngôn ngữ", ["Tất cả", "VI", "EN"],
                                key="doc_lang")

    q = (query or "").strip().lower()
    filtered = [r for r in rows
                if (sel in (None, "Tất cả") or r["category"] == sel)
                and (lang_sel in (None, "Tất cả") or r.get("lang", "").upper() == lang_sel)
                and (not q or q in r["title"].lower()
                     or q in r["doc_id"].lower()
                     or q in r["category"].lower()
                     or q == r.get("lang", "").lower())]
    st.caption(f"Hiển thị {len(filtered)} / {len(rows)} tài liệu khớp bộ lọc")

    if not filtered:
        st.info("Không tìm thấy tài liệu nào khớp với bộ lọc hoặc từ khóa hiện tại.")
        return

    cols = st.columns(3)
    for i, r in enumerate(filtered):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(render_doc_card_html(r, max_chunks, total_chunks),
                            unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                with b1:
                    if r["url"]:
                        st.link_button("Mở Drive", r["url"], width="stretch")
                    else:
                        st.button("Chỉ mục", key=f"idx_{r['doc_id']}",
                                  width="stretch", disabled=True,
                                  help="Tài liệu chỉ có trong Vector Index")
                with b2:
                    if st.button("Hỏi tài liệu", key=f"ask_{r['doc_id']}",
                                 width="stretch"):
                        ask_about_document(state, r["doc_id"], r["sample_question"])
                        st.rerun()
