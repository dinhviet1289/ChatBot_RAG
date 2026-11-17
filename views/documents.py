"""Documents view: corpus KPIs, distribution bar, category filter, search, cards."""

import streamlit as st

from ui.catalog import CATEGORY_ORDER, summarize_docstore
from ui.state import ask_about_document
from ui.theme import render_corpus_overview_html, render_doc_card_html


def render_documents_view(bundle):
    state = st.session_state
    rows = summarize_docstore(bundle["db"].docstore._dict)
    total_chunks = sum(r["chunk_count"] for r in rows)
    max_chunks = max((r["chunk_count"] for r in rows), default=0)

    st.subheader("Kho tài liệu đã nạp")
    st.caption("Toàn bộ học liệu được đánh chỉ mục FAISS + BM25 cho trợ lý học tập ELC3008.")

    st.markdown(render_corpus_overview_html(rows), unsafe_allow_html=True)

    sel = st.pills("Nhóm tài liệu", ["Tất cả"] + list(CATEGORY_ORDER),
                   default="Tất cả", key="doc_cat")
    query = st.text_input("Tìm tài liệu", placeholder="Nhập tên, doc_id hoặc ngôn ngữ (EN/VI)...",
                          key="doc_q")
    q = (query or "").strip().lower()
    filtered = [r for r in rows
                if (sel in (None, "Tất cả") or r["category"] == sel)
                and (not q or q in r["title"].lower()
                     or q in r["doc_id"].lower()
                     or q in r["category"].lower()
                     or q == r.get("lang", "").lower())]
    st.caption(f"Hiển thị {len(filtered)} / {len(rows)} tài liệu khớp bộ lọc")

    cols = st.columns(2)
    for i, r in enumerate(filtered):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(render_doc_card_html(r, max_chunks, total_chunks),
                            unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                with b1:
                    if r["url"]:
                        st.link_button("Mở Drive", r["url"], width="stretch")
                    else:
                        st.button("Chỉ có trong chỉ mục", key=f"idx_{r['doc_id']}",
                                  width="stretch", disabled=True)
                with b2:
                    if st.button("Hỏi thử tài liệu này", key=f"ask_{r['doc_id']}",
                                 width="stretch"):
                        ask_about_document(state, r["doc_id"], r["sample_question"])
                        st.rerun()
