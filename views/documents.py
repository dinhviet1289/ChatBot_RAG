"""Documents view: corpus KPIs, category filter, search, per-document cards."""

import streamlit as st

from ui.catalog import CATEGORY_ORDER, summarize_docstore
from ui.state import ask_about_document


def _kpi(value, label, sub=None):
    with st.container(border=True):
        st.markdown(f"<div style='font-size:1.6rem;font-weight:700;letter-spacing:-0.02em;'>{value}</div>",
                    unsafe_allow_html=True)
        st.caption(label)
        if sub:
            st.caption(sub)


def render_documents_view(bundle):
    state = st.session_state
    rows = summarize_docstore(bundle["db"].docstore._dict)
    total_chunks = sum(r["chunk_count"] for r in rows)

    st.subheader("Kho tài liệu đã nạp")
    st.caption("Toàn bộ học liệu được đánh chỉ mục FAISS + BM25 cho trợ lý.")

    c1, c2, c3 = st.columns(3)
    with c1:
        _kpi(len(rows), "Học liệu", f"{len(rows)} doc_id trong chỉ mục")
    with c2:
        _kpi(f"{total_chunks:,}", "Chunks", "đã vector hoá")
    with c3:
        _kpi("384", "Chiều embedding", "paraphrase-multilingual-MiniLM-L12-v2")

    sel = st.pills("Nhóm tài liệu", ["Tất cả"] + list(CATEGORY_ORDER),
                   default="Tất cả", key="doc_cat")
    query = st.text_input("Tìm tài liệu", placeholder="Nhập tên hoặc doc_id...", key="doc_q")
    q = (query or "").lower()
    filtered = [r for r in rows
                if (sel in (None, "Tất cả") or r["category"] == sel)
                and (not q or q in r["title"].lower() or q in r["doc_id"].lower())]
    st.caption(f"{len(filtered)} tài liệu khớp bộ lọc")

    cols = st.columns(2)
    for i, r in enumerate(filtered):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{r['title']}**")
                st.caption(f"{r['doc_id']} · {r['category']}")
                if r["page_min"] is None:
                    pages = "—"
                elif r["page_min"] == r["page_max"]:
                    pages = str(r["page_min"])
                else:
                    pages = f"{r['page_min']}-{r['page_max']}"
                st.caption(f"{r['chunk_count']} chunks · trang {pages}")
                b1, b2 = st.columns(2)
                with b1:
                    if r["url"]:
                        st.link_button("Mở Drive", r["url"], width="stretch")
                    else:
                        st.markdown(
                            "<span style='display:inline-block;background:var(--active-pill);"
                            "color:var(--primary);border-radius:999px;padding:4px 12px;"
                            "font-size:0.8rem;font-weight:600;'>Đã nạp chỉ mục</span>",
                            unsafe_allow_html=True)
                with b2:
                    if st.button("Hỏi thử tài liệu này", key=f"ask_{r['doc_id']}", width="stretch"):
                        ask_about_document(state, r["doc_id"], r["sample_question"])
                        st.rerun()