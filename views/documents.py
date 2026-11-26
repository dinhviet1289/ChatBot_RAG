"""Documents view: header, corpus KPIs, distribution bar, compact 1-row filter, 2-col cards."""

import streamlit as st

from ui.catalog import CATEGORY_ORDER, summarize_docstore
from ui.state import ask_about_document
from ui.theme import (
    render_corpus_overview_html,
    render_doc_card_html,
    render_page_header_html,
)


def _reset_doc_filters():
    st.session_state["doc_q"] = ""
    st.session_state["doc_cat"] = "Tất cả"
    st.session_state["doc_lang"] = "Tất cả"
    st.session_state["doc_sort"] = "Mặc định"


def render_documents_view(bundle):
    state = st.session_state
    rows = bundle.get("doc_summary") or summarize_docstore(bundle["db"].docstore._dict)
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

    f1, f2, f3, f4 = st.columns([2.2, 1.8, 0.9, 1.3], vertical_alignment="bottom")
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
    with f4:
        sort_sel = st.selectbox(
            "Sắp xếp",
            ["Mặc định", "Nhiều chunks nhất", "Tên A → Z"],
            key="doc_sort",
        )

    q = (query or "").strip().lower()
    filtered = [r for r in rows
                if (sel in (None, "Tất cả") or r["category"] == sel)
                and (lang_sel in (None, "Tất cả") or r.get("lang", "").upper() == lang_sel)
                and (not q or q in r["title"].lower()
                     or q in r["doc_id"].lower()
                     or q in r["category"].lower()
                     or q == r.get("lang", "").lower())]
    if sort_sel == "Nhiều chunks nhất":
        filtered.sort(key=lambda r: (-int(r["chunk_count"]), r["title"].lower()))
    elif sort_sel == "Tên A → Z":
        filtered.sort(key=lambda r: r["title"].lower())
    has_filter = bool(
        q
        or (sel not in (None, "Tất cả"))
        or (lang_sel not in (None, "Tất cả"))
        or (sort_sel not in (None, "Mặc định"))
    )
    if has_filter:
        c_cnt, c_rst = st.columns([4, 1], vertical_alignment="center")
        with c_cnt:
            st.caption(f"Hiển thị {len(filtered)} / {len(rows)} tài liệu khớp bộ lọc")
        with c_rst:
            st.button("Đặt lại bộ lọc", key="doc_reset", width="stretch",
                      on_click=_reset_doc_filters)
    else:
        st.caption(f"Hiển thị {len(filtered)} / {len(rows)} tài liệu khớp bộ lọc")

    if not filtered:
        st.info("Không tìm thấy tài liệu nào khớp với bộ lọc hoặc từ khóa hiện tại.")
        return

    cols = st.columns(2)
    for i, r in enumerate(filtered):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(render_doc_card_html(r, max_chunks, total_chunks),
                            unsafe_allow_html=True)
                if r["url"]:
                    b1, b2 = st.columns(2)
                    with b1:
                        btn_label = (
                            ":material/open_in_new: Mở Drive"
                            if "drive.google.com" in r["url"]
                            else ":material/open_in_new: Mở PDF"
                        )
                        st.link_button(btn_label, r["url"], width="stretch")
                    with b2:
                        if st.button(":material/chat: Hỏi tài liệu",
                                     key=f"ask_{r['doc_id']}", width="stretch"):
                            ask_about_document(state, r["doc_id"], r["sample_question"])
                            st.rerun()
                else:
                    if st.button(":material/chat: Hỏi thử tài liệu này",
                                 key=f"ask_{r['doc_id']}", width="stretch"):
                        ask_about_document(state, r["doc_id"], r["sample_question"])
                        st.rerun()
