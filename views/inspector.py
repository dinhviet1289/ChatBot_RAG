"""Kiểm tra truy hồi — trang nav riêng: mổ xẻ tầng hybrid retrieve + rerank cho
một câu hỏi bất kỳ, không gọi LLM nên không tốn token hạn mức. Tách khỏi màn
hình hội thoại để cuộc hội thoại mới luôn mở ra bằng ô chat trống.
"""

import streamlit as st

from rag_engine_web import inspect_retrieval


def render_retrieval_panel(res):
    t1, t2, t3 = st.columns(3)
    with t1:
        st.metric("Hybrid retrieve", f"{res['retrieve_ms']:.0f} ms")
    with t2:
        st.metric("Cohere rerank", f"{res['rerank_ms']:.0f} ms" if res["rerank_enabled"] else "TẮT")
    with t3:
        st.metric("Ứng viên", res["total_candidates"])

    st.markdown("**Top-K sau rerank**")
    for item in res["selected"]:
        d = item["rank_delta"]
        if d > 0:
            badge = f":material/trending_up: +{d}"
        elif d < 0:
            badge = f":material/trending_down: {d}"
        else:
            badge = ":material/drag_handle: 0"
        with st.container(border=True):
            st.markdown(f"**{item['doc_title']}** · trang {item['trang']} · {badge}")
            st.caption(item["snippet"][:200])
    with st.expander(f"Toàn bộ {res['total_candidates']} ứng viên stage-1"):
        for item in res["candidates"]:
            st.caption(f"#{item['initial_rank']} {item['doc_title']} · trang {item['trang']}")


def render_inspector_view(bundle):
    st.markdown("### Kiểm tra truy hồi")
    st.caption("Nhập một câu hỏi để xem tầng truy hồi chọn tài liệu nào: thứ tự hybrid, "
               "rerank kéo hạng nào lên/xuống, mất bao nhiêu ms. Không gọi LLM — không tốn token.")
    with st.container(border=True):
        dq = st.text_input("Câu hỏi kiểm tra", key="dbg_q",
                           placeholder="Nhập câu hỏi để xem tầng truy hồi chọn tài liệu nào...")
        dq = (dq or "").strip()
        if st.button("Chạy kiểm tra", key="dbg_run") and dq:
            with st.spinner("Đang truy hồi tài liệu..."):
                render_retrieval_panel(inspect_retrieval(bundle, dq))