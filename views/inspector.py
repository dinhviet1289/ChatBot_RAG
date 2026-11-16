"""Inspector view: mổ xẻ 2 tầng truy hồi (hybrid retrieve + rerank) cho một câu hỏi."""

import streamlit as st

from rag_engine_web import describe_llm_error, inspect_retrieval

PRESET_QUESTIONS = ["So sánh DFS và BFS", "PEAS là gì?", "GAN khác VAE thế nào?"]


def render_inspector_view(bundle):
    pc = st.columns(3)
    for c, p in zip(pc, PRESET_QUESTIONS):
        if c.button(p, width="stretch", key=f"preset_{p}"):
            st.session_state["insp_q"] = p
    q = st.text_input("Câu hỏi kiểm tra truy hồi", key="insp_q")
    gen = st.checkbox("Sinh luôn câu trả lời LLM để đối chiếu")
    if st.button("Chạy inspector", type="primary") and q:
        res = inspect_retrieval(bundle, q)
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
        if gen:
            ctx = "\n\n".join(it["snippet"] for it in res["selected"])
            try:
                st.markdown(bundle["answer_chain"].invoke({"context": ctx, "question": q}))
            except Exception as exc:
                st.error(describe_llm_error(exc))
