"""Hỏi đáp & Kiểm tra truy hồi: lịch sử hội thoại + mổ xẻ 2 tầng truy hồi.

Mỗi lượt chạy có sinh LLM được lưu vào cuộc hội thoại đang mở (câu hỏi, câu trả
lời, nguồn trích dẫn, độ trễ). Chạy chỉ truy hồi (gen=False) không đụng lịch sử.
"""

import time

import streamlit as st

from rag_engine_web import (
    describe_llm_error,
    inspect_retrieval,
    model_chain,
    rotation_pairs,
    stream_answer,
)
from ui.catalog import should_show_sources, sources_from_inspection
from ui.state import (
    append_assistant_turn,
    append_user_turn,
    get_active_conversation,
    prepare_regenerate,
)
from ui.theme import render_citation_box_html

PRESET_QUESTIONS = ["So sánh DFS và BFS", "PEAS là gì?", "GAN khác VAE thế nào?"]
USER_AVATAR = ":material/person:"
AI_AVATAR = ":material/school:"


def _render_history(conv):
    for msg in conv["messages"]:
        avatar = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                st.markdown(render_citation_box_html(msg["sources"]), unsafe_allow_html=True)


def _render_retrieval_panel(res):
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
    state = st.session_state
    conv = get_active_conversation(state)

    st.markdown("### Hỏi đáp & Kiểm tra truy hồi")
    st.caption("Câu trả lời kèm nguồn trích dẫn được lưu vào cuộc hội thoại đang mở; "
               "số liệu bên dưới cho thấy vì sao tài liệu đó được chọn.")

    _render_history(conv)

    if conv["messages"] and conv["messages"][-1]["role"] == "assistant":
        if st.button(":material/refresh: Tạo lại câu trả lời", key="regen"):
            q = prepare_regenerate(state)
            if q:
                state["pending_prompt"] = q
                st.rerun()

    queued = False
    if state.get("pending_prompt"):
        st.session_state["insp_q"] = state.pop("pending_prompt")
        queued = True

    pc = st.columns(3)
    for c, p in zip(pc, PRESET_QUESTIONS):
        if c.button(p, width="stretch", key=f"preset_{p}"):
            st.session_state["insp_q"] = p
    q = st.text_input("Câu hỏi", key="insp_q",
                      placeholder="Nhập câu hỏi hoặc chọn một gợi ý...")
    q = (q or "").strip()
    gen = st.checkbox("Sinh câu trả lời LLM và lưu vào cuộc hội thoại", value=True)
    run = st.button(":material/play_arrow: Chạy", type="primary") and q
    if not q or not (run or queued):
        return

    started = time.perf_counter()
    if gen:
        append_user_turn(state, q)
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(q)

    with st.spinner("Đang truy hồi tài liệu..."):
        res = inspect_retrieval(bundle, q)
    _render_retrieval_panel(res)
    if not gen:
        return

    ctx = "\n\n".join(it["snippet"] for it in res["selected"])
    keys = bundle["config"]["groq_keys"]
    served = {}
    with st.chat_message("assistant", avatar=AI_AVATAR):
        try:
            answer = st.write_stream(stream_answer(
                bundle["make_answer_chain"],
                rotation_pairs(keys, model_chain(bundle["config"]["llm_model"])),
                {"context": ctx, "question": q},
                report=lambda model, key, fell_back: served.update(
                    model=model, key=key, fell_back=fell_back)))
            sources = sources_from_inspection(res["selected"]) if should_show_sources(answer) else []
            if served:
                where = (f" · key {keys.index(served['key']) + 1}/{len(keys)}"
                         if len(keys) > 1 else "")
                note = (f" :material/swap_horiz: `{bundle['config']['llm_model']}` vừa cạn hạn mức, "
                        f"câu này trả lời bằng model khác." if served.get("fell_back") else "")
                st.caption(f"LLM `{served['model']}`{where}.{note}")
            if sources:
                st.markdown(render_citation_box_html(sources), unsafe_allow_html=True)
        except Exception as exc:
            answer = describe_llm_error(exc)
            sources = []
            st.error(answer)
            if st.button(":material/refresh: Thử lại", key="retry_after_error"):
                again = prepare_regenerate(state)
                if again:
                    state["pending_prompt"] = again
                    st.rerun()
    append_assistant_turn(state, answer, sources, latency_s=time.perf_counter() - started)