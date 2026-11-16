"""Màn hình hội thoại — mở từ sidebar (nút tạo mới, chọn cuộc hội thoại, hoặc
"Hỏi thử tài liệu này" bên Kho tài liệu), không nằm trong nav. Ô chat dưới đáy,
Enter để gửi; mỗi lượt được lưu kèm trích dẫn và expander số liệu truy hồi.
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
    SUGGESTED_CARDS,
    append_assistant_turn,
    append_user_turn,
    get_active_conversation,
    prepare_regenerate,
)
from ui.theme import render_citation_box_html
from views.inspector import render_retrieval_panel

USER_AVATAR = ":material/person:"
AI_AVATAR = ":material/school:"


def _render_history(conv):
    for msg in conv["messages"]:
        avatar = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                st.markdown(render_citation_box_html(msg["sources"]), unsafe_allow_html=True)


def _render_empty_state(state):
    st.markdown(
        "<div style='text-align:center;padding:2.5rem 0 1.5rem;'>"
        "<div style='font-size:2rem;font-weight:700;letter-spacing:-0.02em;'>Bạn muốn ôn tập phần nào?</div>"
        "<div style='color:var(--muted);margin-top:6px;'>Chọn một gợi ý hoặc nhập câu hỏi bên dưới.</div>"
        "</div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, card in enumerate(SUGGESTED_CARDS):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f":material/{card['icon']}: **{card['title']}**")
                st.caption(card["subtitle"])
                if st.button(card["prompt"], key=f"sug_{i}", width="stretch"):
                    # Không rerun: pending_prompt được tiêu ngay trong cùng lần chạy.
                    state["pending_prompt"] = card["prompt"]


def render_conversation_view(bundle):
    state = st.session_state
    conv = get_active_conversation(state)

    if conv["messages"]:
        st.markdown(f"### {conv['title']}")
        st.caption("Enter để gửi câu hỏi. Câu trả lời kèm trích dẫn được lưu vào cuộc hội thoại; "
                   "số liệu vì sao tài liệu được chọn nằm trong mục mở rộng dưới mỗi lượt.")

    _render_history(conv)

    if conv["messages"] and conv["messages"][-1]["role"] == "assistant":
        if st.button(":material/refresh: Tạo lại câu trả lời", key="regen"):
            q = prepare_regenerate(state)
            if q:
                state["pending_prompt"] = q
                # Lịch sử đã render cả lượt cũ; phải rerun để vẽ lại sau khi pop.
                st.rerun()

    if not conv["messages"] and not state.get("pending_prompt"):
        _render_empty_state(state)

    prompt = st.chat_input("Nhập câu hỏi của bạn...")
    if state.get("pending_prompt"):
        prompt = state.pop("pending_prompt")
    prompt = (prompt or "").strip()
    if not prompt:
        return

    started = time.perf_counter()
    append_user_turn(state, prompt)
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.spinner("Đang truy hồi tài liệu..."):
        res = inspect_retrieval(bundle, prompt)

    ctx = "\n\n".join(it["snippet"] for it in res["selected"])
    keys = bundle["config"]["groq_keys"]
    served = {}
    with st.chat_message("assistant", avatar=AI_AVATAR):
        try:
            answer = st.write_stream(stream_answer(
                bundle["make_answer_chain"],
                rotation_pairs(keys, model_chain(bundle["config"]["llm_model"])),
                {"context": ctx, "question": prompt},
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
    with st.expander("Chi tiết truy hồi của lượt này"):
        render_retrieval_panel(res)