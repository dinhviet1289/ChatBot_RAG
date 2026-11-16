"""Chat view: hero + suggested cards, streaming answers, persistent citations."""

import streamlit as st

from rag_engine_web import (
    describe_llm_error,
    format_docs,
    model_chain,
    rotation_pairs,
    stream_answer,
)
from ui.catalog import serialize_sources, should_show_sources
from ui.state import (
    SUGGESTED_CARDS,
    append_assistant_turn,
    append_user_turn,
    get_active_conversation,
    prepare_regenerate,
)
from ui.theme import render_citation_box_html

USER_AVATAR = ":material/person:"
AI_AVATAR = ":material/school:"


def _render_sources(sources):
    st.markdown(render_citation_box_html(sources), unsafe_allow_html=True)


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
                    state["pending_prompt"] = card["prompt"]
                    st.rerun()


def render_chat_view(bundle):
    state = st.session_state
    conv = get_active_conversation(state)

    if not conv["messages"] and not state.get("pending_prompt"):
        _render_empty_state(state)

    for msg in conv["messages"]:
        avatar = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    if conv["messages"] and conv["messages"][-1]["role"] == "assistant":
        if st.button(":material/refresh: Tạo lại câu trả lời", key="regen"):
            q = prepare_regenerate(state)
            if q:
                state["pending_prompt"] = q
                st.rerun()

    prompt = st.chat_input("Nhập câu hỏi của bạn...")
    if state.get("pending_prompt"):
        prompt = state.pop("pending_prompt")

    if prompt:
        append_user_turn(state, prompt)
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.spinner("Đang tra cứu tài liệu..."):
                docs = bundle["retriever"].invoke(prompt)
            try:
                served = {}
                keys = bundle["config"]["groq_keys"]
                answer = st.write_stream(stream_answer(
                    bundle["make_answer_chain"],
                    rotation_pairs(keys, model_chain(bundle["config"]["llm_model"])),
                    {"context": format_docs(docs), "question": prompt},
                    report=lambda model, key, fell_back: served.update(
                        model=model, key=key, fell_back=fell_back)))
                sources = serialize_sources(docs) if should_show_sources(answer) else []
                if served.get("fell_back"):
                    where = (f" · key {keys.index(served['key']) + 1}/{len(keys)}"
                             if len(keys) > 1 else "")
                    st.caption(
                        f":material/swap_horiz: `{bundle['config']['llm_model']}` vừa cạn hạn mức, "
                        f"câu này trả lời bằng `{served['model']}`{where}.")
            except Exception as exc:
                answer = describe_llm_error(exc)
                sources = []
                st.error(answer)
            if sources:
                _render_sources(sources)
        append_assistant_turn(state, answer, sources)
        st.rerun()