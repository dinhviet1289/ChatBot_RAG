"""Màn hình hội thoại — mở từ sidebar (nút tạo mới, chọn cuộc hội thoại, hoặc
"Hỏi thử tài liệu này" bên Kho tài liệu), không nằm trong nav. Ô chat dưới đáy,
Enter để gửi; mỗi lượt được lưu kèm telemetry, trích dẫn và biến động hạng sau rerank.
"""

import html
import time

import streamlit as st

from rag_engine_web import (
    describe_llm_error,
    inspect_retrieval,
    model_chain,
    rotation_pairs,
    stream_answer,
)
from ui.catalog import (
    CATALOG,
    format_selected_context,
    should_show_sources,
    sources_from_inspection,
)
from ui.state import (
    SUGGESTED_CARDS,
    append_assistant_turn,
    append_user_turn,
    get_active_conversation,
    prepare_regenerate,
)
from ui.theme import (
    latex_to_plain_text,
    render_citation_box_html,
    render_turn_telemetry_html,
)

USER_AVATAR = ":material/person:"
AI_AVATAR = ":material/school:"


def _build_telemetry(cfg, res, served, keys, latency_s):
    key_slot = ""
    if served and len(keys) > 1 and served.get("key") in keys:
        key_slot = f"{keys.index(served['key']) + 1}/{len(keys)}"
    return {
        "latency_s": round(float(latency_s), 2),
        "retrieve_ms": res.get("retrieve_ms"),
        "rerank_ms": res.get("rerank_ms"),
        "rerank_enabled": bool(res.get("rerank_enabled")),
        "selected_count": len(res.get("selected") or []),
        "total_candidates": int(res.get("total_candidates") or 0),
        "model": (served or {}).get("model") or cfg.get("llm_model") or "",
        "key_slot": key_slot,
        "fell_back": bool((served or {}).get("fell_back")),
        "configured_model": cfg.get("llm_model") or "",
    }


def _render_history(conv):
    for msg in conv["messages"]:
        avatar = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            content = (latex_to_plain_text(msg["content"])
                       if msg["role"] == "assistant" else msg["content"])
            st.markdown(content)
            if msg["role"] == "assistant":
                if msg.get("telemetry"):
                    st.markdown(render_turn_telemetry_html(msg["telemetry"]),
                                unsafe_allow_html=True)
                if msg.get("sources"):
                    with st.expander(f"Nguồn trích dẫn & truy hồi ({len(msg['sources'])})",
                                     expanded=False):
                        st.markdown(render_citation_box_html(msg["sources"]),
                                    unsafe_allow_html=True)


def _render_empty_state(state, bundle=None):
    cfg = (bundle or {}).get("config") or {}
    db = (bundle or {}).get("db")
    chunk_count = len(db.docstore._dict) if db is not None else 15510
    doc_count = len(CATALOG)
    provider = str(cfg.get("rerank_provider") or "off").strip().lower()
    rerank_pill = "Cohere Rerank" if provider == "cohere" else (
        "Local Rerank" if provider == "local" else "Hybrid Stage-1"
    )
    embed_dim = int(cfg.get("embed_dim") or 384)
    st.markdown(
        f'<div class="rag-hero">'
        f'<span class="rag-hero-badge">ELC3008 KNOWLEDGE OBSERVATORY</span>'
        f'<div class="rag-hero-title">Bạn muốn ôn tập phần nào?</div>'
        f'<div class="rag-hero-sub">Chọn một chủ đề mẫu hoặc nhập câu hỏi bên dưới — '
        f'mỗi câu trả lời đều kèm nguồn trang &amp; số liệu truy hồi.</div>'
        f'<div class="rag-hero-pills">'
        f'<span class="rag-hero-pill">{doc_count} Học liệu</span>'
        f'<span class="rag-hero-pill">{chunk_count:,} Chunks</span>'
        f'<span class="rag-hero-pill">BM25 + FAISS {embed_dim}d</span>'
        f'<span class="rag-hero-pill">{html.escape(rerank_pill)}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, card in enumerate(SUGGESTED_CARDS):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(
                    f'<div class="rag-sug-head">'
                    f'<span><strong>{html.escape(card["title"])}</strong></span>'
                    f'<span class="rag-sug-idx">0{i + 1}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.caption(card["subtitle"])
                if st.button(f":material/{card['icon']}: {card['prompt']}",
                             key=f"sug_{i}", width="stretch"):
                    # Không rerun: pending_prompt được tiêu ngay trong cùng lần chạy.
                    state["pending_prompt"] = card["prompt"]


def render_conversation_view(bundle):
    state = st.session_state
    conv = get_active_conversation(state)

    header_slot = st.empty()
    if conv["messages"]:
        with header_slot.container():
            st.markdown(f"### {conv['title']}")
            st.caption("Enter để gửi câu hỏi. Mỗi lượt lưu kèm độ trễ truy hồi, nguồn trang "
                       "và biến động hạng sau rerank.")

    _render_history(conv)

    if conv["messages"] and conv["messages"][-1]["role"] == "assistant":
        if st.button(":material/refresh: Tạo lại câu trả lời", key="regen"):
            q = prepare_regenerate(state)
            if q:
                state["pending_prompt"] = q
                # Lịch sử đã render cả lượt cũ; phải rerun để vẽ lại sau khi pop.
                st.rerun()

    empty_slot = st.empty()
    if not conv["messages"] and not state.get("pending_prompt"):
        with empty_slot.container():
            _render_empty_state(state, bundle)

    prompt = st.chat_input("Nhập câu hỏi của bạn...")
    if state.get("pending_prompt"):
        prompt = state.pop("pending_prompt")
    prompt = (prompt or "").strip()
    if not prompt:
        return

    empty_slot.empty()
    was_empty = not conv["messages"]
    started = time.perf_counter()
    append_user_turn(state, prompt)
    if was_empty:
        with header_slot.container():
            st.markdown(f"### {conv['title']}")
            st.caption("Enter để gửi câu hỏi. Mỗi lượt lưu kèm độ trễ truy hồi, nguồn trang "
                       "và biến động hạng sau rerank.")
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    keys = bundle["config"]["groq_keys"]
    served = {}
    with st.chat_message("assistant", avatar=AI_AVATAR):
        stage_slot = st.empty()
        stage_slot.markdown(
            '<div class="rag-stage-live">'
            '<span class="rag-stage-dot"></span>'
            '<span>Đang truy hồi Hybrid BM25 + FAISS &amp; chấm điểm Rerank...</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        res = inspect_retrieval(bundle, prompt)
        stage_slot.empty()

        ctx = format_selected_context(res["selected"])
        try:
            answer_slot = st.empty()
            with answer_slot.container():
                raw_answer = st.write_stream(stream_answer(
                    bundle["make_answer_chain"],
                    rotation_pairs(keys, model_chain(bundle["config"]["llm_model"])),
                    {"context": ctx, "question": prompt},
                    report=lambda model, key, fell_back: served.update(
                        model=model, key=key, fell_back=fell_back)))
            answer = latex_to_plain_text(raw_answer)
            if answer != (raw_answer or ""):
                answer_slot.markdown(answer)
            sources = sources_from_inspection(res["selected"]) if should_show_sources(answer) else []
            total_s = time.perf_counter() - started
            telemetry = _build_telemetry(bundle["config"], res, served, keys, total_s)
            st.markdown(render_turn_telemetry_html(telemetry), unsafe_allow_html=True)
            if sources:
                with st.expander(f"Nguồn trích dẫn & truy hồi ({len(sources)})", expanded=False):
                    st.markdown(render_citation_box_html(sources), unsafe_allow_html=True)
            append_assistant_turn(
                state, answer, sources,
                latency_s=total_s,
                telemetry=telemetry,
            )
        except Exception as exc:
            answer = describe_llm_error(exc)
            sources = []
            total_s = time.perf_counter() - started
            st.error(answer)
            append_assistant_turn(
                state, answer, sources,
                latency_s=total_s,
            )
            if st.button(":material/refresh: Thử lại", key="retry_after_error"):
                again = prepare_regenerate(state)
                if again:
                    state["pending_prompt"] = again
                    st.rerun()
