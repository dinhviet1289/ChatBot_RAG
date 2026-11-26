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
    build_followup_questions,
    get_active_conversation,
    last_user_question,
    open_in_inspector,
    prepare_regenerate,
)
from ui.theme import (
    normalize_latex_for_streamlit,
    render_citation_box_html,
    render_page_header_html,
    render_source_pills_html,
    render_turn_telemetry_html,
)

USER_AVATAR = ":material/person:"
AI_AVATAR = ":material/school:"


def _on_inspect_last():
    q = last_user_question(st.session_state)
    if q:
        open_in_inspector(st.session_state, q)


def _on_followup(question):
    st.session_state["pending_prompt"] = question


def _clear_stage_on_first_chunk(stream, stage_slot):
    cleared = False
    for chunk in stream:
        if not cleared:
            stage_slot.empty()
            cleared = True
        yield chunk
    if not cleared:
        stage_slot.empty()


def _build_telemetry(cfg, res, served, keys, latency_s):
    key_slot = ""
    if served and len(keys) > 1 and served.get("key") in keys:
        key_slot = f"{keys.index(served['key']) + 1}/{len(keys)}"
    return {
        "latency_s": round(float(latency_s), 2),
        "retrieve_ms": res.get("retrieve_ms"),
        "rerank_ms": res.get("rerank_ms"),
        "rerank_enabled": bool(res.get("rerank_enabled")),
        "rerank_fallback": bool(res.get("rerank_fallback")),
        "selected_count": len(res.get("selected") or []),
        "total_candidates": int(res.get("total_candidates") or 0),
        "model": (served or {}).get("model") or cfg.get("llm_model") or "",
        "key_slot": key_slot,
        "fell_back": bool((served or {}).get("fell_back")),
        "configured_model": cfg.get("llm_model") or "",
    }


def _render_conv_header(conv, bundle):
    cfg = (bundle or {}).get("config") or {}
    embed_dim = int(cfg.get("embed_dim") or 384)
    provider = str(cfg.get("rerank_provider") or "off").strip().lower()
    rerank_chip = ("Cohere Rerank", "success") if provider == "cohere" else (
        ("Local Rerank", "success") if provider == "local" else ("Stage-1 Only", "warn")
    )
    llm_short = str(cfg.get("llm_model") or "Groq").split("/")[-1]
    chips = [
        (llm_short, "primary"),
        rerank_chip,
        (f"BM25 + FAISS {embed_dim}d", "info"),
    ]
    st.markdown(
        render_page_header_html(
            conv["title"],
            "Enter để gửi câu hỏi · Mỗi lượt lưu kèm độ trễ truy hồi, nguồn trang và biến động hạng sau rerank.",
            chips,
        ),
        unsafe_allow_html=True,
    )


def _render_history(conv):
    for msg in conv["messages"]:
        avatar = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant" and msg.get("is_error"):
                st.error(msg["content"])
                continue
            content = (normalize_latex_for_streamlit(msg["content"])
                       if msg["role"] == "assistant" else msg["content"])
            st.markdown(content)
            if msg["role"] == "assistant":
                if msg.get("telemetry"):
                    st.markdown(render_turn_telemetry_html(msg["telemetry"]),
                                unsafe_allow_html=True)
                if msg.get("sources"):
                    st.markdown(render_source_pills_html(msg["sources"]),
                                unsafe_allow_html=True)
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
        f'<span class="rag-hero-badge">ELC3008 · TRỢ LÝ ÔN TẬP AI</span>'
        f'<div class="rag-hero-title">Bạn muốn ôn tập phần nào?</div>'
        f'<div class="rag-hero-sub">Chọn một chủ đề mẫu hoặc nhập câu hỏi bên dưới — '
        f'mỗi câu trả lời đều kèm nguồn trang &amp; số liệu truy hồi.</div>'
        f'<div class="rag-hero-pills">'
        f'<span class="rag-hero-pill">{doc_count} Học liệu</span>'
        f'<span class="rag-hero-pill">{chunk_count:,} Chunks</span>'
        f'<span class="rag-hero-pill">BM25 + FAISS {embed_dim}d</span>'
        f'<span class="rag-hero-pill">{html.escape(rerank_pill)}</span>'
        f'</div>'
        f'<div class="rag-hero-flow">'
        f'<span class="rag-flow-step">1. Hybrid BM25 + FAISS</span>'
        f'<span class="rag-flow-arrow" aria-hidden="true">→</span>'
        f'<span class="rag-flow-step">2. Cross-Encoder Rerank</span>'
        f'<span class="rag-flow-arrow" aria-hidden="true">→</span>'
        f'<span class="rag-flow-step">3. Trích nguồn &amp; trang gốc</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, card in enumerate(SUGGESTED_CARDS):
        with cols[i % 2]:
            if st.button(f":material/{card['icon']}: **{card['title']}** · {card['subtitle']}\n\n{card['prompt']} →",
                         key=f"sug_{i}", width="stretch"):
                # Không rerun: pending_prompt được tiêu ngay trong cùng lần chạy.
                state["pending_prompt"] = card["prompt"]


def render_conversation_view(bundle):
    state = st.session_state
    conv = get_active_conversation(state)

    header_slot = st.empty()
    if conv["messages"]:
        with header_slot.container():
            _render_conv_header(conv, bundle)

    _render_history(conv)

    regen_slot = st.empty()
    if conv["messages"] and conv["messages"][-1]["role"] == "assistant":
        last_msg = conv["messages"][-1]
        last_q = last_user_question(state)
        history_qs = [m.get("content") for m in conv["messages"] if m.get("role") == "user"]
        with regen_slot.container():
            act1, act2 = st.columns([1, 1.15])
            if act1.button(":material/refresh: Tạo lại câu trả lời", key="regen", width="stretch"):
                q = prepare_regenerate(state)
                if q:
                    state["pending_prompt"] = q
                    # Lịch sử đã render cả lượt cũ; phải rerun để vẽ lại sau khi pop.
                    st.rerun()
            if last_q:
                act2.button(
                    ":material/science: Mổ xẻ truy hồi câu này",
                    key="inspect_last",
                    width="stretch",
                    on_click=_on_inspect_last,
                )
            if not last_msg.get("is_error"):
                followups = build_followup_questions(
                    last_q or "",
                    last_msg.get("sources"),
                    history_qs,
                    last_msg.get("content") or "",
                )
                if followups:
                    st.markdown(
                        '<div class="rag-followup-label rag-mono">GỢI Ý HỎI TIẾP</div>',
                        unsafe_allow_html=True,
                    )
                    for idx, fq in enumerate(followups):
                        st.button(
                            f":material/subdirectory_arrow_right: {fq}",
                            key=f"fup_{conv['id']}_{len(conv['messages'])}_{idx}",
                            width="stretch",
                            on_click=_on_followup,
                            args=(fq,),
                        )

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
    regen_slot.empty()
    was_empty = not conv["messages"]
    started = time.perf_counter()
    append_user_turn(state, prompt)
    if was_empty:
        with header_slot.container():
            _render_conv_header(conv, bundle)
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    keys = bundle.get("groq_keys") if "groq_keys" in bundle else (bundle.get("config") or {}).get("groq_keys") or []
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
        try:
            res = inspect_retrieval(bundle, prompt)
            stage_ms = float(res.get("retrieve_ms") or 0.0) + (
                float(res.get("rerank_ms") or 0.0) if res.get("rerank_enabled") else 0.0
            )
            stage_slot.markdown(
                f'<div class="rag-stage-live is-synth">'
                f'<span class="rag-stage-dot"></span>'
                f'<span>Đã chọn {len(res.get("selected") or [])} đoạn trích '
                f'({stage_ms:.0f} ms) · Đang tổng hợp câu trả lời...</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            ctx = format_selected_context(res["selected"])
            answer_slot = st.empty()
            with answer_slot.container():
                raw_answer = st.write_stream(_clear_stage_on_first_chunk(
                    stream_answer(
                        bundle["make_answer_chain"],
                        rotation_pairs(keys, model_chain(bundle["config"]["llm_model"])),
                        {"context": ctx, "question": prompt},
                        report=lambda model, key, fell_back: served.update(
                            model=model, key=key, fell_back=fell_back)),
                    stage_slot,
                ))
            answer = normalize_latex_for_streamlit(raw_answer)
            if answer != (raw_answer or ""):
                answer_slot.markdown(answer)
            sources = sources_from_inspection(res["selected"]) if should_show_sources(answer) else []
            total_s = time.perf_counter() - started
            telemetry = _build_telemetry(bundle["config"], res, served, keys, total_s)
            st.markdown(render_turn_telemetry_html(telemetry), unsafe_allow_html=True)
            if sources:
                st.markdown(render_source_pills_html(sources), unsafe_allow_html=True)
                with st.expander(f"Nguồn trích dẫn & truy hồi ({len(sources)})", expanded=False):
                    st.markdown(render_citation_box_html(sources), unsafe_allow_html=True)
            append_assistant_turn(
                state, answer, sources,
                latency_s=total_s,
                telemetry=telemetry,
            )
            st.rerun()
        except Exception as exc:
            stage_slot.empty()
            answer = describe_llm_error(exc)
            total_s = time.perf_counter() - started
            st.error(answer)
            append_assistant_turn(
                state, answer, [],
                latency_s=total_s,
                is_error=True,
            )
            st.rerun()
