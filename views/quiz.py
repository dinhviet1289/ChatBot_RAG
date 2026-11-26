"""Luyện tập trắc nghiệm (Mini-Quiz) — sinh câu hỏi A/B/C/D bám sát học liệu trong FAISS."""

import html
import logging
import uuid

import streamlit as st

from rag_engine_web import (
    build_quiz_chain,
    describe_llm_error,
    invoke_text,
    model_chain,
    rotation_pairs,
)
from ui.catalog import CATEGORY_ORDER
from ui.quiz import (
    QUIZ_TOPICS,
    QuizParseError,
    format_quiz_context,
    grade_quiz,
    parse_quiz_response,
    sample_quiz_chunks,
)
from ui.theme import (
    normalize_latex_for_streamlit,
    render_flashcards_html,
    render_page_header_html,
)

logger = logging.getLogger(__name__)
ALL_CATEGORIES_LABEL = "Tất cả (44 học liệu)"


def _queue_quiz_preset(category, topic):
    st.session_state["quiz_cat"] = category
    st.session_state["quiz_topic"] = topic
    st.session_state["quiz_auto"] = True


def _reset_quiz_attempt():
    sess = st.session_state.get("quiz_session")
    if sess:
        sess["graded"] = None
        sess["attempt"] = int(sess.get("attempt", 0)) + 1


def _request_new_quiz():
    st.session_state["quiz_auto"] = True


def _source_link_html(q):
    title_esc = html.escape(str(q.get("doc_title") or "Tài liệu"))
    trang_esc = html.escape(str(q.get("trang") or "?"))
    url = q.get("url")
    label = f"{title_esc} · Trang {trang_esc}"
    if isinstance(url, str) and url.startswith("https://"):
        url_esc = html.escape(url, quote=True)
        return (
            f'<a class="rag-source-pill" href="{url_esc}" target="_blank" '
            f'rel="noopener noreferrer"><span class="rag-source-name">{label}</span></a>'
        )
    return f'<span class="rag-source-pill"><span class="rag-source-name">{label}</span></span>'


def render_quiz_view(bundle):
    cfg = (bundle or {}).get("config") or {}
    db = (bundle or {}).get("db")
    docstore = getattr(getattr(db, "docstore", None), "_dict", {}) if db is not None else {}

    st.markdown(
        render_page_header_html(
            "Trắc nghiệm ôn tập (Mini-Quiz)",
            "Bốc trực tiếp đoạn trích từ 44 học liệu để tạo câu hỏi A/B/C/D kèm giải thích và link trang PDF gốc.",
            [
                (str(cfg.get("llm_model") or "Groq").split("/")[-1], "primary"),
                ("Grounded PDF Quiz", "info"),
                ("Chấm điểm tức thì", "success"),
            ],
        ),
        unsafe_allow_html=True,
    )

    cat_options = [ALL_CATEGORIES_LABEL] + list(CATEGORY_ORDER)
    st.session_state.setdefault("quiz_cat", "Bài giảng ELC3008")
    if st.session_state["quiz_cat"] not in cat_options:
        st.session_state["quiz_cat"] = "Bài giảng ELC3008"
    st.session_state.setdefault("quiz_topic", "")
    st.session_state.setdefault("quiz_num", 5)

    auto_gen = bool(st.session_state.pop("quiz_auto", False))

    with st.container(border=True):
        c1, c2, c3 = st.columns([1.95, 1.95, 0.75], vertical_alignment="bottom")
        with c1:
            sel_cat = st.selectbox("Nhóm học phần", cat_options, key="quiz_cat")
        with c2:
            topic_q = st.text_input(
                "Chủ đề trọng tâm",
                key="quiz_topic",
                placeholder="VD: PEAS, DFS vs BFS, Minimax, SVM...",
            )
        with c3:
            num_q = st.selectbox("Số câu", [3, 5], key="quiz_num")

        gen_clicked = st.button(
            ":material/auto_awesome: Tạo bộ câu hỏi mới",
            key="quiz_generate",
            type="primary",
            width="stretch",
        )
        st.caption("Chọn nhanh chủ đề ôn thi")
        p_cols = st.columns(len(QUIZ_TOPICS))
        for idx, (label, p_cat, p_topic) in enumerate(QUIZ_TOPICS):
            p_cols[idx].button(
                label,
                key=f"quiz_preset_{idx}",
                width="stretch",
                on_click=_queue_quiz_preset,
                args=(p_cat, p_topic),
            )

    if gen_clicked or auto_gen:
        sampled = sample_quiz_chunks(
            docstore,
            category=sel_cat,
            topic=topic_q,
            max_chunks=max(5, int(num_q) + 1),
        )
        if not sampled:
            st.warning("Chưa tìm thấy đoạn trích phù hợp trong nhóm học liệu đã chọn.")
            return

        keys = (
            bundle.get("groq_keys")
            if "groq_keys" in (bundle or {})
            else (cfg.get("groq_keys") or [])
        )
        make_quiz = (
            build_quiz_chain
            if db is not None
            else ((bundle or {}).get("make_quiz_chain") or (bundle or {}).get("make_answer_chain") or build_quiz_chain)
        )
        pairs = rotation_pairs(keys, model_chain(cfg.get("llm_model") or ""))
        served = {}
        topic_hint = (topic_q or "").strip() or sel_cat
        quiz_inputs = {
            "num_questions": int(num_q),
            "topic_hint": topic_hint,
            "context": format_quiz_context(sampled),
        }
        questions = None
        best_questions = []
        best_served = {}
        with st.spinner("Đang tổng hợp bộ câu hỏi bám sát giáo trình..."):
            try:
                attempt_pairs = list(pairs)
                for attempt_idx in range(2):
                    if not attempt_pairs:
                        break
                    served.clear()
                    raw = invoke_text(
                        make_quiz,
                        attempt_pairs,
                        quiz_inputs,
                        report=lambda model, key, fell_back: served.update(
                            model=model, key=key, fell_back=fell_back
                        ),
                    )
                    try:
                        cand = parse_quiz_response(raw, sampled, expected_count=int(num_q))
                    except QuizParseError as parse_exc:
                        logger.warning(
                            "Quiz JSON parse thất bại (attempt %d, model=%s): %s",
                            attempt_idx + 1,
                            served.get("model"),
                            parse_exc,
                        )
                        if attempt_idx == 1 and not best_questions:
                            raise
                        attempt_pairs = [
                            (m, k) for m, k in attempt_pairs if m != served.get("model")
                        ] or attempt_pairs[1:]
                        continue

                    if len(cand) > len(best_questions):
                        best_questions = cand
                        best_served = dict(served)
                    if len(best_questions) >= int(num_q) or attempt_idx == 1:
                        break
                    attempt_pairs = [
                        (m, k) for m, k in attempt_pairs if m != served.get("model")
                    ] or attempt_pairs[1:]

                questions = best_questions
                if best_served:
                    served = best_served
            except QuizParseError:
                st.error("Mô hình trả về định dạng JSON chưa chuẩn sau khi thử lại. Bấm **Tạo bộ câu hỏi mới** hoặc chọn chủ đề khác.")
                return
            except Exception as exc:
                st.error(describe_llm_error(exc))
                return

        st.session_state["quiz_session"] = {
            "id": uuid.uuid4().hex[:6],
            "attempt": 0,
            "category": sel_cat,
            "topic": (topic_q or "").strip(),
            "model": served.get("model") or cfg.get("llm_model") or "",
            "questions": questions,
            "graded": None,
        }

    sess = st.session_state.get("quiz_session")
    if not sess:
        st.info("Chọn nhóm học phần hoặc một chủ đề ôn thi phía trên rồi bấm **Tạo bộ câu hỏi mới** để bắt đầu.")
        return

    questions = sess["questions"]
    quiz_id = sess["id"]
    attempt = int(sess.get("attempt", 0))
    meta_parts = [f"Phạm vi: {sess['category']}"]
    if sess.get("topic"):
        meta_parts.append(f"Chủ đề: {sess['topic']}")
    if sess.get("model"):
        meta_parts.append(f"Model: {sess['model']}")
    meta_parts.append(f"{len(questions)} câu hỏi")
    st.markdown(
        '<div class="rag-telemetry-strip rag-mono">'
        + "".join(f'<span class="rag-tel-chip">{html.escape(part)}</span>' for part in meta_parts)
        + "</div>",
        unsafe_allow_html=True,
    )

    mode_quiz = "Làm bài trắc nghiệm"
    mode_flash = "Lật thẻ Flashcard 3D"
    st.session_state.setdefault("quiz_view_mode", mode_quiz)
    if hasattr(st, "segmented_control"):
        view_mode = (
            st.segmented_control(
                "Chế độ ôn tập",
                [mode_quiz, mode_flash],
                key="quiz_view_mode",
                label_visibility="collapsed",
            )
            or mode_quiz
        )
    else:
        view_mode = st.radio(
            "Chế độ ôn tập",
            [mode_quiz, mode_flash],
            key="quiz_view_mode",
            horizontal=True,
            label_visibility="collapsed",
        )

    if view_mode == mode_flash:
        st.caption(
            "Nhấn trực tiếp vào từng thẻ để lật xem đáp án & giải thích · Bấm link bên dưới thẻ để mở trang PDF gốc."
        )
        st.markdown(render_flashcards_html(questions), unsafe_allow_html=True)
        st.button(
            ":material/auto_awesome: Sinh bộ thẻ mới",
            key="fc_regen",
            type="primary",
            width="stretch",
            on_click=_request_new_quiz,
        )
        return

    graded = sess.get("graded")
    if graded is None:
        with st.form(f"quiz_answer_form_{quiz_id}_{attempt}", border=False):
            selected_answers = {}
            for idx, q in enumerate(questions):
                with st.container(border=True):
                    doc_title_esc = html.escape(str(q.get("doc_title") or "Tài liệu"))
                    trang_esc = html.escape(str(q.get("trang") or "?"))
                    st.markdown(
                        f'<div class="rag-quiz-qhead">'
                        f'<span class="rag-rank-badge rag-mono">Câu {idx + 1}/{len(questions)}</span>'
                        f'<span class="rag-quiz-src rag-mono">{doc_title_esc} · Trang {trang_esc}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(normalize_latex_for_streamlit(q["question"]))
                    selected_answers[idx] = st.radio(
                        f"Đáp án câu {idx + 1}",
                        ["A", "B", "C", "D"],
                        index=None,
                        format_func=lambda letter, opts=q["options"]: f"{letter}. {opts[letter]}",
                        key=f"quiz_ans_{quiz_id}_{attempt}_{idx}",
                        label_visibility="collapsed",
                    )
            submitted = st.form_submit_button(
                ":material/check_circle: Chấm điểm ngay",
                type="primary",
                width="stretch",
            )
        if submitted:
            graded_res = grade_quiz(questions, selected_answers)
            sess["graded"] = graded_res
            if graded_res["pct"] >= 80.0:
                sess["celebrate_now"] = True
            st.rerun()
        return

    if sess.pop("celebrate_now", False):
        st.balloons()

    score = graded["score"]
    total = graded["total"]
    pct = graded["pct"]
    tone = "is-ok" if pct >= 80.0 else ("is-warn" if pct >= 50.0 else "is-danger")
    score_note = (
        "Xuất sắc! Bạn nắm rất chắc phần học liệu này · Đối chiếu chi tiết bên dưới"
        if pct >= 80.0
        else (
            "Khá tốt! Xem lại các câu sai và trang tài liệu gốc bên dưới"
            if pct >= 50.0
            else "Cần ôn kỹ thêm! Đọc giải thích và mở trang PDF gốc bên dưới"
        )
    )
    bar_w = max(2.0, min(100.0, pct))
    st.markdown(
        f'<div class="rag-quiz-score {tone}">'
        f'<div class="rag-quiz-score-top">'
        f'<div><div class="rag-lab-k">KẾT QUẢ BÀI LÀM</div>'
        f'<div class="rag-quiz-score-num rag-mono">{score} / {total} câu đúng ({pct:.0f}%)</div></div>'
        f'<div class="rag-quiz-score-note">{score_note}</div>'
        f'</div>'
        f'<div class="rag-doc-bar"><span style="width:{bar_w:.1f}%;"></span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for res in graded["results"]:
        idx = res["index"]
        q = questions[idx]
        chosen = res["chosen"]
        correct = res["correct"]
        is_ok = res["is_correct"]
        with st.container(border=True):
            status_cls = "is-ok" if is_ok else "is-wrong"
            status_txt = "ĐÚNG" if is_ok else ("CHƯA CHỌN" if not chosen else f"SAI (chọn {chosen})")
            st.markdown(
                f'<div class="rag-quiz-qhead">'
                f'<span class="rag-rank-badge rag-mono">Câu {idx + 1}/{total}</span>'
                f'<span class="rag-quiz-status {status_cls} rag-mono">{html.escape(status_txt)}</span>'
                f'<span class="rag-quiz-ans rag-mono">Đáp án: {html.escape(correct)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(normalize_latex_for_streamlit(q["question"]))
            st.markdown(
                f"**Đáp án đúng ({correct}):** {normalize_latex_for_streamlit(q['options'][correct])}"
            )
            if not is_ok and chosen and chosen in q["options"]:
                st.caption(f"Bạn đã chọn ({chosen}): {q['options'][chosen]}")
            if q.get("explanation"):
                st.markdown(
                    f'<div class="rag-quiz-fb {status_cls}">'
                    f'{html.escape(q["explanation"])}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(_source_link_html(q), unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    b1.button(":material/refresh: Làm lại đề này", key="quiz_retry",
              width="stretch", on_click=_reset_quiz_attempt)
    b2.button(":material/auto_awesome: Sinh đề mới", key="quiz_regen",
              type="primary", width="stretch", on_click=_request_new_quiz)
