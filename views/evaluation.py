"""Evaluation view: Quality Gate / RAGAS report + live retrieval inspector."""

import streamlit as st

from rag_engine_web import inspect_retrieval
from ui.eval_schema import (
    classify_sample_passed,
    evaluate_gate,
    load_default_eval_data,
    parse_uploaded_eval,
)
from ui.theme import render_pastel_bar_chart_html

COLAB_SNIPPET = """# Chạy trong Colab SAU khi eval_rag.py tạo eval_report.json / eval_runs.jsonl
import json, urllib.request
from scripts.export_eval_to_web import build_web_eval_payload  # hoặc copy hàm này vào Colab

report = json.load(open("eval_report.json"))
runs = [json.loads(l) for l in open("eval_runs.jsonl")]
payload = build_web_eval_payload(report["metrics"], runs,
                                 baseline_metrics=baseline_report["metrics"])
json.dump(payload, open("eval_results.json", "w"), ensure_ascii=False, indent=2)
# Tải eval_results.json về rồi nạp vào tab này bằng file_uploader.
"""

KPI_KEYS = ["recall_at_k", "mrr", "faithfulness", "answer_relevancy"]


def _pct(v):
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "N/A"


def _gate_badge(gate):
    ok = gate["pass"]
    color = "#16A34A" if ok else "#DC2626"
    bg = "#DCFCE7" if ok else "#FEE2E2"
    label = "PASS" if ok else "FAIL"
    st.markdown(
        f"<div style='display:inline-block;background:{bg};color:{color};border-radius:999px;"
        f"padding:6px 18px;font-weight:700;letter-spacing:0.03em;'>QUALITY GATE: {label}</div>",
        unsafe_allow_html=True)


def _report_tab():
    up = st.file_uploader("Nạp kết quả eval từ Colab (.json / .jsonl / .csv)",
                          type=["json", "jsonl", "csv"])
    data = parse_uploaded_eval(up.getvalue(), up.name) if up else load_default_eval_data()

    if data.get("is_sample"):
        st.warning("Đây là SỐ LIỆU MINH HOẠ (is_sample=true), không phải kết quả thực nghiệm. "
                   "Chạy eval trong Colab rồi nạp file để thay thế.")
        with st.expander("Cách xuất kết quả thật từ Colab"):
            st.code(COLAB_SNIPPET, language="python")

    runs = data["runs"]
    primary = runs[data["primary_run"]]
    baseline = runs.get(data["baseline_run"]) if data.get("baseline_run") else None
    gate = evaluate_gate(primary["metrics"], data.get("thresholds"))
    _gate_badge(gate)
    st.caption(f"Run chính: {primary['label']}" + (f" · so sánh: {baseline['label']}" if baseline else ""))

    cols = st.columns(len(KPI_KEYS))
    for c, k in zip(cols, KPI_KEYS):
        pv = primary["metrics"].get(k)
        bv = baseline["metrics"].get(k) if baseline else None
        delta = None
        if isinstance(pv, (int, float)) and isinstance(bv, (int, float)):
            delta = f"{(pv - bv) * 100:+.1f} pt"
        with c:
            st.metric(k.replace("_", " "), _pct(pv), delta)

    st.markdown(render_pastel_bar_chart_html(runs, data.get("thresholds"), metric_keys=KPI_KEYS),
                unsafe_allow_html=True)

    st.subheader("Chi tiết từng câu")
    st.caption("Quy tắc: Recall@k == 0 hoặc Faithfulness < 0.85 hoặc Answer Relevancy < 0.70 → Cần xem lại.")
    filt = st.pills("Lọc", ["Tất cả", "Đạt", "Cần xem lại"], default="Tất cả", key="eval_filt")
    for s in data.get("samples", []):
        ok, reasons = classify_sample_passed(s)
        if filt == "Đạt" and not ok:
            continue
        if filt == "Cần xem lại" and ok:
            continue
        with st.container(border=True):
            st.markdown(f"**{s.get('question', '')}**")
            st.caption("ĐẠT" if ok else "CẦN XEM LẠI · " + "; ".join(reasons))
            mc = st.columns(3)
            for c, k in zip(mc, ["recall_at_k", "faithfulness", "answer_relevancy"]):
                with c:
                    st.metric(k.replace("_", " "), _pct(s.get(k)))


def _inspector_tab(bundle):
    presets = ["So sánh DFS và BFS", "PEAS là gì?", "GAN khác VAE thế nào?"]
    pc = st.columns(3)
    for c, p in zip(pc, presets):
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
            st.markdown(bundle["answer_chain"].invoke({"context": ctx, "question": q}))


def render_evaluation_view(bundle):
    tab1, tab2 = st.tabs([":material/monitoring: Báo cáo Quality Gate & RAGAS",
                          ":material/science: Live Pipeline Inspector"])
    with tab1:
        _report_tab()
    with tab2:
        _inspector_tab(bundle)