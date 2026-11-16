"""Evaluation schema, Quality Gate và parser output Colab cho RAG ELC3008.

Logic thuần stdlib (json/csv/math/io/pathlib/datetime), dùng chung cho view
"Đánh giá RAG" và ``scripts/export_eval_to_web.py``. Khớp luật của
``eval_rag.py`` chạy trên Colab: retrieval metrics cấp trang nguồn
(Recall@k / Precision@k / MRR) + RAGAS generation metrics, Quality Gate với
``hallucination_rate`` là cận trên, các metric còn lại là cận dưới.

Mọi giá trị NaN / None / không hợp lệ đều được ép về ``None`` để gate và
biểu đồ hiển thị "N/A" thay vì crash (Review Focus #2 của plan).
"""

from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Hằng số khớp eval_rag.py (Colab)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "recall_at_k": 0.85,
    "mrr": 0.70,
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "hallucination_rate": 0.05,  # cận trên (op "<")
}

METRIC_ORDER = [
    "recall_at_k",
    "mrr",
    "precision_at_k",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "hallucination_rate",
]

PRIMARY_RUN_ID = "primary"
BASELINE_RUN_ID = "baseline"
DEFAULT_PRIMARY_LABEL = "Hybrid + Cohere Rerank (v3.0)"
DEFAULT_BASELINE_LABEL = "Hybrid (BM25 + FAISS, tắt Rerank)"

# Số chunk cuối cùng đưa vào context sau rerank (RERANK_TOP_K trong notebook).
RETRIEVAL_TOP_K = 5

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DATA_PATH = _REPO_ROOT / "data" / "eval_results.json"

_NULL_TOKENS = {"", "nan", "none", "null", "n/a", "na", "inf", "-inf", "+inf", "infinity"}


# ---------------------------------------------------------------------------
# Làm sạch số liệu
# ---------------------------------------------------------------------------

def _clean_num(val):
    """Ép ``val`` về float; None / "" / "NaN" / nan / inf / không số -> None."""
    if val is None:
        return None
    if isinstance(val, str):
        text = val.strip()
        if text.lower() in _NULL_TOKENS:
            return None
        try:
            val = float(text)
        except (TypeError, ValueError):
            return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _derive_hallucination_rate(metrics: dict) -> dict:
    """hallucination_rate = round(1 - faithfulness, 4) khi thiếu và faithfulness hợp lệ."""
    if metrics.get("hallucination_rate") is None:
        faith = metrics.get("faithfulness")
        if faith is not None:
            metrics["hallucination_rate"] = round(1.0 - faith, 4)
    return metrics


def _clean_metrics(metrics) -> dict:
    """Làm sạch toàn bộ giá trị metric (giữ None để render N/A) + dẫn xuất hallucination."""
    cleaned = {key: _clean_num(val) for key, val in (metrics or {}).items()}
    return _derive_hallucination_rate(cleaned)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Quality Gate & phân loại từng câu
# ---------------------------------------------------------------------------

def evaluate_gate(metrics: dict, thresholds: dict | None = None) -> dict:
    """PASS/FAIL từng metric + tổng, khớp ``quality_gate`` của eval_rag.py.

    - ``op`` là ">=" cho mọi metric, riêng ``hallucination_rate`` là "<".
    - Giá trị None (NaN/timeout) -> ``pass`` None và KHÔNG làm gate fail.
    - Gate pass = mọi check khác None đều pass.
    """
    bounds = dict(DEFAULT_THRESHOLDS)
    for key, val in (thresholds or {}).items():
        cleaned = _clean_num(val)
        if cleaned is not None:
            bounds[key] = cleaned

    metrics = metrics or {}
    checks = {}
    for key, bound in bounds.items():
        value = _clean_num(metrics.get(key))
        op = "<" if key == "hallucination_rate" else ">="
        if value is None:
            check_pass = None
        elif op == "<":
            check_pass = value < bound
        else:
            check_pass = value >= bound
        checks[key] = {"value": value, "bound": bound, "op": op, "pass": check_pass}

    gate_pass = all(c["pass"] for c in checks.values() if c["pass"] is not None)
    return {"pass": gate_pass, "checks": checks}


def classify_sample_passed(row: dict) -> tuple[bool, list[str]]:
    """Luật từng câu (khớp plan): FAIL khi Recall@k == 0, Faithfulness < 0.85
    hoặc Answer Relevancy < 0.70 (chỉ xét giá trị khác None)."""
    row = row or {}
    recall = _clean_num(row.get("recall_at_k"))
    faith = _clean_num(row.get("faithfulness"))
    relevancy = _clean_num(row.get("answer_relevancy"))

    reasons: list[str] = []
    if recall is not None and recall == 0.0:
        reasons.append("Recall@k = 0: không truy xuất được trang tài liệu vàng")
    if faith is not None and faith < 0.85:
        reasons.append(f"Faithfulness = {faith:.2f} < 0.85")
    if relevancy is not None and relevancy < 0.70:
        reasons.append(f"Answer Relevancy = {relevancy:.2f} < 0.70")
    return (not reasons, reasons)


# ---------------------------------------------------------------------------
# Chuẩn hóa sample & aggregate
# ---------------------------------------------------------------------------

def _normalize_sample(row) -> dict:
    """Chuẩn hóa 1 dòng (CSV / JSONL / web sample) về schema chung của web.

    Chấp nhận metric phẳng hoặc lồng trong ``"metrics"``; alias RAGAS
    (``user_input`` / ``response`` / ``reference_answer`` / ``ground_truth``).
    """
    row = dict(row or {})
    nested = row.pop("metrics", None)
    if isinstance(nested, dict):
        for key, val in nested.items():
            row.setdefault(key, val)

    sample = {
        "question": str(row.get("question") or row.get("user_input") or "").strip(),
        "reference": str(
            row.get("reference") or row.get("reference_answer") or row.get("ground_truth") or ""
        ).strip(),
        "answer": str(row.get("answer") or row.get("response") or "").strip(),
    }
    for key in METRIC_ORDER:
        if key in row:
            sample[key] = _clean_num(row[key])
    _derive_hallucination_rate(sample)

    # Giữ vết trang nguồn (nhẹ, hữu ích cho đối chiếu) nếu có; bỏ contexts nặng.
    for key in ("gold_doc_id", "gold_pages", "retrieved_pages"):
        if row.get(key) is not None:
            sample[key] = row[key]

    passed, reasons = classify_sample_passed(sample)
    sample["passed"] = passed
    sample["reasons"] = reasons
    return sample


def _aggregate_metrics(samples: list[dict]) -> dict:
    """Trung bình metric trên các câu (bỏ qua None), dẫn xuất hallucination_rate."""
    out: dict = {}
    for key in METRIC_ORDER:
        vals = [v for v in (_clean_num(s.get(key)) for s in samples) if v is not None]
        if vals:
            out[key] = round(sum(vals) / len(vals), 4)
    return _derive_hallucination_rate(out)


def _page_retrieval_metrics(gold_pages, retrieved_pages, k: int = RETRIEVAL_TOP_K) -> dict:
    """Recall@k / Precision@k / MRR cấp trang — khớp ``retrieval_metrics`` eval_rag.py."""
    gold = {str(p).strip() for p in (gold_pages or []) if p is not None}
    topk = [str(p).strip() if p is not None else None for p in (retrieved_pages or [])][:k]
    relevant_count = sum(1 for p in topk if p is not None and p in gold)
    covered = gold & {p for p in topk if p is not None}
    recall = len(covered) / len(gold) if gold else 0.0
    precision = relevant_count / len(topk) if topk else 0.0
    rr = 0.0
    for rank, page in enumerate(topk, 1):
        if page is not None and page in gold:
            rr = 1.0 / rank
            break
    return {
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "mrr": round(rr, 4),
    }


def _base_payload(samples: list[dict], primary_metrics: dict) -> dict:
    return {
        "is_sample": False,
        "evaluated_at": _now_iso(),
        "dataset_size": len(samples),
        "primary_run": PRIMARY_RUN_ID,
        "baseline_run": None,
        "runs": {
            PRIMARY_RUN_ID: {"label": DEFAULT_PRIMARY_LABEL, "metrics": primary_metrics},
        },
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Parser 4 định dạng upload
# ---------------------------------------------------------------------------

def _normalize_web_payload(data: dict) -> dict:
    """Chuẩn hóa payload web đầy đủ (eval_results.json) — làm sạch NaN, giữ is_sample."""
    runs = {}
    for run_id, run in (data.get("runs") or {}).items():
        run = run or {}
        runs[str(run_id)] = {
            "label": str(run.get("label") or run_id),
            "metrics": _clean_metrics(run.get("metrics")),
        }
    samples = [_normalize_sample(s) for s in (data.get("samples") or [])]

    primary = data.get("primary_run")
    primary = str(primary) if primary is not None else next(iter(runs), None)
    baseline = data.get("baseline_run")
    baseline = str(baseline) if baseline is not None else None

    thresholds = dict(DEFAULT_THRESHOLDS)
    for key, val in (data.get("thresholds") or {}).items():
        cleaned = _clean_num(val)
        if cleaned is not None:
            thresholds[key] = cleaned

    size = _clean_num(data.get("dataset_size"))
    return {
        "is_sample": bool(data.get("is_sample", False)),
        "evaluated_at": str(data.get("evaluated_at") or _now_iso()),
        "dataset_size": int(size) if size is not None else len(samples),
        "primary_run": primary,
        "baseline_run": baseline,
        "runs": runs,
        "thresholds": thresholds,
        "samples": samples,
    }


def _parse_colab_report(data: dict) -> dict:
    """eval_report.json của Colab: {"metrics": ..., "gate": ..., "n": ...}."""
    payload = _base_payload([], _clean_metrics(data.get("metrics")))
    n = _clean_num(data.get("n"))
    payload["dataset_size"] = int(n) if n is not None else 0
    if data.get("evaluated_at"):
        payload["evaluated_at"] = str(data["evaluated_at"])
    return payload


def _parse_jsonl(raw_bytes: bytes) -> dict:
    """eval_runs.jsonl: mỗi dòng 1 câu; tính retrieval metrics khi có gold/retrieved pages."""
    samples = []
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        if row.get("gold_pages") is not None and row.get("retrieved_pages") is not None:
            computed = _page_retrieval_metrics(row["gold_pages"], row["retrieved_pages"])
            for key, val in computed.items():
                if _clean_num(row.get(key)) is None:
                    row[key] = val
        samples.append(_normalize_sample(row))
    return _base_payload(samples, _aggregate_metrics(samples))


def _parse_csv(raw_bytes: bytes) -> dict:
    """CSV điểm RAGAS (result.to_pandas().to_csv()) — NaN được ép về None."""
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    samples = []
    for row in csv.DictReader(io.StringIO(text)):
        if not any(isinstance(v, str) and v.strip() for v in row.values()):
            continue  # bỏ dòng trống hoàn toàn
        samples.append(_normalize_sample(row))
    return _base_payload(samples, _aggregate_metrics(samples))


def parse_uploaded_eval(raw_bytes: bytes, filename: str) -> dict:
    """Parse file upload (.json / .jsonl / .csv) về payload web chuẩn hóa.

    Hỗ trợ 4 định dạng:
      1. ``eval_results.json`` web đầy đủ (runs + samples) — kể cả file seed.
      2. ``eval_report.json`` Colab ({"metrics", "gate", "n"}).
      3. ``eval_runs.jsonl`` Colab (tính Recall@k/Precision@k/MRR cấp trang).
      4. ``.csv`` điểm RAGAS từng câu (có thể chứa NaN).
    """
    if isinstance(raw_bytes, str):
        raw_bytes = raw_bytes.encode("utf-8")
    name = (filename or "").lower().strip()

    if name.endswith(".csv"):
        return _parse_csv(raw_bytes)
    if name.endswith(".jsonl"):
        return _parse_jsonl(raw_bytes)

    data = json.loads(raw_bytes.decode("utf-8-sig", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("JSON không hợp lệ: mong đợi một object (eval_results.json hoặc eval_report.json).")
    if isinstance(data.get("runs"), dict):
        return _normalize_web_payload(data)
    if "metrics" in data:
        return _parse_colab_report(data)
    raise ValueError("JSON không nhận diện được: cần khóa 'runs' (web) hoặc 'metrics' (Colab eval_report.json).")


def load_default_eval_data(path: Path | None = None) -> dict:
    """Đọc ``data/eval_results.json`` mặc định (seed is_sample=true) và chuẩn hóa."""
    target = Path(path) if path is not None else DEFAULT_EVAL_DATA_PATH
    with target.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return _normalize_web_payload(data)