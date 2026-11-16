"""Gom output eval Colab (eval_report.json / eval_runs.jsonl / CSV) thành
``data/eval_results.json`` cho web ELC3008 — stdlib only, không side effect khi import.

Chạy trên Colab (sau khi eval_rag.py sinh output) hoặc local:

    python scripts/export_eval_to_web.py \
        --report eval_report.json --runs eval_runs.jsonl \
        --baseline-report eval_report_no_rerank.json \
        --output data/eval_results.json

Import ``build_web_eval_payload`` để dùng như thư viện trong notebook.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # chạy trực tiếp dạng script: thêm repo root vào sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.eval_schema import (  # noqa: E402
    DEFAULT_BASELINE_LABEL,
    DEFAULT_PRIMARY_LABEL,
    DEFAULT_THRESHOLDS,
    _aggregate_metrics,
    _clean_metrics,
    _normalize_sample,
    _now_iso,
    evaluate_gate,
    parse_uploaded_eval,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_web_eval_payload(
    primary_metrics: dict,
    samples: list[dict],
    baseline_metrics: dict | None = None,
    primary_label: str = DEFAULT_PRIMARY_LABEL,
    baseline_label: str = DEFAULT_BASELINE_LABEL,
) -> dict:
    """Payload web chuẩn: 2 run (primary + baseline tùy chọn), samples đã phân loại.

    ``hallucination_rate`` được dẫn xuất ``round(1 - faithfulness, 4)`` khi thiếu;
    mọi NaN/None được ép về None; mỗi sample gắn ``passed`` + ``reasons``.
    """
    normalized = [_normalize_sample(s) for s in (samples or [])]
    runs = {
        "primary": {"label": primary_label, "metrics": _clean_metrics(primary_metrics)},
    }
    baseline_run = None
    if baseline_metrics is not None:
        baseline_run = "baseline"
        runs["baseline"] = {"label": baseline_label, "metrics": _clean_metrics(baseline_metrics)}

    return {
        "is_sample": False,
        "evaluated_at": _now_iso(),
        "dataset_size": len(normalized),
        "primary_run": "primary",
        "baseline_run": baseline_run,
        "runs": runs,
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "samples": normalized,
    }


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Xuất eval_report.json / eval_runs.jsonl / CSV từ Colab sang data/eval_results.json cho web.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ví dụ (Colab):\n"
            "  python scripts/export_eval_to_web.py --report eval/eval_report.json \\\n"
            "      --runs eval/eval_runs.jsonl --output data/eval_results.json\n"
        ),
    )
    parser.add_argument("--report", type=Path, default=None, help="eval_report.json của run chính (Cohere Rerank)")
    parser.add_argument("--runs", type=Path, default=None, help="eval_runs.jsonl (từng câu hỏi) của run chính")
    parser.add_argument("--csv", type=Path, default=None, help="CSV điểm RAGAS từng câu (thay cho --runs)")
    parser.add_argument("--baseline-report", type=Path, default=None, help="eval_report.json của baseline (tắt Rerank)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "eval_results.json", help="File JSON đầu ra")
    parser.add_argument("--primary-label", default=DEFAULT_PRIMARY_LABEL)
    parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
    args = parser.parse_args(argv)

    if not (args.report or args.runs or args.csv):
        parser.error("Cần ít nhất một trong --report / --runs / --csv.")

    samples: list[dict] = []
    if args.runs:
        samples = parse_uploaded_eval(args.runs.read_bytes(), args.runs.name)["samples"]
    elif args.csv:
        samples = parse_uploaded_eval(args.csv.read_bytes(), args.csv.name)["samples"]

    report = None
    if args.report:
        report = _load_report(args.report)
        primary_metrics = _clean_metrics(report.get("metrics"))
    else:
        primary_metrics = _aggregate_metrics(samples)

    baseline_metrics = None
    if args.baseline_report:
        baseline_metrics = _clean_metrics(_load_report(args.baseline_report).get("metrics"))

    payload = build_web_eval_payload(
        primary_metrics,
        samples,
        baseline_metrics=baseline_metrics,
        primary_label=args.primary_label,
        baseline_label=args.baseline_label,
    )
    if report is not None and not samples:
        n = report.get("n")  # chỉ có report, không có per-question rows
        if isinstance(n, int) and n >= 0:
            payload["dataset_size"] = n

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    gate = evaluate_gate(payload["runs"][payload["primary_run"]]["metrics"], payload["thresholds"])
    verdict = "PASS" if gate["pass"] else "FAIL"
    print(
        f"Đã ghi {args.output} | dataset_size={payload['dataset_size']} | "
        f"samples={len(payload['samples'])} | Quality Gate: {verdict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())