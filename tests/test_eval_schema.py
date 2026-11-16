import json
import math
import unittest
from pathlib import Path
from ui.eval_schema import (
    DEFAULT_THRESHOLDS,
    classify_sample_passed,
    evaluate_gate,
    load_default_eval_data,
    parse_uploaded_eval,
)
from scripts.export_eval_to_web import build_web_eval_payload


class TestEvalSchema(unittest.TestCase):
    def test_gate_matches_colab_eval_rag_rules_and_handles_nan(self):
        metrics = {
            "recall_at_k": 0.88,
            "mrr": 0.75,
            "faithfulness": 0.96,
            "answer_relevancy": float("nan"),
            "hallucination_rate": 0.04,
        }
        gate = evaluate_gate(metrics)
        self.assertTrue(gate["pass"])
        self.assertIsNone(gate["checks"]["answer_relevancy"]["value"])
        self.assertIsNone(gate["checks"]["answer_relevancy"]["pass"])

        failing = dict(metrics, hallucination_rate=0.08)
        self.assertFalse(evaluate_gate(failing)["pass"])

    def test_classify_sample_passed_flags_zero_recall_low_faithfulness_or_relevancy(self):
        ok, reasons = classify_sample_passed({"recall_at_k": 1.0, "faithfulness": 0.9, "answer_relevancy": 0.82})
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

        ok2, reasons2 = classify_sample_passed({"recall_at_k": 0.0, "faithfulness": 0.92, "answer_relevancy": 0.85})
        self.assertFalse(ok2)
        self.assertTrue(any("Recall" in r for r in reasons2))

        ok3, _ = classify_sample_passed({"recall_at_k": None, "faithfulness": float("nan"), "answer_relevancy": 0.65})
        self.assertFalse(ok3)

    def test_parse_uploaded_colab_eval_report_json_and_csv(self):
        colab_report = json.dumps({
            "n": 24,
            "metrics": {"recall_at_k": 0.79, "mrr": 0.68, "faithfulness": 0.88, "answer_relevancy": 0.83, "hallucination_rate": 0.12},
        }).encode("utf-8")
        parsed = parse_uploaded_eval(colab_report, "eval_report.json")
        self.assertFalse(parsed["is_sample"])
        self.assertEqual(parsed["dataset_size"], 24)
        self.assertAlmostEqual(parsed["runs"][parsed["primary_run"]]["metrics"]["recall_at_k"], 0.79)

        csv_bytes = "question,faithfulness,answer_relevancy,recall_at_k\nDFS la gi?,0.92,0.85,1.0\nBFS?,NaN,0.60,0.0\n".encode("utf-8")
        parsed_csv = parse_uploaded_eval(csv_bytes, "ragas_scores.csv")
        self.assertEqual(len(parsed_csv["samples"]), 2)
        self.assertTrue(parsed_csv["samples"][0]["passed"])
        self.assertFalse(parsed_csv["samples"][1]["passed"])

    def test_default_seed_file_is_marked_is_sample_true(self):
        data = load_default_eval_data()
        self.assertTrue(data["is_sample"])
        self.assertIn("runs", data)
        self.assertGreaterEqual(len(data["samples"]), 5)

    def test_exporter_builds_valid_two_run_payload(self):
        payload = build_web_eval_payload(
            primary_metrics={"recall_at_k": 0.875, "mrr": 0.76, "faithfulness": 0.91, "answer_relevancy": 0.86},
            baseline_metrics={"recall_at_k": 0.708, "mrr": 0.59, "faithfulness": 0.86, "answer_relevancy": 0.81},
            samples=[{"question": "Q1", "reference": "R1", "answer": "A1", "recall_at_k": 1.0, "faithfulness": 0.95}],
        )
        self.assertFalse(payload["is_sample"])
        self.assertIsNotNone(payload["baseline_run"])
        self.assertAlmostEqual(payload["runs"][payload["primary_run"]]["metrics"]["hallucination_rate"], 0.09)