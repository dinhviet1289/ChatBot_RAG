import random
import unittest

from rag_engine_web import invoke_text
from ui.quiz import (
    QuizParseError,
    format_quiz_context,
    grade_quiz,
    parse_quiz_response,
    sample_quiz_chunks,
)
from ui.state import build_followup_questions


class FakeDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = dict(metadata)


class RateLimitError(Exception):
    pass


class TestQuizAndFollowups(unittest.TestCase):
    def test_sample_quiz_chunks_filters_category_prioritizes_topic_and_links_drive(self):
        long_peas = "<!-- page: 5 --> Mô hình PEAS gồm Performance, Environment, Actuators, Sensors. " * 6
        long_search = "Thuật toán tìm kiếm BFS dùng hàng đợi FIFO còn DFS dùng ngăn xếp LIFO. " * 6
        long_other = "Cây quyết định phân tách thuộc tính dựa vào độ lợi thông tin Entropy. " * 6
        store = {
            "1": FakeDoc("Quá ngắn", {"doc_id": "elc3008_ch02", "page_start": 1, "page_end": 1}),
            "2": FakeDoc(long_peas, {"doc_id": "elc3008_ch02", "page_start": 5, "page_end": 5}),
            "3": FakeDoc(long_search, {"doc_id": "elc3008_ch03", "page_start": 8, "page_end": 9}),
            "4": FakeDoc(long_other, {"doc_id": "l07_cay_quyet_dinh", "page_start": 4, "page_end": 4}),
        }
        picked = sample_quiz_chunks(
            store,
            category="Bài giảng ELC3008",
            topic="PEAS Sensors",
            max_chunks=2,
            rng=random.Random(42),
        )
        self.assertEqual(len(picked), 2)
        self.assertEqual(picked[0]["doc_id"], "elc3008_ch02")
        self.assertEqual(picked[0]["trang"], "5")
        self.assertTrue(picked[0]["url"].startswith("https://drive.google.com/file/d/"))
        self.assertNotIn("<!-- page:", picked[0]["excerpt"])
        ctx = format_quiz_context(picked)
        self.assertIn("[Nguồn 1 | Slide ELC3008 — Chương 2 | Trang 5]", ctx)

    def test_parse_quiz_response_strips_think_and_fences_and_binds_sources(self):
        sampled = [
            {
                "source_idx": 1,
                "doc_id": "elc3008_ch02",
                "doc_title": "Slide ELC3008 — Chương 2",
                "trang": "5",
                "url": "https://drive.google.com/file/d/1LrKzoc3u8i8e9Q7UEv7gNomM29QlyFVL/view",
            }
        ]
        raw = """<think>suy nghĩ nội bộ</think>
```json
[
  {
    "question": "Chữ P trong PEAS viết tắt của từ nào?",
    "options": {
      "A": "A. Performance measure",
      "B": "B) Policy",
      "C": "C. Planning",
      "D": "D: Prediction"
    },
    "answer": "a.",
    "explanation": "P là Performance measure.",
    "source_idx": 99
  }
]
```"""
        qs = parse_quiz_response(raw, sampled, expected_count=5)
        self.assertEqual(len(qs), 1)
        self.assertEqual(qs[0]["answer"], "A")
        self.assertEqual(qs[0]["options"]["A"], "Performance measure")
        self.assertEqual(qs[0]["doc_title"], "Slide ELC3008 — Chương 2")
        self.assertEqual(qs[0]["trang"], "5")
        self.assertTrue(qs[0]["url"].startswith("https://"))

        with self.assertRaises(QuizParseError):
            parse_quiz_response("not json", sampled)

    def test_parse_quiz_response_recovers_latex_trailing_brackets_lists_and_truncated_arrays(self):
        sampled = [
            {
                "source_idx": 1,
                "doc_id": "elc3008_ch03",
                "doc_title": "Slide ELC3008 — Chương 3",
                "trang": "10",
                "url": "https://drive.google.com/file/d/1vbnTrAOYfWiCTwGdLBv6eJnSW1mNWAu9/view",
            }
        ]
        messy_raw = r"""<think>đang suy nghĩ chưa đóng thẻ
[
  {
    "question": "Độ phức tạp của thuật toán $\alpha$-$\beta$ với nhánh \frac{b}{2}?",
    "options": ["A. $O(b^d)$", "B. O(b^{d/2})", "C. \\alpha + \\beta", "D. O(1)"],
    "correct_answer": "B",
    "explanation": "Cắt tỉa \alpha-\beta giảm số nhánh [Nguồn 1].",
    "source_idx": 1,
  },
  {
    "question": "Hàng đợi ưu tiên dùng trong thuật toán nào?",
    "choices": {"A": "DFS", "B": "UCS", "C": "Hill", "D": "Random"},
    "answer": 1,
    "explanation": "UCS mở rộng nút theo chi phí g(n).",
  },
  {
    "question": "Câu thứ 3 bị cắt giữa chừng chưa kịp đóng JSON"""
        qs = parse_quiz_response(messy_raw, sampled, expected_count=5)
        self.assertEqual(len(qs), 2)
        self.assertEqual(qs[0]["answer"], "B")
        self.assertEqual(qs[0]["options"]["A"], "$O(b^d)$")
        self.assertEqual(qs[1]["answer"], "B")
        self.assertEqual(qs[1]["options"]["B"], "UCS")

    def test_grade_quiz_computes_score_and_percentage(self):
        qs = [
            {"question": "Q1", "options": {"A": "1", "B": "2", "C": "3", "D": "4"}, "answer": "A"},
            {"question": "Q2", "options": {"A": "1", "B": "2", "C": "3", "D": "4"}, "answer": "C"},
        ]
        res = grade_quiz(qs, {0: "A", 1: "B"})
        self.assertEqual(res["score"], 1)
        self.assertEqual(res["total"], 2)
        self.assertEqual(res["pct"], 50.0)
        self.assertTrue(res["results"][0]["is_correct"])
        self.assertFalse(res["results"][1]["is_correct"])

    def test_invoke_text_rotates_on_rate_limit_and_reports_fallback(self):
        attempts = []

        class Chain:
            def __init__(self, model, key):
                self.model = model
                self.key = key

            def invoke(self, inputs):
                attempts.append((self.model, self.key))
                if self.model == "m1":
                    raise RateLimitError("429")
                return "[]"

        reports = []
        out = invoke_text(
            lambda m, k: Chain(m, k),
            [("m1", "k1"), ("m2", "k1")],
            {"context": "c"},
            report=lambda m, k, fb: reports.append((m, k, fb)),
        )
        self.assertEqual(out, "[]")
        self.assertEqual(attempts, [("m1", "k1"), ("m2", "k1")])
        self.assertEqual(reports, [("m2", "k1", True)])

    def test_build_followup_questions_returns_three_unique_suggestions(self):
        q1 = "Mô hình PEAS gồm những thành phần nào?"
        hist = [q1, "Áp dụng mô hình PEAS cho một bài toán kinh doanh hoặc xe tự hành?"]
        fups = build_followup_questions(
            q1,
            sources=[{"doc_id": "elc3008_ch02"}, {"doc_id": "cs188_ch01_search"}],
            history_questions=hist,
            answer="PEAS gồm Performance, Environment, Actuators, Sensors.",
        )
        self.assertEqual(len(fups), 3)
        self.assertEqual(len({f.casefold() for f in fups}), 3)
        for existing in hist:
            self.assertNotIn(existing, fups)


if __name__ == "__main__":
    unittest.main()
