import unittest
from ui.theme import render_citation_box_html


class TestChatCitation(unittest.TestCase):
    def test_citation_box_contains_verbatim_snippets_and_no_duplicate_expander(self):
        src = {
            "doc_id": "cs188_ch01_search",
            "doc_title": "CS188 Berkeley",
            "trang": "9",
            "url": None,
            "snippet": "To define the task environment, we use the PEAS description.",
            "rerank_score": 0.9,
        }
        html = render_citation_box_html([src])
        self.assertIn(src["snippet"], html)
        detail_snippets = html.count('class="rag-cite-snippet"')
        self.assertEqual(detail_snippets, 1)

    def test_citation_box_collapses_snippet_under_source(self):
        src = {"doc_id": "a", "doc_title": "A", "snippet": "Đoạn trích gốc đầy đủ để đọc ngay."}
        html = render_citation_box_html([src])
        self.assertIn("<details", html)
        self.assertIn("Trích đoạn", html)


if __name__ == "__main__":
    unittest.main()
