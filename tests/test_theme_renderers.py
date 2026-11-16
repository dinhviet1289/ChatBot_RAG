import unittest
from ui.theme import (
    PALETTE,
    _global_css,
    clean_snippet,
    render_citation_box_html,
)


class TestThemeRenderers(unittest.TestCase):
    def test_clean_snippet_strips_page_comments_and_escapes_html(self):
        raw = "<!-- page: 38 -->\nThuật toán <DFS> & BFS dùng O(b^d) bộ nhớ <script>alert(1)</script>"
        cleaned = clean_snippet(raw, max_len=120)
        self.assertNotIn("<!-- page:", cleaned)
        self.assertNotIn("<script>", cleaned)
        self.assertIn("&lt;DFS&gt;", cleaned)
        self.assertIn("&amp;", cleaned)

    def test_render_citation_box_handles_linked_and_unlinked_sources_safely(self):
        sources = [
            {
                "doc_id": "cs188_ch01_search",
                "doc_title": "CS188 <Search>",
                "trang": "9",
                "url": "https://drive.google.com/file/d/abc/view",
                "snippet": "<!-- page: 9 --> Fringe FIFO",
                "rerank_score": 0.9123,
            },
            {
                "doc_id": "ttnt_tu_minh_phuong",
                "doc_title": "Trí tuệ nhân tạo — Từ Minh Phương",
                "trang": "38-39",
                "url": None,
                "snippet": "Tìm kiếm theo chiều sâu",
                "rerank_score": None,
            },
        ]
        html = render_citation_box_html(sources)
        self.assertIn(PALETTE["callout"], html)
        self.assertIn("https://drive.google.com/file/d/abc/view", html)
        self.assertIn("CS188 &lt;Search&gt;", html)
        self.assertIn("0.912", html)
        self.assertIn("Từ Minh Phương", html)

    def test_global_css_keeps_sidebar_expand_control_reachable(self):
        # stExpandSidebarButton lives inside stToolbar; hiding the whole toolbar
        # strands a collapsed sidebar with no way to reopen it.
        css = _global_css()
        self.assertNotIn('[data-testid="stToolbar"] { display:none; }', css)
        self.assertIn('[data-testid="stToolbarActions"] { display:none; }', css)


if __name__ == "__main__":
    unittest.main()