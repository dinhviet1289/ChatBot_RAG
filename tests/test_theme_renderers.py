import unittest
from ui.theme import (
    PALETTE,
    _global_css,
    clean_snippet,
    render_citation_box_html,
    render_corpus_overview_html,
    render_doc_card_html,
    render_retrieval_panel_html,
    render_system_card_html,
    render_turn_telemetry_html,
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

    def test_global_css_stabilizes_streaming_and_locks_historical_messages(self):
        css = _global_css()
        self.assertNotIn("scroll-behavior:smooth", css)
        self.assertNotIn("scroll-behavior: smooth", css)
        self.assertIn('[data-testid="stChatMessage"]:not(:last-of-type) { animation:none;', css)

    def test_system_card_escapes_config_and_never_leaks_keys(self):
        cfg = {
            "rerank_provider": "cohere",
            "rerank_model": "rerank-multilingual-v3.0<x>",
            "top_k": 5,
            "retrieve_n": 20,
            "llm_model": "qwen/qwen3.8-27b<script>",
            "embed_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2<script>",
            "groq_keys": ["gsk_secret_alpha_111", "gsk_secret_beta_222"],
        }
        out = render_system_card_html(cfg)
        self.assertIn(">Groq<", out)
        self.assertNotIn("qwen/qwen3.8-27b", out)
        self.assertNotIn("gsk_secret_alpha_111", out)
        self.assertNotIn("gsk_secret_beta_222", out)
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)
        self.assertIn("&lt;x&gt;", out)

    def test_turn_telemetry_renders_latency_stages_and_fallback_badge(self):
        out = render_turn_telemetry_html({
            "latency_s": 1.84,
            "retrieve_ms": 42.3,
            "rerank_ms": 115.7,
            "rerank_enabled": True,
            "selected_count": 5,
            "total_candidates": 20,
            "model": "openai/gpt-oss-20b",
            "key_slot": "2/2",
            "fell_back": True,
            "configured_model": "qwen/qwen3.8-27b<tag>",
        })
        self.assertIn("1.84s", out)
        self.assertIn("Stage-1 42ms", out)
        self.assertIn("Rerank 116ms", out)
        self.assertIn("5/20 chunks", out)
        self.assertIn("openai/gpt-oss-20b · key 2/2", out)
        self.assertIn("Dự phòng", out)
        self.assertNotIn("<tag>", out)
        self.assertEqual(render_turn_telemetry_html(None), "")

    def test_retrieval_panel_renders_deltas_stage1_table_and_escapes_html(self):
        res = {
            "question": "DFS vs BFS",
            "retrieve_ms": 38.4,
            "rerank_ms": 124.2,
            "rerank_enabled": True,
            "total_candidates": 3,
            "selected": [
                {
                    "doc_id": "cs188_ch01_search",
                    "doc_title": "CS188 <Search>",
                    "trang": "9",
                    "url": "https://drive.google.com/file/d/abc/view",
                    "snippet": "<!-- page: 9 --> Fringe <b>LIFO</b> vs FIFO",
                    "initial_rank": 3,
                    "final_rank": 1,
                    "rank_delta": 2,
                    "rerank_score": 0.9123,
                },
                {
                    "doc_id": "elc3008_ch03",
                    "doc_title": "Slide ELC3008",
                    "trang": "12",
                    "url": None,
                    "snippet": "Tìm kiếm không có thông tin",
                    "initial_rank": 1,
                    "final_rank": 2,
                    "rank_delta": -1,
                    "rerank_score": 0.654,
                },
            ],
            "candidates": [
                {
                    "doc_id": "elc3008_ch03", "doc_title": "Slide ELC3008", "trang": "12",
                    "initial_rank": 1, "final_rank": 2, "rank_delta": -1, "rerank_score": 0.654,
                },
                {
                    "doc_id": "ttnt_tu_minh_phuong", "doc_title": "Từ Minh Phương", "trang": "38",
                    "initial_rank": 2, "final_rank": 2, "rank_delta": 0, "rerank_score": 0.41,
                },
                {
                    "doc_id": "cs188_ch01_search", "doc_title": "CS188 <Search>", "trang": "9",
                    "initial_rank": 3, "final_rank": 1, "rank_delta": 2, "rerank_score": 0.9123,
                },
            ],
        }
        out = render_retrieval_panel_html(res)
        self.assertIn("38 ms", out)
        self.assertIn("124 ms", out)
        self.assertIn("2 / 3", out)
        self.assertIn("▲ +2", out)
        self.assertIn("▼ -1", out)
        self.assertIn("● 0", out)
        self.assertIn("0.9123", out)
        self.assertIn("CS188 &lt;Search&gt;", out)
        self.assertNotIn("<b>LIFO</b>", out)
        self.assertIn('class="rag-stage1-details"', out)
        self.assertIn("TOP-K", out)

    def test_corpus_overview_and_doc_card_render_kpis_lang_and_escape(self):
        rows = [
            {
                "doc_id": "cs188_ch01_search",
                "title": "CS188 <Ch1>",
                "category": "UC Berkeley CS188",
                "lang": "EN",
                "url": "https://drive.google.com/file/d/abc/view",
                "chunk_count": 42,
                "page_min": 2,
                "page_max": 23,
            },
            {
                "doc_id": "elc3008_ch01",
                "title": "Slide ELC3008",
                "category": "Bài giảng ELC3008",
                "lang": "VI",
                "url": None,
                "chunk_count": 22,
                "page_min": 2,
                "page_max": 29,
            },
        ]
        overview = render_corpus_overview_html(rows)
        self.assertIn(">2<", overview)
        self.assertIn(">64<", overview)
        self.assertIn("1 Tiếng Việt · 1 English", overview)
        self.assertIn("384d", overview)

        card_en = render_doc_card_html(rows[0], max_chunks=42, total_chunks=64)
        self.assertIn("CS188 &lt;Ch1&gt;", card_en)
        self.assertIn(">EN<", card_en)
        self.assertIn("PDF Drive", card_en)
        self.assertIn("Trang 2-23", card_en)

        card_vi = render_doc_card_html(rows[1], max_chunks=42, total_chunks=64)
        self.assertIn(">VI<", card_vi)
        self.assertIn("Vector Index", card_vi)

    def test_render_citation_box_shows_rank_badge_and_rerank_shift(self):
        sources = [
            {
                "doc_id": "build_llm",
                "doc_title": "Build a Large Language Model (From Scratch)",
                "trang": "25",
                "url": None,
                "snippet": "LLMs utilize an...",
                "rerank_score": 0.888,
                "initial_rank": 5,
                "final_rank": 1,
                "rank_delta": 4,
            },
            {
                "doc_id": "elc3008_ch03",
                "doc_title": "Slide ELC3008",
                "trang": "12",
                "url": None,
                "snippet": "DFS vs BFS",
                "rerank_score": 0.497,
                "initial_rank": 1,
                "final_rank": 2,
                "rank_delta": -1,
            },
            {
                "doc_id": "cs188_ch01_search",
                "doc_title": "CS188 Berkeley",
                "trang": "9",
                "url": None,
                "snippet": "Fringe FIFO",
                "rerank_score": 0.428,
                "initial_rank": 3,
                "final_rank": 3,
                "rank_delta": 0,
            },
        ]
        out = render_citation_box_html(sources)
        self.assertIn('<span class="rag-rank-badge rag-mono">#1</span>', out)
        self.assertIn('#5 → #1', out)
        self.assertIn('▲ +4', out)
        self.assertIn('▼ -1', out)
        self.assertIn('● 0', out)
        self.assertIn('class="rag-cite-score-wrap"', out)
        self.assertNotIn('[1]', out)

        legacy_out = render_citation_box_html([{
            "doc_id": "elc3008_ch01",
            "doc_title": "Slide ELC3008",
            "trang": "3",
            "url": None,
            "snippet": "Legacy snippet",
            "rerank_score": None,
        }])
        self.assertIn('<span class="rag-rank-badge rag-mono">#1</span>', legacy_out)
        self.assertNotIn("rag-rank-flow", legacy_out)
        self.assertNotIn("rag-delta-", legacy_out)


if __name__ == "__main__":
    unittest.main()