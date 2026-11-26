import unittest
from ui.theme import (
    PALETTE,
    _global_css,
    clean_snippet,
    latex_to_plain_text,
    normalize_latex_for_streamlit,
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

    def test_latex_to_plain_text_converts_delimited_and_bare_formulas(self):
        self.assertEqual(
            latex_to_plain_text(r"Hàm sigmoid: $\sigma(x) = \frac{1}{1 + e^{-x}}$."),
            "Hàm sigmoid: σ(x) = 1 / (1 + e^(-x)).",
        )
        self.assertEqual(
            latex_to_plain_text(r"\[P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}\]"),
            "P(A|B) = (P(B|A) · P(A)) / P(B)",
        )
        self.assertEqual(
            latex_to_plain_text(
                r"$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)]$$"
            ),
            "min_G max_D V(D, G) = E_(x ~ p_data(x))[log D(x)]",
        )
        self.assertEqual(
            latex_to_plain_text(
                r"Công thức: \(\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\)"
            ),
            "Công thức: softmax(QK^T / √(d_k))V",
        )
        self.assertEqual(
            latex_to_plain_text(
                r"Cập nhật: \theta_{t+1} = \theta_t - \alpha \cdot \nabla J(\theta_t)"
            ),
            "Cập nhật: θ_(t+1) = θ_t - α · ∇ J(θ_t)",
        )

    def test_latex_to_plain_text_preserves_currency_and_code_blocks(self):
        currency = "Giá từ $5 đến $10 hoặc $5-$10 cho mỗi lượt."
        self.assertEqual(latex_to_plain_text(currency), currency)

        code_sample = (
            "Dùng `my_var_1` và:\n"
            "```python\n"
            "price = '$5-$10'\n"
            "expr = r'\\frac{1}{2}'\n"
            "```\n"
            "Và công thức $\\hat{y} = \\vec{w} \\cdot x$."
        )
        out = latex_to_plain_text(code_sample)
        self.assertIn("`my_var_1`", out)
        self.assertIn("price = '$5-$10'", out)
        self.assertIn("expr = r'\\frac{1}{2}'", out)
        self.assertIn("hat(y) = vec(w) · x", out)

    def test_clean_snippet_converts_latex_and_keeps_html_escaping(self):
        raw = r"<!-- page: 12 --> Độ phức tạp $O(b^d)$ với $x \leq y$ và <tag>"
        cleaned = clean_snippet(raw, max_len=120)
        self.assertNotIn("<!-- page:", cleaned)
        self.assertNotIn("$", cleaned)
        self.assertIn("O(b^d)", cleaned)
        self.assertIn("x ≤ y", cleaned)
        self.assertIn("&lt;tag&gt;", cleaned)

    def test_normalize_latex_for_streamlit_converts_delimiters_and_preserves_code(self):
        src = (
            "Công thức:\n"
            r"\[ \pi_\theta(s, a) = \frac{\exp(\beta \hat{Q}_\theta(s, a))}{\sum_{a'} \exp(\beta \hat{Q}_\theta(s, a'))} \]"
            "\nVà biểu thức inline \\( f(n) = g(n) + h(n) \\), $ \\frac{1}{2} $, $$E = mc^2$$.\n"
            "```latex\n"
            r"\alpha + \beta"
            "\n```\n"
            "Giá từ $5 đến $10, giữ code `\\(x\\)` và:\n"
            "```python\n"
            "x = r'\\[a + b\\]'\n"
            "```"
        )
        out = normalize_latex_for_streamlit(src)
        self.assertIn(
            "\n\n$$\n\\pi_\\theta(s, a) = \\frac{\\exp(\\beta \\hat{Q}_\\theta(s, a))}{\\sum_{a'} \\exp(\\beta \\hat{Q}_\\theta(s, a'))}\n$$\n\n",
            out,
        )
        self.assertIn("$f(n) = g(n) + h(n)$", out)
        self.assertIn(r"$\frac{1}{2}$", out)
        self.assertIn("$$\nE = mc^2\n$$", out)
        self.assertIn("$$\n\\alpha + \\beta\n$$", out)
        self.assertIn("Giá từ $5 đến $10", out)
        self.assertIn("`\\(x\\)`", out)
        self.assertIn("x = r'\\[a + b\\]'", out)

    def test_global_css_isolates_nested_citation_details_and_spaces_telemetry(self):
        css = _global_css()
        self.assertIn("[data-testid=\"stExpander\"] > details {", css)
        self.assertNotIn("[data-testid=\"stExpander\"] details {", css)
        self.assertIn(".rag-cite-details", css)
        self.assertIn("border:none !important", css)
        self.assertIn("margin:10px 0 14px;", css)
        self.assertIn('[data-testid="stChatMessage"] [data-testid="stExpander"]', css)
        self.assertIn(".katex-display", css)

    def test_render_citation_box_rejects_unsafe_urls(self):
        out = render_citation_box_html([
            {"doc_id": "a", "doc_title": "Doc JS", "url": "javascript:alert(1)", "snippet": "s"},
            {"doc_id": "b", "doc_title": "Doc Data", "url": "data:text/html,<script>", "snippet": "s"},
        ])
        self.assertNotIn("<a ", out)
        self.assertNotIn("javascript:", out)
        self.assertNotIn("data:text/html", out)
        self.assertIn("Doc JS", out)

    def test_retrieval_panel_distinguishes_local_provider_and_rerank_fallback(self):
        local_out = render_retrieval_panel_html({
            "retrieve_ms": 20.0,
            "rerank_ms": 150.0,
            "rerank_enabled": True,
            "rerank_provider": "local",
            "total_candidates": 1,
            "selected": [{"doc_id": "a", "doc_title": "Doc A", "trang": "1",
                          "initial_rank": 1, "final_rank": 1, "rank_delta": 0, "rerank_score": 0.8}],
            "candidates": [],
        })
        self.assertIn("Local Cross-Encoder", local_out)

        fb_out = render_retrieval_panel_html({
            "retrieve_ms": 20.0,
            "rerank_ms": 40.0,
            "rerank_enabled": False,
            "rerank_fallback": True,
            "rerank_provider": "cohere",
            "total_candidates": 1,
            "selected": [{"doc_id": "a", "doc_title": "Doc A", "trang": "1",
                          "initial_rank": 1, "final_rank": 1, "rank_delta": 0, "rerank_score": None}],
            "candidates": [],
        })
        self.assertIn("LỖI", fb_out)
        self.assertIn("Rerank lỗi — giữ thứ tự Stage-1", fb_out)

    def test_turn_telemetry_shows_warning_chip_on_rerank_fallback(self):
        out = render_turn_telemetry_html({
            "latency_s": 1.5,
            "retrieve_ms": 30.0,
            "rerank_ms": 25.0,
            "rerank_enabled": False,
            "rerank_fallback": True,
            "selected_count": 5,
            "total_candidates": 20,
            "model": "qwen/qwen3.8-27b",
        })
        self.assertIn("Rerank lỗi · Stage-1", out)
        self.assertIn("rag-tel-warn", out)

    def test_light_palette_meets_wcag_aa_contrast(self):
        def rel_lum(hex_c):
            h = hex_c.lstrip("#")
            rgb = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
            lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
            return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

        def contrast(c1, c2):
            l1, l2 = rel_lum(c1), rel_lum(c2)
            return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

        pairs = [
            (PALETTE["muted"], PALETTE["sidebar"]),
            (PALETTE["muted"], PALETTE["card"]),
            (PALETTE["warning"], "#FFFBEB"),
            (PALETTE["success"], "#ECFDF5"),
            (PALETTE["info"], "#F0FDFA"),
            (PALETTE["info"], "#F0F9FF"),
            (PALETTE["danger"], "#FEF2F2"),
            (PALETTE["link"], PALETTE["active_pill"]),
            ("#FFFFFF", PALETTE["primary"]),
        ]
        for fg, bg in pairs:
            self.assertGreaterEqual(contrast(fg, bg), 4.5, f"Low contrast for {fg} on {bg}")

    def test_global_css_covers_form_submit_markdown_tables_and_visible_delete_targets(self):
        css = _global_css()
        self.assertIn("stBaseButton-primaryFormSubmit", css)
        self.assertIn('[data-testid="stForm"]', css)
        self.assertIn("@media (max-width: 1120px)", css)
        self.assertIn("text-wrap:balance", css)
        self.assertIn('[data-testid="stChatMessage"] table', css)
        self.assertIn("opacity:0.78", css)
        self.assertNotIn("opacity:0.45", css)

    def test_typography_loads_vietnamese_fonts_without_breaking_icons(self):
        from ui.theme import _FONT_LINKS_HTML

        self.assertIn("Plus+Jakarta+Sans", _FONT_LINKS_HTML)
        self.assertIn("Be+Vietnam+Pro", _FONT_LINKS_HTML)
        self.assertIn("JetBrains+Mono", _FONT_LINKS_HTML)
        self.assertIn('<link rel="stylesheet"', _FONT_LINKS_HTML)

        css = _global_css()
        self.assertIn("--font-display", css)
        self.assertIn("font-family:var(--font-sans) !important", css)
        self.assertNotIn("* { font-family", css)
        self.assertIn('[data-testid="stChatMessage"] { animation:rag-rise', css)
        self.assertIn("background:#FFFFFF; border:1px solid var(--border); border-radius:16px;", css)

    def test_inject_global_css_separates_links_and_keeps_style_block_contiguous(self):
        import sys
        import types
        from unittest.mock import patch

        from ui.theme import inject_global_css

        css = _global_css()
        self.assertNotIn("\n\n", css)

        calls = []
        fake_st = types.SimpleNamespace(
            markdown=lambda body, unsafe_allow_html=False: calls.append((body, unsafe_allow_html))
        )
        with patch.dict(sys.modules, {"streamlit": fake_st}):
            inject_global_css()

        self.assertEqual(len(calls), 2)
        links_body, links_unsafe = calls[0]
        style_body, style_unsafe = calls[1]
        self.assertTrue(links_unsafe)
        self.assertTrue(style_unsafe)
        self.assertIn("<link", links_body)
        self.assertNotIn("<style>", links_body)
        self.assertTrue(style_body.startswith("<style>"))
        self.assertTrue(style_body.endswith("</style>"))
        self.assertNotIn("\n\n", style_body)

    def test_render_source_pills_handles_links_overflow_and_unsafe_urls(self):
        from ui.theme import render_source_pills_html

        self.assertEqual(render_source_pills_html([]), "")
        sources = [
            {
                "doc_id": "d1",
                "doc_title": "Giáo trình AI <Part 1>",
                "trang": "14",
                "final_rank": 1,
                "url": "https://drive.google.com/file/d/abc/view",
            },
            {
                "doc_id": "d2",
                "doc_title": "Slide tìm kiếm",
                "trang": "6",
                "final_rank": 2,
                "url": "javascript:alert(1)",
            },
            {"doc_id": "d3", "doc_title": "Doc 3", "trang": "9", "final_rank": 3},
            {
                "doc_id": "d4",
                "doc_title": "Tài liệu rất dài vượt quá ba mươi bốn ký tự",
                "trang": "12",
                "final_rank": 4,
            },
            {"doc_id": "d5", "doc_title": "Doc 5", "trang": "20", "final_rank": 5},
        ]
        out = render_source_pills_html(sources)
        self.assertIn("https://drive.google.com/file/d/abc/view", out)
        self.assertIn("Giáo trình AI &lt;Part 1&gt;", out)
        self.assertIn("tr.14", out)
        self.assertNotIn("javascript:alert", out)
        self.assertIn("…", out)
        self.assertNotIn("Doc 5", out)
        self.assertIn("+1 nguồn", out)

    def test_global_css_styles_custom_avatars_followups_quiz_and_distinct_sidebar(self):
        css = _global_css()
        self.assertIn('[data-testid^="stChatMessageAvatar"]', css)
        self.assertIn(".rag-followup-label", css)
        self.assertIn('[class*="st-key-fup_"]', css)
        self.assertIn(".rag-quiz-score", css)
        self.assertIn('[class*="st-key-conv_"] [data-testid="stBaseButton-primary"]', css)
        self.assertNotIn("\n\n", css)


if __name__ == "__main__":
    unittest.main()