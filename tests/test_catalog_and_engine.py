import types
import unittest
from ui.catalog import (
    CATALOG,
    CATEGORY_ORDER,
    DRIVE_LINKS,
    serialize_sources,
    should_show_sources,
    summarize_docstore,
)
from rag_engine_web import inspect_retrieval


class FakeDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = dict(metadata)


class TestCatalogAndEngine(unittest.TestCase):
    def test_bm25_matches_pdf_acronyms_and_query_punctuation(self):
        from rag_engine_web import tokenize_bm25
        query = tokenize_bm25("Mô hình PEAS là gì?")
        chunk = tokenize_bm25("thePEAS (Performance Measure,Environment,Actuators, Sensors)")
        self.assertIn("peas", query)
        self.assertIn("peas", chunk)
        self.assertIn("environment", chunk)
        self.assertNotIn("gì", query)
        self.assertNotIn("là", query)
        self.assertIn("hình", query)
        self.assertIn("dfs", tokenize_bm25("DFS/BFS"))

    def test_build_doc_freq_counts_documents_not_repetitions(self):
        from rag_engine_web import build_doc_freq
        df = build_doc_freq([["peas", "peas", "hình"], ["hình", "mô"]])
        self.assertEqual(df, {"peas": 1, "hình": 2, "mô": 1})

    def test_select_query_terms_drops_words_that_dilute_the_decisive_keyword(self):
        from rag_engine_web import build_doc_freq, select_query_terms
        # 1 chunk chứa "peas"; 9 chunk khác đều nói về "mô hình/thành phần/ví dụ".
        noise = ["mô", "hình", "thành", "phần", "gồm", "ví", "dụ"]
        corpus = [noise + ["peas"]] + [list(noise) for _ in range(9)]
        df = build_doc_freq(corpus)
        kept = select_query_terms("Mô hình PEAS gồm những thành phần nào? Cho ví dụ.", df, len(corpus))
        self.assertEqual(kept, ["peas"])

    def test_select_query_terms_falls_back_when_no_term_is_specific(self):
        from rag_engine_web import build_doc_freq, select_query_terms
        corpus = [["mô", "hình"] for _ in range(10)]
        df = build_doc_freq(corpus)
        self.assertEqual(select_query_terms("mô hình", df, len(corpus)), ["mô", "hình"])

    def test_select_query_terms_keeps_query_order_and_handles_empty_input(self):
        from rag_engine_web import build_doc_freq, select_query_terms
        df = build_doc_freq([["dfs"], ["bfs"], ["dfs", "bfs"]])
        self.assertEqual(select_query_terms("BFS và DFS", df, 100), ["bfs", "dfs"])
        self.assertEqual(select_query_terms("", df, 3), [])
        self.assertEqual(select_query_terms("peas", {}, 0), ["peas"])

    def test_catalog_covers_all_18_indexed_docs_and_13_links(self):
        self.assertEqual(len(CATALOG), 18)
        self.assertEqual(len(DRIVE_LINKS), 13)
        for meta in CATALOG.values():
            self.assertIn(meta["category"], CATEGORY_ORDER)
            self.assertTrue(meta["sample_question"])

    def test_summarize_docstore_handles_known_unknown_and_missing_pages(self):
        store = {
            "1": FakeDoc("a", {"doc_id": "elc3008_ch01", "doc_title": "Ch1", "page_start": 2, "page_end": 5}),
            "2": FakeDoc("b", {"doc_id": "elc3008_ch01", "doc_title": "Ch1", "page_start": 6, "page_end": 9}),
            "3": FakeDoc("c", {"doc_id": "mystery_doc", "page_start": None, "page_end": None}),
        }
        rows = summarize_docstore(store)
        by_id = {r["doc_id"]: r for r in rows}
        self.assertEqual(by_id["elc3008_ch01"]["chunk_count"], 2)
        self.assertEqual((by_id["elc3008_ch01"]["page_min"], by_id["elc3008_ch01"]["page_max"]), (2, 9))
        self.assertEqual(by_id["mystery_doc"]["category"], "Tài liệu khác")
        self.assertIsNone(by_id["mystery_doc"]["page_min"])

    def test_should_show_sources_is_case_insensitive_on_refusal(self):
        self.assertFalse(should_show_sources("Tài liệu môn học KHÔNG ĐỀ CẬP nội dung này."))
        self.assertFalse(should_show_sources(""))
        self.assertTrue(should_show_sources("DFS dùng ngăn xếp LIFO (CS188, trang 9)."))
        docs = [FakeDoc("<!-- page: 9 --> Nội dung DFS", {"doc_id": "cs188_ch01_search", "page_start": 9, "page_end": 9, "rerank_score": 0.88})]
        sources = serialize_sources(docs)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["trang"], "9")
        self.assertEqual(sources[0]["url"], DRIVE_LINKS["cs188_ch01_search"])

    def test_serialize_sources_merges_same_doc_and_page_keeping_best_chunk(self):
        docs = [
            FakeDoc("chunk yếu", {"doc_id": "cs188_ch01_search", "doc_title": "CS188",
                                  "page_start": 3, "page_end": 3, "rerank_score": 0.12}),
            FakeDoc("chunk mạnh", {"doc_id": "cs188_ch01_search", "doc_title": "CS188",
                                   "page_start": 3, "page_end": 3, "rerank_score": 0.89}),
            FakeDoc("trang khác", {"doc_id": "cs188_ch01_search", "doc_title": "CS188",
                                   "page_start": 23, "page_end": 23, "rerank_score": 0.5}),
        ]
        out = serialize_sources(docs)
        self.assertEqual([s["trang"] for s in out], ["3", "23"])
        self.assertEqual(out[0]["snippet"], "chunk mạnh")
        self.assertEqual(out[0]["rerank_score"], 0.89)

    def test_serialize_sources_keeps_first_chunk_when_scores_are_missing(self):
        docs = [
            FakeDoc("một", {"doc_id": "elc3008_ch01", "page_start": 2, "page_end": 2}),
            FakeDoc("hai", {"doc_id": "elc3008_ch01", "page_start": 2, "page_end": 2}),
        ]
        out = serialize_sources(docs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["snippet"], "một")
        self.assertIsNone(out[0]["rerank_score"])

    def test_describe_llm_error_turns_rate_limit_into_action(self):
        from rag_engine_web import describe_llm_error

        class RateLimitError(Exception):
            pass

        msg = describe_llm_error(RateLimitError("upstream body with token abc123"))
        self.assertIn("hạn mức", msg.lower())
        self.assertIn("Tạo lại", msg)
        self.assertIn("ngày", msg)
        self.assertIn("LLM_MODEL", msg)
        self.assertNotIn("abc123", msg)

    def test_describe_llm_error_names_other_failures_without_leaking_body(self):
        from rag_engine_web import describe_llm_error
        msg = describe_llm_error(ValueError("upstream body with token abc123"))
        self.assertIn("ValueError", msg)
        self.assertNotIn("abc123", msg)

    def test_describe_llm_error_turns_model_not_found_into_model_switch(self):
        # Groq trả 404 khi model bị kéo khỏi free tier (llama-3.3-70b-versatile),
        # bấm lại không bao giờ khỏi — phải đổi model.
        from rag_engine_web import describe_llm_error

        class NotFoundError(Exception):
            pass

        msg = describe_llm_error(NotFoundError("upstream body with token abc123"))
        self.assertIn("LLM_MODEL", msg)
        self.assertIn("openai/gpt-oss-20b", msg)
        self.assertNotIn("abc123", msg)

    def test_default_llm_model_is_still_on_groq_free_tier(self):
        import rag_engine_web
        free_models = {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"}
        self.assertIn(rag_engine_web.LLM_MODEL, free_models)

    def test_default_llm_model_is_the_one_the_deployed_key_has_room_for(self):
        # Bản deploy không còn dòng LLM_MODEL trong Secrets -> default trong code là
        # cấu hình thật. Key Groq đang dùng đã đốt trần ngày (TPD trôi 24h) trên
        # gpt-oss-120b và gpt-oss-20b, chỉ qwen còn trống.
        import rag_engine_web
        self.assertEqual(rag_engine_web.LLM_MODEL, "qwen/qwen3.8-27b")

    def test_inspect_retrieval_computes_rank_deltas_without_mutating_originals(self):
        d1 = FakeDoc("chunk one", {"doc_id": "elc3008_ch01", "start_index": 0, "page_start": 3, "page_end": 3})
        d2 = FakeDoc("chunk two", {"doc_id": "ttnt_tu_minh_phuong", "start_index": 10, "page_start": 38, "page_end": 39})
        ensemble = types.SimpleNamespace(invoke=lambda q: [d1, d2])
        reranker = types.SimpleNamespace(predict=lambda pairs: [0.15, 0.92])
        bundle = {
            "ensemble": ensemble,
            "reranker": reranker,
            "config": {"top_k": 1, "retrieve_n": 20, "rerank_provider": "cohere"},
        }
        res = inspect_retrieval(bundle, "So sánh DFS và BFS")
        self.assertTrue(res["rerank_enabled"])
        self.assertEqual(res["total_candidates"], 2)
        self.assertEqual(len(res["selected"]), 1)
        top = res["selected"][0]
        self.assertEqual(top["doc_id"], "ttnt_tu_minh_phuong")
        self.assertEqual(top["initial_rank"], 2)
        self.assertEqual(top["final_rank"], 1)
        self.assertEqual(top["rank_delta"], 1)
        self.assertNotIn("rerank_score", d2.metadata)


if __name__ == "__main__":
    unittest.main()
