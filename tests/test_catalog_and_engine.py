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
