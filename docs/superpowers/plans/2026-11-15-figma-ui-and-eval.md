# ELC3008 Figma Indigo UI & RAG Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the ELC3008 Streamlit RAG Chatbot into a multi-view Figma Indigo interface (`Trợ lý Chat`, `Kho tài liệu`, `Đánh giá RAG`) with persistent citations, multi-session chat history, real-time streaming, and a Colab-compatible Quality Gate / Live Retrieval Inspector for thesis defense.

**Architecture:** Keep `requirements.txt` and `@st.cache_resource` model loading untouched (~550MB RAM ceiling on Streamlit Cloud Free), extend `rag_engine_web.py` with `get_rag_bundle()` and `inspect_retrieval()`, and separate pure testable logic (`ui/catalog.py`, `ui/eval_schema.py`, `ui/theme.py`, `ui/state.py`) from thin Streamlit view renderers (`views/chat.py`, `views/documents.py`, `views/evaluation.py`, wired in `app.py`).

**Tech Stack:** Python 3.12 (`C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe`), Streamlit >= 1.50, LangChain (`FAISS`, `BM25Retriever`, `EnsembleRetriever`, `ChatGroq`), Cohere Rerank v3.0, pure HTML/CSS components, `unittest`.

**Spec:** `D:\Workspace\Qocode\AI\chatbox-rag\docs\superpowers\specs\2026-11-15-ui-eval-design.md`

## Global Constraints

- Do NOT modify `requirements.txt` or add third-party packages (`plotly`, `ragas`, `matplotlib`, etc. are forbidden in the web repo to stay within ~1GB RAM on Streamlit Community Cloud Free).
- Keep embedding model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and `data/faiss_index_huggingface/` (`index.faiss` + `index.pkl`, 18 documents, 3,760 chunks) unchanged.
- Preserve `get_rag()` in `rag_engine_web.py` returning `(retriever, answer_chain)` for backward compatibility while adding `get_rag_bundle()` and `inspect_retrieval()`.
- Load heavy RAG resources (`embeddings`, `db`, `ensemble`, `reranker`, `llm`, `answer_chain`) exactly once via `@st.cache_resource` in `app.py` and pass `bundle` to all 3 views.
- Seed `data/eval_results.json` with `"is_sample": true` and render a visible demo notice until replaced by real Colab output; never present placeholder numbers as actual experimental results.
- Never write real API keys to any tracked file.
- Local test runner command (working directory `D:\Workspace\Qocode\AI\chatbox-rag`):
  `& "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m unittest`

## Review Focus

1. **Raw HTML comments (`<!-- page: N -->`) or `<`/`>` characters in indexed chunks breaking custom HTML cards:** `clean_snippet()` and `render_citation_box_html()` must strip `<!-- page: \d+ -->` markers and HTML-escape all text fields before interpolation (pinned in Task 3).
2. **`NaN` / `None` metric values from timed-out Colab RAGAS jobs crashing gate checks or bar charts:** `evaluate_gate()`, `classify_sample_passed()`, `parse_uploaded_eval()`, and `render_pastel_bar_chart_html()` must coerce `NaN`/non-numeric values to `None` and render `"N/A"` with `0%` bar height without raising (pinned in Tasks 2 & 3).
3. **Case-insensitive refusal responses (`"Không đề cập"`, `"tài liệu môn học không đề cập"`) displaying misleading citations:** `should_show_sources(answer)` must match case-insensitively and store `sources=[]` on refusal turns so historic turns stay clean (pinned in Task 1).
4. **Unlisted `doc_id` or missing `page_start`/`page_end` in `db.docstore._dict`:** `summarize_docstore()` must still count every chunk under fallback group `"Tài liệu khác"` and handle `None` page numbers without `KeyError` or `TypeError` (pinned in Task 1).
5. **Clicking `↻ Tạo lại câu trả lời` repeatedly or across multiple conversations:** `prepare_regenerate()` must pop only the trailing assistant turn of the active conversation and return its preceding user prompt without duplicating the user message or mutating other conversations (pinned in Task 4).

---

### Task 1: Corpus Catalog (`ui/catalog.py`) & Engine Retrieval Inspector (`rag_engine_web.py`)

**Files:**
- Create: `D:\Workspace\Qocode\AI\chatbox-rag\ui\__init__.py`
- Create: `D:\Workspace\Qocode\AI\chatbox-rag\ui\catalog.py`
- Modify: `D:\Workspace\Qocode\AI\chatbox-rag\rag_engine_web.py`
- Test: `D:\Workspace\Qocode\AI\chatbox-rag\tests\test_catalog_and_engine.py`

**Interfaces:**
- Consumes: `db.docstore._dict` (`dict[str, Document]`), `ensemble.invoke(question)`, `reranker.predict(pairs)` from `rag_engine_web.py`.
- Produces:
  - `ui.catalog.DRIVE_LINKS: dict[str, str]` (the 13 existing Google Drive URLs from `app.py`)
  - `ui.catalog.CATEGORY_ORDER: list[str]` (`["Bài giảng ELC3008", "UC Berkeley CS188", "Deep Learning (UDL)", "Large Language Models", "Giáo trình Tiếng Việt & Cơ sở ML"]`)
  - `ui.catalog.CATALOG: dict[str, dict]` (all 18 indexed `doc_id`s mapped to `{title, category, url, sample_question}`)
  - `ui.catalog.summarize_docstore(docstore_dict: dict) -> list[dict]` -> sorted list of `{doc_id, title, category, url, sample_question, chunk_count, page_min, page_max}`
  - `ui.catalog.should_show_sources(answer: str) -> bool`
  - `ui.catalog.serialize_sources(docs: list) -> list[dict]` -> `[{doc_id, doc_title, trang, url, snippet, rerank_score}]`
  - `rag_engine_web.get_rag_bundle() -> dict` -> `{db, ensemble, reranker, retriever, answer_chain, config}`
  - `rag_engine_web.inspect_retrieval(bundle: dict, question: str) -> dict` -> `{question, retrieve_ms, rerank_ms, rerank_enabled, total_candidates, candidates: list[dict], selected: list[dict]}` where each item contains `{doc_id, doc_title, trang, url, snippet, initial_rank, final_rank, rank_delta, rerank_score}`

- [ ] **Step 1: Write the failing tests in `tests/test_catalog_and_engine.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run (in `D:\Workspace\Qocode\AI\chatbox-rag`):
`& "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m unittest tests/test_catalog_and_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui'`.

- [ ] **Step 3: Implement `ui/catalog.py` and extend `rag_engine_web.py`**

- In `ui/catalog.py`, define `DRIVE_LINKS` (preserving all 13 URLs from `app.py`), `CATEGORY_ORDER`, and `CATALOG` for all 18 indexed IDs (`elc3008_ch01`..`ch06`, `cs188_ch01_search`, `cs188_ch02_csps`, `cs188_ch03_games`, `udl_ch15_gans`, `udl_ch17_vaes`, `build_a_large_language_model_from_scratc`, `hands_on_large_language_models_language_`, `gt_tri_tue_nhan_tao`, `ttnt_nguyen_quang_hoan`, `ttnt_tu_minh_phuong`, `it3190_tai_lieu_doc`, `ml_co_ban`). Note: ensure `rag_engine_web.py` imports `streamlit` lazily inside `_secret` (or guards `st.secrets`) so unit tests run cleanly even without Streamlit runtime initialized.
- In `rag_engine_web.py`, avoid mutating `d.metadata` on shared docstore objects inside `RerankRetriever.invoke` (shallow-copy `Document` metadata) and implement `get_rag_bundle()` + `inspect_retrieval(bundle, question)` using `time.perf_counter()`, keep `get_rag()` returning `(bundle["retriever"], bundle["answer_chain"])`.

- [ ] **Step 4: Run test to verify it passes**

Run:
`& "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m unittest tests/test_catalog_and_engine.py -v`
Expected: PASS (4 tests OK).

---

### Task 2: Evaluation Schema, Parser & Colab Exporter (`ui/eval_schema.py`, `scripts/export_eval_to_web.py`, `data/eval_results.json`)

**Files:**
- Create: `D:\Workspace\Qocode\AI\chatbox-rag\ui\eval_schema.py`
- Create: `D:\Workspace\Qocode\AI\chatbox-rag\scripts\export_eval_to_web.py`
- Create: `D:\Workspace\Qocode\AI\chatbox-rag\data\eval_results.json`
- Test: `D:\Workspace\Qocode\AI\chatbox-rag\tests\test_eval_schema.py`

**Interfaces:**
- Consumes: Colab outputs matching `eval_rag.py` (`eval_report.json`, `eval_runs.jsonl`, CSV, or unified `eval_results.json`).
- Produces:
  - `ui.eval_schema.DEFAULT_THRESHOLDS: dict[str, float]` (`{"recall_at_k": 0.85, "mrr": 0.70, "faithfulness": 0.85, "answer_relevancy": 0.80, "hallucination_rate": 0.05}`)
  - `ui.eval_schema.METRIC_ORDER: list[str]` (`["recall_at_k", "mrr", "precision_at_k", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "hallucination_rate"]`)
  - `ui.eval_schema.evaluate_gate(metrics: dict, thresholds: dict | None = None) -> dict` -> `{"pass": bool, "checks": {key: {"value": float | None, "bound": float, "op": ">=" | "<", "pass": bool | None}}}`
  - `ui.eval_schema.classify_sample_passed(row: dict) -> tuple[bool, list[str]]` -> returns `(False, reasons)` if `recall_at_k == 0.0` (when not `None`), `faithfulness < 0.85` (when not `None`), or `answer_relevancy < 0.70` (when not `None`); otherwise `(True, [])`.
  - `ui.eval_schema.parse_uploaded_eval(raw_bytes: bytes, filename: str) -> dict` -> normalized web payload `{is_sample, evaluated_at, dataset_size, primary_run, baseline_run, runs, thresholds, samples}`
  - `ui.eval_schema.load_default_eval_data(path: Path | None = None) -> dict`
  - `scripts.export_eval_to_web.build_web_eval_payload(primary_metrics: dict, samples: list[dict], baseline_metrics: dict | None = None, primary_label: str = "Hybrid + Cohere Rerank (v3.0)", baseline_label: str = "Hybrid (BM25 + FAISS, tắt Rerank)") -> dict`

- [ ] **Step 1: Write the failing tests in `tests/test_eval_schema.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`& "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m unittest tests/test_eval_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui.eval_schema'`.

- [ ] **Step 3: Implement `ui/eval_schema.py`, `scripts/export_eval_to_web.py`, and `data/eval_results.json`**

- Implement `_clean_num(val)` in `ui/eval_schema.py` converting `None`, `""`, `"NaN"`, `float("nan")`, `float("inf")` to `None`; derive `hallucination_rate = round(1.0 - faithfulness, 4)` whenever missing and `faithfulness` is valid.
- Support 4 upload formats in `parse_uploaded_eval`: (1) full web `eval_results.json`, (2) Colab `eval_report.json` (`{"metrics": ..., "gate": ..., "n": ...}`), (3) `.jsonl` (`eval_runs.jsonl` computing page-level `recall_at_k`/`precision_at_k`/`mrr` when `gold_pages` + `retrieved_pages` exist), and (4) `.csv` (`result.to_pandas().to_csv()`).
- Seed `data/eval_results.json` with `"is_sample": true` and realistic ELC3008 sample questions (DFS/BFS, PEAS, N-Queens CSP, Minimax Alpha-Beta, GAN vs VAE, Transformer Self-Attention) including at least 1 `passed: false` example so filtering can be demonstrated before Colab re-run.

- [ ] **Step 4: Run test to verify it passes**

Run:
`& "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m unittest tests/test_eval_schema.py -v`
Expected: PASS (5 tests OK).

---

### Task 3: Figma Indigo Design System & Safe HTML Renderers (`ui/theme.py`, `.streamlit/config.toml`)

**Files:**
- Modify: `D:\Workspace\Qocode\AI\chatbox-rag\.streamlit\config.toml`
- Create: `D:\Workspace\Qocode\AI\chatbox-rag\ui\theme.py`
- Test: `D:\Workspace\Qocode\AI\chatbox-rag\tests\test_theme_renderers.py`

**Interfaces:**
- Consumes: `sources` dicts from `ui.catalog.serialize_sources`, `runs` & `thresholds` dicts from `ui.eval_schema`.
- Produces:
  - `ui.theme.PALETTE: dict[str, str]` (`canvas="#F4F6FB"`, `sidebar="#FFFFFF"`, `card="#FFFFFF"`, `primary="#5B6EF5"`, `primary_end="#6A5AE0"`, `active_pill="#EBF0FF"`, `callout="#EEF3FF"`)
  - `ui.theme.clean_snippet(text: str, max_len: int = 240) -> str`
  - `ui.theme.render_citation_box_html(sources: list[dict]) -> str`
  - `ui.theme.render_pastel_bar_chart_html(runs: dict, thresholds: dict | None = None, metric_keys: list[str] | None = None) -> str`
  - `ui.theme.inject_global_css() -> None`

- [ ] **Step 1: Write the failing tests in `tests/test_theme_renderers.py`**

```python
import unittest
from ui.theme import (
    PALETTE,
    clean_snippet,
    render_citation_box_html,
    render_pastel_bar_chart_html,
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

    def test_pastel_bar_chart_renders_single_or_dual_runs_and_nan_metrics(self):
        runs = {
            "baseline": {"label": "No Rerank", "metrics": {"recall_at_k": 0.65, "faithfulness": None}},
            "rerank": {"label": "Cohere Rerank", "metrics": {"recall_at_k": 0.88, "faithfulness": float("nan")}},
        }
        html = render_pastel_bar_chart_html(runs, metric_keys=["recall_at_k", "faithfulness"])
        self.assertIn("88.0%", html)
        self.assertIn("N/A", html)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`& "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m unittest tests/test_theme_renderers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui.theme'`.

- [ ] **Step 3: Update `.streamlit/config.toml` and implement `ui/theme.py`**

- Update `.streamlit/config.toml` theme colors to `primaryColor = "#5B6EF5"`, `backgroundColor = "#F4F6FB"`, `secondaryBackgroundColor = "#FFFFFF"`, `textColor = "#1E293B"`.
- In `ui/theme.py`, implement `clean_snippet` (regex strip `r"<!--\s*page:\s*\d+\s*-->"`, collapse whitespace, truncate, `html.escape`), `render_citation_box_html` (Figma `#EEF3FF` callout block with `→` arrows, rejecting non-`https://` URLs), `render_pastel_bar_chart_html` (pill-topped pastel columns matching the Figma chart aesthetic: mint `#4FD1C5`, amber `#F6AD55`, periwinkle `#63B3ED`, violet `#9F7AEA`, coral `#FC8181`, lime `#68D391`), and `inject_global_css()` styling sidebar pills, assistant white cards, and bottom floating `st.chat_input`.

- [ ] **Step 4: Run test to verify it passes**

Run:
`& "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m unittest tests/test_theme_renderers.py -v`
Expected: PASS (3 tests OK).

---

### Task 4: Multi-Conversation State & Chat View (`ui/state.py`, `views/__init__.py`, `views/chat.py`)

**Files:**
- Create: `D:\Workspace\Qocode\AI\chatbox-rag\ui\state.py`
- Create: `D:\Workspace\Qocode\AI\chatbox-rag\views\__init__.py`
- Create: `D:\Workspace\Qocode\AI\chatbox-rag\views\chat.py`
- Test: `D:\Workspace\Qocode\AI\chatbox-rag\tests\test_chat_state.py`

**Interfaces:**
- Consumes: `bundle` from `get_rag_bundle()`, `format_docs` from `rag_engine_web`, `serialize_sources` & `should_show_sources` from `ui.catalog`, `render_citation_box_html` & `clean_snippet` from `ui.theme`.
- Produces:
  - `ui.state.SUGGESTED_CARDS: list[dict]` (4 two-line empty-state prompt cards `{icon, title, subtitle, prompt}`)
  - `ui.state.init_state(state: MutableMapping) -> None`
  - `ui.state.get_active_conversation(state: MutableMapping) -> dict`
  - `ui.state.create_new_conversation(state: MutableMapping) -> str`
  - `ui.state.clear_all_conversations(state: MutableMapping) -> None`
  - `ui.state.delete_conversation(state: MutableMapping, conv_id: str) -> None`
  - `ui.state.append_user_turn(state: MutableMapping, prompt: str) -> None`
  - `ui.state.append_assistant_turn(state: MutableMapping, content: str, sources: list[dict], latency_s: float = 0.0) -> None`
  - `ui.state.prepare_regenerate(state: MutableMapping) -> str | None`
  - `ui.state.ask_about_document(state: MutableMapping, doc_id: str, prompt: str) -> None`
  - `views.chat.render_chat_view(bundle: dict) -> None`

- [ ] **Step 1: Write the failing tests in `tests/test_chat_state.py`**

```python
import unittest
from ui.state import (
    append_assistant_turn,
    append_user_turn,
    ask_about_document,
    clear_all_conversations,
    create_new_conversation,
    delete_conversation,
    get_active_conversation,
    init_state,
    prepare_regenerate,
)


class TestChatState(unittest.TestCase):
    def test_conversation_lifecycle_auto_title_and_persistence_of_sources(self):
        state = {}
        init_state(state)
        first_id = state["active_conv_id"]
        append_user_turn(state, "So sánh chi tiết thuật toán tìm kiếm DFS và BFS trong AI")
        conv = get_active_conversation(state)
        self.assertTrue(conv["title"].startswith("So sánh chi tiết"))
        self.assertLessEqual(len(conv["title"]), 38)

        src = [{"doc_id": "cs188_ch01_search", "doc_title": "CS188", "trang": "9", "url": None, "snippet": "abc", "rerank_score": 0.9}]
        append_assistant_turn(state, "Câu trả lời 1", src, latency_s=1.2)
        append_user_turn(state, "Câu hỏi 2")
        append_assistant_turn(state, "Câu trả lời 2", [], latency_s=0.8)
        self.assertEqual(len(conv["messages"][1]["sources"]), 1)

        second_id = create_new_conversation(state)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(len(get_active_conversation(state)["messages"]), 0)

        delete_conversation(state, second_id)
        self.assertEqual(state["active_conv_id"], first_id)

        clear_all_conversations(state)
        self.assertEqual(len(state["conversations"]), 1)
        self.assertEqual(len(get_active_conversation(state)["messages"]), 0)

    def test_prepare_regenerate_pops_only_trailing_assistant_of_active_thread(self):
        state = {}
        init_state(state)
        append_user_turn(state, "PEAS là gì?")
        append_assistant_turn(state, "Trả lời cũ", [{"doc_id": "elc3008_ch02"}])

        other_id = create_new_conversation(state)
        append_user_turn(state, "GAN là gì?")
        append_assistant_turn(state, "GAN trả lời", [])

        rerun_q = prepare_regenerate(state)
        self.assertEqual(rerun_q, "GAN là gì?")
        active_msgs = get_active_conversation(state)["messages"]
        # Trailing assistant popped AND preceding user popped so the re-queued prompt won't duplicate user bubble
        self.assertEqual(len(active_msgs), 0)
        # First conversation untouched
        first_conv = [c for c in state["conversations"] if c["id"] != other_id][0]
        self.assertEqual(len(first_conv["messages"]), 2)

    def test_ask_about_document_switches_to_chat_with_pending_prompt(self):
        state = {}
        init_state(state)
        state["active_page"] = "documents"
        ask_about_document(state, "udl_ch15_gans", "Giải thích hàm mục tiêu của mạng GAN.")
        self.assertEqual(state["active_page"], "chat")
        self.assertEqual(state["pending_prompt"], "Giải thích hàm mục tiêu của mạng GAN.")
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`& "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m unittest tests/test_chat_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui.state'`.

- [ ] **Step 3: Implement `ui/state.py` and `views/chat.py`**

- In `ui/state.py`, implement pure dictionary state helpers. In `prepare_regenerate(state)`, pop the trailing assistant message AND pop the immediately preceding user message returning its `content` (so `state["pending_prompt"] = rerun_q` feeds through the exact single chat-turn execution path without duplicating the user bubble).
- In `views/chat.py`, implement `render_chat_view(bundle)`:
  - When `len(conv["messages"]) == 0` and no `pending_prompt`: render centered hero greeting + 2x2 `SUGGESTED_CARDS` (`st.columns(2)`).
  - Render historic messages with custom Indigo/User avatars; for assistant turns with `msg["sources"]`, render `render_citation_box_html(msg["sources"])` + `st.expander("Xem đoạn trích nguyên văn")` using `clean_snippet`.
  - When active conversation ends with an assistant turn, render centered `↻ Tạo lại câu trả lời` pill button (`prepare_regenerate` + `st.rerun()`).
  - On new prompt (`st.chat_input` or `pending_prompt`), retrieve `docs = bundle["retriever"].invoke(prompt)` once, stream answer via `st.write_stream(bundle["answer_chain"].stream({"context": format_docs(docs), "question": prompt}))`, attach `serialize_sources(docs)` iff `should_show_sources(answer)`, save to state, and render citations + regenerate button cleanly.

- [ ] **Step 4: Run test to verify it passes**

Run:
`& "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m unittest tests/test_chat_state.py -v`
Expected: PASS (3 tests OK).

---

### Task 5: Documents View, Evaluation View, App Shell & Docs (`views/documents.py`, `views/evaluation.py`, `app.py`, `README.md`)

**Files:**
- Create: `D:\Workspace\Qocode\AI\chatbox-rag\views\documents.py`
- Create: `D:\Workspace\Qocode\AI\chatbox-rag\views\evaluation.py`
- Modify: `D:\Workspace\Qocode\AI\chatbox-rag\app.py`
- Modify: `D:\Workspace\Qocode\AI\chatbox-rag\README.md`
- Test: `D:\Workspace\Qocode\AI\chatbox-rag\tests\test_app_wiring.py`

**Interfaces:**
- Consumes: All interfaces produced in Tasks 1–4.
- Produces:
  - `views.documents.render_documents_view(bundle: dict) -> None`
  - `views.evaluation.render_evaluation_view(bundle: dict) -> None`
  - `app.py` entrypoint (`streamlit run app.py`)

- [ ] **Step 1: Write the failing smoke & wiring test in `tests/test_app_wiring.py`**

```python
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestAppWiring(unittest.TestCase):
    def test_all_python_modules_compile_and_expose_entrypoints(self):
        expected_files = [
            "app.py",
            "rag_engine_web.py",
            "ui/catalog.py",
            "ui/eval_schema.py",
            "ui/theme.py",
            "ui/state.py",
            "views/chat.py",
            "views/documents.py",
            "views/evaluation.py",
            "scripts/export_eval_to_web.py",
        ]
        for rel in expected_files:
            p = ROOT / rel
            self.assertTrue(p.exists(), f"Missing {rel}")
            ast.parse(p.read_text(encoding="utf-8"), filename=rel)

    def test_requirements_txt_remains_untouched_lightweight_stack(self):
        req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("torch==2.6.0+cpu", req)
        self.assertNotIn("ragas", req.lower())
        self.assertNotIn("plotly", req.lower())
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`& "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m unittest tests/test_app_wiring.py -v`
Expected: FAIL with `AssertionError: False is not true : Missing views/documents.py`.

- [ ] **Step 3: Implement `views/documents.py`, `views/evaluation.py`, `app.py`, and update `README.md`**

- **`views/documents.py`:** Call `summarize_docstore(bundle["db"].docstore._dict)`, render 3 top KPI cards (`18 học liệu`, total chunks, `384 chiều · MiniLM-L12-v2`), category filter pills (`Tất cả` + `CATEGORY_ORDER`) + search input, and 2-column document cards showing title, `doc_id`, chunk count, page range, Drive link button (or `"Đã nạp chỉ mục"` badge when `url is None`), and `"Hỏi thử tài liệu này"` button (`ask_about_document` + `st.rerun()`).
- **`views/evaluation.py`:** Two sub-tabs (`st.tabs(["📊 Báo cáo Quality Gate & RAGAS", "🔬 Live Pipeline Inspector"])`):
  - *Tab 1:* Load `load_default_eval_data()` (or session-uploaded file via `st.file_uploader` accepting `.json`/`.jsonl`/`.csv` parsed by `parse_uploaded_eval`). When `is_sample` is true, show prominent amber banner + expander with ready-to-paste Colab snippet using `scripts/export_eval_to_web.py`. Render overall **Quality Gate PASS/FAIL** badge (`evaluate_gate`), KPI delta cards (comparing `primary_run` vs `baseline_run` when available), `render_pastel_bar_chart_html`, and filterable (`Tất cả` / `Đạt` / `Cần xem lại`) per-question breakdown with explicit caption stating the per-question rule (`Recall@k == 0` or `Faithfulness < 0.85` or `Answer Relevancy < 0.70`).
  - *Tab 2:* Input box + 3 defense preset questions + checkbox `"Sinh luôn câu trả lời LLM để đối chiếu"`. Run `inspect_retrieval(bundle, question)`, display timing pills (`Hybrid Retrieve ms`, `Cohere Rerank ms`, candidate counts), render Top-K selected chunks with `▲ +N` / `▼ -N` / `= 0` rank-delta badges alongside full stage-1 candidate expander.
- **`app.py`:** Set `st.set_page_config(page_title="ELC3008 | Trợ lý học tập AI", page_icon=":material/school:", layout="wide")`, call `inject_global_css()` and `init_state(st.session_state)`, build Figma Left Sidebar (Indigo `ELC3008` brand header, pill `＋ Đoạn chat mới`, 3 navigation buttons updating `st.session_state.active_page`, `Cuộc trò chuyện` list + `Xóa tất cả` + active delete button, and bottom system status card reading `bundle["config"]`), cache `get_rag_bundle()` via `@st.cache_resource`, and route to `render_chat_view`, `render_documents_view`, or `render_evaluation_view`.
- **`README.md`:** Document the 3-page architecture and the 1-step Colab `export_eval_to_web.py` workflow.

- [ ] **Step 4: Run full test suite to verify everything passes**

Run:
`& "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m unittest discover -s tests -v`
Expected: PASS (all test suites green, 0 errors).
