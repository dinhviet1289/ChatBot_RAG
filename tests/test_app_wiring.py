import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestAppWiring(unittest.TestCase):
    def test_all_python_modules_compile_and_expose_entrypoints(self):
        expected_files = [
            "app.py",
            "rag_engine_web.py",
            "ui/catalog.py",
            "ui/theme.py",
            "ui/state.py",
            "views/documents.py",
            "views/inspector.py",
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

    def test_nav_routes_to_inspector_and_keeps_no_eval_page(self):
        # Báo cáo Quality Gate/RAGAS sống trong Colab; web chỉ cần chỗ debug truy hồi.
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('("inspector"', src)
        self.assertIn("render_inspector_view", src)
        self.assertNotIn("evaluation", src)
        self.assertNotIn("Đánh giá RAG", src)

    def test_chat_page_is_gone_and_documents_is_the_landing_page(self):
        self.assertFalse((ROOT / "views" / "chat.py").exists())
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("render_chat_view", src)
        self.assertNotIn("Trợ lý Chat", src)
        self.assertNotIn('("chat"', src)
        self.assertIn('("documents"', src)
        self.assertIn('("inspector"', src)
        state_src = (ROOT / "ui" / "state.py").read_text(encoding="utf-8")
        self.assertIn('setdefault("active_page", "documents")', state_src)

    def test_inspector_streams_through_the_model_fallback_and_persists_turns(self):
        # Gọi thẳng answer_chain.stream() sẽ bỏ qua bước đổi model/key khi hết hạn mức.
        # Inspector giờ là mặt trận hỏi đáp chính: phải stream trực tiếp, lưu lượt
        # hỏi/trả lời vào cuộc hội thoại và render lại lịch sử + trích dẫn.
        src = (ROOT / "views" / "inspector.py").read_text(encoding="utf-8")
        self.assertIn("stream_answer", src)
        self.assertIn("model_chain", src)
        self.assertIn("st.write_stream", src)
        self.assertIn("append_user_turn", src)
        self.assertIn("append_assistant_turn", src)
        self.assertIn("get_active_conversation", src)
        self.assertIn("render_citation_box_html", src)
        self.assertNotIn('bundle["answer_chain"].stream(', src)
        self.assertNotIn('bundle["answer_chain"].invoke(', src)

    def test_inspector_is_conversation_first_not_a_query_form(self):
        # Bấm "Cuộc hội thoại mới" phải ra màn hình chat (ô nhập dưới đáy, Enter để
        # gửi), không phải form truy vấn bắt bấm "Chạy"; số liệu debug nằm trong expander.
        src = (ROOT / "views" / "inspector.py").read_text(encoding="utf-8")
        self.assertIn("st.chat_input", src)
        self.assertIn("st.expander", src)
        self.assertNotIn("st.checkbox", src)
        self.assertNotIn('key="insp_q"', src)

    def test_inspector_rotates_keys_and_never_prints_one(self):
        # Hết mọi model của một key thì phải sang key kế; key chỉ hiện dưới dạng
        # số thứ tự, in nguyên văn là lộ secret ra giao diện công khai.
        src = (ROOT / "views" / "inspector.py").read_text(encoding="utf-8")
        self.assertIn("rotation_pairs", src)
        self.assertIn("groq_keys", src)
        self.assertNotIn("GROQ_API_KEY", src)

    def test_global_css_has_smooth_motion_and_palette_scrollbars(self):
        from ui.theme import _global_css

        css = _global_css()
        self.assertIn("prefers-reduced-motion", css)
        self.assertIn("scrollbar", css)
        self.assertIn("transition", css)
        self.assertIn("focus-visible", css)
        # Ô chat dưới đáy là input chính của màn hình hội thoại, phải được style.
        self.assertIn("stChatInput", css)

    def test_secrets_example_documents_the_key_array(self):
        text = (ROOT / ".streamlit/secrets.toml.example").read_text(encoding="utf-8")
        self.assertIn("GROQ_API_KEYS", text)

    def test_answer_factory_accepts_a_key_and_passes_it_to_chatgroq(self):
        # langchain không cài ở môi trường test nên không gọi thử được; sai chữ ký
        # ở đây chỉ nổ khi lên cloud (TypeError lúc hỏi câu đầu tiên).
        import ast

        tree = ast.parse((ROOT / "rag_engine_web.py").read_text(encoding="utf-8"))
        factory = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == "make_answer_chain")
        self.assertEqual([a.arg for a in factory.args.args], ["model", "key"])
        calls = [n for n in ast.walk(factory)
                 if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "ChatGroq"]
        self.assertEqual(len(calls), 1)
        self.assertIn("api_key", [kw.arg for kw in calls[0].keywords])

    def test_docs_only_advertise_groq_models_the_free_key_can_call(self):
        # 404 "model does not exist or you do not have access": 2 model Llama đã bị
        # Groq chuyển sang Enterprise, tài liệu gọi được chúng là bẫy cho người chạy local.
        dead_ids = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        for rel in ["README.md", ".streamlit/secrets.toml.example"]:
            text = (ROOT / rel).read_text(encoding="utf-8")
            for model_id in dead_ids:
                self.assertNotIn(model_id, text, f"{rel} still lists {model_id}")

    def test_docs_advertise_the_same_default_model_the_code_uses(self):
        # Bản deploy không ghi LLM_MODEL trong Secrets, nên default trong code là cấu hình
        # thật; README/mẫu Secrets nói sai model khiến người dùng sửa nhầm chỗ.
        import rag_engine_web

        documented = {
            "README.md": re.search(r"Groq `([^`]+)`", (ROOT / "README.md").read_text(encoding="utf-8")),
            ".streamlit/secrets.toml.example": re.search(r'LLM_MODEL = "([^"]+)"',
                                                         (ROOT / ".streamlit/secrets.toml.example").read_text(encoding="utf-8")),
        }
        for rel, match in documented.items():
            self.assertIsNotNone(match, f"{rel} no longer names its default model")
            self.assertEqual(match.group(1), rag_engine_web.LLM_MODEL, f"{rel} documents a stale default")


if __name__ == "__main__":
    unittest.main()