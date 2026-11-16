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
            "views/chat.py",
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

    def test_chat_view_streams_through_the_model_fallback(self):
        # Gọi thẳng answer_chain.stream() sẽ bỏ qua bước đổi model khi hết hạn mức,
        # tức lỗi "hết token" quay lại đúng lúc nhiều người dùng cùng hỏi.
        src = (ROOT / "views" / "chat.py").read_text(encoding="utf-8")
        self.assertIn("stream_answer", src)
        self.assertIn("model_chain", src)
        self.assertNotIn('bundle["answer_chain"].stream(', src)
        insp = (ROOT / "views" / "inspector.py").read_text(encoding="utf-8")
        self.assertNotIn('bundle["answer_chain"].invoke(', insp)
        self.assertIn("stream_answer", insp)

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