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

    def test_docs_only_advertise_groq_models_the_free_key_can_call(self):
        # 404 "model does not exist or you do not have access": 2 model Llama đã bị
        # Groq chuyển sang Enterprise, tài liệu gọi được chúng là bẫy cho người chạy local.
        dead_ids = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        for rel in ["README.md", ".streamlit/secrets.toml.example"]:
            text = (ROOT / rel).read_text(encoding="utf-8")
            for model_id in dead_ids:
                self.assertNotIn(model_id, text, f"{rel} still lists {model_id}")


if __name__ == "__main__":
    unittest.main()