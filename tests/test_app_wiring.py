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


if __name__ == "__main__":
    unittest.main()