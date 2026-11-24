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
            "views/conversation.py",
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

    def test_chat_page_is_gone_and_conversation_is_the_landing_page(self):
        self.assertFalse((ROOT / "views" / "chat.py").exists())
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("render_chat_view", src)
        self.assertNotIn("Trợ lý Chat", src)
        self.assertNotIn('("chat"', src)
        self.assertIn('("documents"', src)
        self.assertIn('("inspector"', src)
        state_src = (ROOT / "ui" / "state.py").read_text(encoding="utf-8")
        self.assertIn('setdefault("active_page", "conversation")', state_src)

    def test_nav_buttons_each_have_a_concrete_job(self):
        # Nút "Hỏi đáp" thừa: mở cuộc hội thoại đã vào màn hình chat rồi. Nav chỉ
        # giữ Kho tài liệu + Kiểm tra truy hồi (debug); màn hình hội thoại đi lối sidebar.
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn('"Hỏi đáp")', src)
        self.assertIn('"Kiểm tra truy hồi")', src)
        self.assertIn("render_conversation_view", src)
        self.assertIn('state["active_page"] = "conversation"', src)

    def test_sidebar_navigation_never_double_reruns(self):
        # Click button đã tự rerun một lần; gọi thêm st.rerun() trong handler là
        # render gấp đôi -> nhấp nháy mỗi lần đổi cuộc hội thoại / tạo mới / đổi trang.
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("st.rerun()", src)

    def test_sidebar_shows_live_process_ram(self):
        # Free tier giới hạn RAM; cần thấy RAM thực của process (gồm model + index +
        # BM25) ngay sidebar để canh ngưỡng OOM. Đọc /proc/self/status (container Linux).
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("VmRSS", src)
        self.assertIn("RAM app", src)

    def test_conversation_view_streams_through_the_model_fallback_and_persists_turns(self):
        # Gọi thẳng answer_chain.stream() sẽ bỏ qua bước đổi model/key khi hết hạn mức.
        # Màn hình hội thoại phải stream trực tiếp, lưu lượt hỏi/trả lời và render
        # lại lịch sử + trích dẫn.
        src = (ROOT / "views" / "conversation.py").read_text(encoding="utf-8")
        self.assertIn("stream_answer", src)
        self.assertIn("model_chain", src)
        self.assertIn("st.write_stream", src)
        self.assertIn("append_user_turn", src)
        self.assertIn("append_assistant_turn", src)
        self.assertIn("get_active_conversation", src)
        self.assertIn("render_citation_box_html", src)
        self.assertNotIn('bundle["answer_chain"].stream(', src)
        self.assertNotIn('bundle["answer_chain"].invoke(', src)

    def test_conversation_screen_is_chat_first(self):
        src = (ROOT / "views" / "conversation.py").read_text(encoding="utf-8")
        self.assertIn("st.chat_input", src)
        self.assertIn("st.expander", src)
        self.assertIn("normalize_latex_for_streamlit", src)
        self.assertNotIn("Chi tiết truy hồi của lượt này", src)
        self.assertNotIn("render_retrieval_panel", src)
        self.assertNotIn("st.checkbox", src)
        self.assertNotIn('key="insp_q"', src)

    def test_new_conversation_opens_with_four_big_suggestion_cards(self):
        # Cuộc hội thoại mới mở ra bằng hero + lưới 2x2 thẻ gợi ý lớn (như bản đầu),
        # không phải hàng chip nhỏ.
        src = (ROOT / "views" / "conversation.py").read_text(encoding="utf-8")
        self.assertIn("SUGGESTED_CARDS", src)
        self.assertIn("st.columns(2)", src)
        self.assertIn("Bạn muốn ôn tập phần nào?", src)
        self.assertNotIn("PRESET_QUESTIONS", src)

    def test_conversation_presets_do_not_force_an_extra_render(self):
        # Bấm thẻ gợi ý chỉ queue pending_prompt rồi đi tiếp trong cùng một lần chạy;
        # st.rerun() ở đây làm màn hình render 2 lần (giật).
        src = (ROOT / "views" / "conversation.py").read_text(encoding="utf-8")
        empty_block = src.split("def _render_empty_state")[1].split("def render_conversation_view")[0]
        self.assertIn("pending_prompt", empty_block)
        self.assertNotIn("st.rerun()", empty_block)

    def test_inspector_is_a_dedicated_retrieval_probe(self):
        # Trang nav "Kiểm tra truy hồi": chỉ mổ xẻ retrieve/rerank, không đụng
        # lịch sử hội thoại, không phải màn hình chat.
        src = (ROOT / "views" / "inspector.py").read_text(encoding="utf-8")
        self.assertIn("inspect_retrieval", src)
        self.assertIn("st.button", src)
        self.assertNotIn("append_user_turn", src)
        self.assertNotIn("st.chat_input", src)
        self.assertNotIn("insp_debug", src)

    def test_messages_and_expanders_have_entrance_motion(self):
        from ui.theme import _global_css

        css = _global_css()
        self.assertIn("@keyframes rag-rise", css)
        self.assertIn("animation:rag-rise", css)
        self.assertIn('[data-testid="stChatMessage"] { animation', css)
        self.assertIn('[data-testid="stExpander"]', css)

    def test_conversation_rotates_keys_and_never_prints_one(self):
        # Hết mọi model của một key thì phải sang key kế; key chỉ hiện dưới dạng
        # số thứ tự, in nguyên văn là lộ secret ra giao diện công khai.
        src = (ROOT / "views" / "conversation.py").read_text(encoding="utf-8")
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

    def test_empty_state_slot_is_cleared_before_rendering_first_turn(self):
        src = (ROOT / "views" / "conversation.py").read_text(encoding="utf-8")
        self.assertIn("empty_slot = st.empty()", src)
        self.assertIn("empty_slot.empty()", src)
        self.assertIn("format_selected_context", src)


    def test_system_card_shows_real_embedding_dim_not_hardcoded_384(self):
        # Index bge-m3 là 1024d; sidebar hardcode "384d" sẽ nói sai chiều thật.
        from ui.theme import render_system_card_html

        out = render_system_card_html({"embed_model": "BAAI/bge-m3", "embed_dim": 1024})
        self.assertIn("1024d", out)
        self.assertNotIn("384d", out)

    def test_system_card_defaults_to_384d_for_minilm(self):
        from ui.theme import render_system_card_html

        out = render_system_card_html(
            {"embed_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"})
        self.assertIn("384d", out)

    def test_hero_and_inspector_labels_use_real_embed_dim(self):
        conv = (ROOT / "views" / "conversation.py").read_text(encoding="utf-8")
        insp = (ROOT / "views" / "inspector.py").read_text(encoding="utf-8")
        self.assertNotIn("FAISS 384d", conv)
        self.assertNotIn("FAISS_384d", insp)
        self.assertIn("embed_dim", conv)
        self.assertIn("embed_dim", insp)

    def test_sidebar_buttons_use_on_click_callbacks(self):
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("st.rerun()", src)
        self.assertGreaterEqual(src.count("on_click="), 5)

    def test_conversation_uses_regen_slot_handles_errors_and_reads_top_level_groq_keys(self):
        src = (ROOT / "views" / "conversation.py").read_text(encoding="utf-8")
        self.assertNotIn("retry_after_error", src)
        self.assertIn("regen_slot = st.empty()", src)
        self.assertIn("regen_slot.empty()", src)
        self.assertIn('is_error=True', src)
        self.assertIn('bundle.get("groq_keys")', src)
        try_block = src.split("        try:", 1)[1].split("        except Exception as exc:", 1)[0]
        self.assertIn("inspect_retrieval(bundle, prompt)", try_block)


if __name__ == "__main__":
    unittest.main()