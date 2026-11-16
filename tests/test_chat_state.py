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

    def test_ask_about_document_switches_to_inspector_with_pending_prompt(self):
        state = {}
        init_state(state)
        state["active_page"] = "documents"
        ask_about_document(state, "udl_ch15_gans", "Giải thích hàm mục tiêu của mạng GAN.")
        self.assertEqual(state["active_page"], "inspector")
        self.assertEqual(state["pending_prompt"], "Giải thích hàm mục tiêu của mạng GAN.")

    def test_init_state_lands_on_documents_page(self):
        state = {}
        init_state(state)
        self.assertEqual(state["active_page"], "documents")

    def test_new_conversations_are_listed_newest_first(self):
        state = {}
        init_state(state)
        first_id = state["active_conv_id"]
        append_user_turn(state, "Câu hỏi đầu tiên")
        second_id = create_new_conversation(state)
        third_id = create_new_conversation(state)
        self.assertEqual([c["id"] for c in state["conversations"]],
                         [third_id, second_id, first_id])
        self.assertEqual(state["active_conv_id"], third_id)


if __name__ == "__main__":
    unittest.main()