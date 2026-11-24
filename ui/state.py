"""Pure multi-conversation chat state helpers (no Streamlit import)."""

import uuid

DEFAULT_TITLE = "Cuộc hội thoại mới"
TITLE_MAX = 38

SUGGESTED_CARDS = [
    {"icon": "school", "title": "Khái niệm nền tảng",
     "subtitle": "PEAS, tác tử, môi trường",
     "prompt": "Mô hình PEAS gồm những thành phần nào? Cho ví dụ."},
    {"icon": "account_tree", "title": "Thuật toán tìm kiếm",
     "subtitle": "DFS, BFS, A*, UCS",
     "prompt": "So sánh thuật toán DFS và BFS."},
    {"icon": "psychology", "title": "Học sâu & LLM",
     "subtitle": "GAN, VAE, Transformer",
     "prompt": "LLM là gì? Kiến trúc gồm những thành phần nào?"},
    {"icon": "casino", "title": "CSP & đối kháng",
     "subtitle": "N-Queens, Minimax",
     "prompt": "Giải thích bài toán N-Queens dưới dạng CSP."},
]


def _new_conv():
    return {"id": uuid.uuid4().hex[:8], "title": DEFAULT_TITLE, "messages": []}


def init_state(state):
    if not state.get("conversations"):
        state["conversations"] = [_new_conv()]
    ids = [c["id"] for c in state["conversations"]]
    if state.get("active_conv_id") not in ids:
        state["active_conv_id"] = ids[0]
    state.setdefault("active_page", "conversation")
    state.setdefault("pending_prompt", None)


def get_active_conversation(state):
    init_state(state)
    for c in state["conversations"]:
        if c["id"] == state["active_conv_id"]:
            return c
    state["active_conv_id"] = state["conversations"][0]["id"]
    return state["conversations"][0]


def create_new_conversation(state):
    init_state(state)
    conv = _new_conv()
    state["conversations"].insert(0, conv)
    state["active_conv_id"] = conv["id"]
    return conv["id"]


def clear_all_conversations(state):
    state["conversations"] = [_new_conv()]
    state["active_conv_id"] = state["conversations"][0]["id"]
    state["pending_prompt"] = None


def delete_conversation(state, conv_id):
    init_state(state)
    state["conversations"] = [c for c in state["conversations"] if c["id"] != conv_id]
    if not state["conversations"]:
        state["conversations"] = [_new_conv()]
    ids = [c["id"] for c in state["conversations"]]
    if state.get("active_conv_id") not in ids:
        state["active_conv_id"] = ids[0]


def append_user_turn(state, prompt):
    conv = get_active_conversation(state)
    if conv["title"] == DEFAULT_TITLE:
        conv["title"] = " ".join(prompt.split())[:TITLE_MAX]
    conv["messages"].append({"role": "user", "content": prompt})


def append_assistant_turn(state, content, sources, latency_s=0.0,
                          telemetry=None, retrieval=None, is_error=False):
    conv = get_active_conversation(state)
    msg = {
        "role": "assistant",
        "content": content,
        "sources": list(sources or []),
        "latency_s": latency_s,
    }
    if telemetry:
        msg["telemetry"] = dict(telemetry)
    if retrieval:
        msg["retrieval"] = dict(retrieval)
    if is_error:
        msg["is_error"] = True
    conv["messages"].append(msg)


def prepare_regenerate(state):
    msgs = get_active_conversation(state)["messages"]
    if not msgs or msgs[-1]["role"] != "assistant":
        return None
    msgs.pop()
    if msgs and msgs[-1]["role"] == "user":
        return msgs.pop()["content"]
    return None


def ask_about_document(state, doc_id, prompt):
    init_state(state)
    state["active_page"] = "conversation"
    state["pending_prompt"] = prompt