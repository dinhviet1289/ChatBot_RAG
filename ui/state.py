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
    state.setdefault("theme_mode", "light")


def toggle_theme_mode(state):
    """Toggle UI theme between 'light' and 'dark' in session state."""
    current = str(state.get("theme_mode") or "light").strip().lower()
    state["theme_mode"] = "dark" if current != "dark" else "light"
    return state["theme_mode"]


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


def last_user_question(state):
    msgs = get_active_conversation(state)["messages"]
    for m in reversed(msgs):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return None


def open_in_inspector(state, question):
    init_state(state)
    q = (question or "").strip()
    if not q:
        return
    state["dbg_q"] = q
    state["dbg_auto"] = True
    state["active_page"] = "inspector"


_FOLLOWUP_CLUSTERS = [
    (
        ("peas", "tác tử", "agent", "môi trường"),
        [
            "Áp dụng mô hình PEAS cho một bài toán kinh doanh hoặc xe tự hành?",
            "Phân biệt các loại môi trường (toàn phần vs bán phần, xác định vs ngẫu nhiên)?",
            "So sánh tác tử phản xạ đơn giản với tác tử dựa trên mục tiêu và tiện ích?",
        ],
    ),
    (
        ("dfs", "bfs", "ucs", "tìm kiếm mù", "uninformed"),
        [
            "Khi nào nên ưu tiên UCS thay vì BFS trong bài toán tìm đường?",
            "So sánh độ phức tạp thời gian, bộ nhớ và tính tối ưu của DFS, BFS, UCS?",
            "Ví dụ minh họa từng bước duyệt đồ thị bằng DFS và BFS?",
        ],
    ),
    (
        ("a*", "heuristic", "greedy", "admissible", "consistent", "cục bộ", "hill climbing"),
        [
            "Thế nào là hàm heuristic chấp nhận được (admissible) và nhất quán?",
            "A* khắc phục nhược điểm của Greedy Best-First Search như thế nào?",
            "Hill Climbing dễ kẹt ở cực trị địa phương ra sao và cách khắc phục?",
        ],
    ),
    (
        ("minimax", "alpha-beta", "trò chơi", "đối kháng", "game theory"),
        [
            "Cắt tỉa Alpha-Beta giúp giảm số nút duyệt nhưng giữ nguyên kết quả thế nào?",
            "Thứ tự xét nút ảnh hưởng ra sao đến hiệu quả cắt tỉa Alpha-Beta?",
            "Expectimax xử lý yếu tố ngẫu nhiên khác Minimax ở điểm nào?",
        ],
    ),
    (
        ("csp", "ràng buộc", "n-queens", "backtracking", "mrv", "lcv", "ac-3"),
        [
            "Heuristic MRV, Degree và LCV tăng tốc quay lui CSP như thế nào?",
            "Thuật toán AC-3 (arc consistency) phát hiện mâu thuẫn trước khi gán biến ra sao?",
            "Phân biệt Forward Checking và AC-3 trên bài toán tô màu bản đồ?",
        ],
    ),
    (
        ("gan", "vae", "elbo", "generator", "discriminator"),
        [
            "So sánh cơ chế sinh dữ liệu và hàm mất mát giữa VAE và GAN?",
            "Hiện tượng mode collapse trong GAN là gì và vì sao xảy ra?",
            "Ý nghĩa của hai thành phần tái tạo và KL-divergence trong hàm ELBO?",
        ],
    ),
    (
        ("llm", "transformer", "attention", "gpt", "tokenizer", "rag", "embedding"),
        [
            "Cơ chế Self-Attention tính trọng số truy vấn Q, K, V như thế nào?",
            "RAG giảm ảo giác của LLM so với fine-tuning ở điểm nào?",
            "Tokenizer (BPE) và vector embedding đóng vai trò gì trong kiến trúc LLM?",
        ],
    ),
    (
        ("svm", "vectơ hỗ trợ", "knn", "cây quyết định", "naive bayes", "nơ-ron", "neural", "backprop"),
        [
            "Kernel trick trong SVM giải quyết dữ liệu không tách tuyến tính thế nào?",
            "Cách chọn thuộc tính phân tách bằng Entropy và Information Gain ở cây quyết định?",
            "Thuật toán lan truyền ngược (backpropagation) cập nhật trọng số ra sao?",
        ],
    ),
    (
        ("hồi quy", "regression", "phân cụm", "k-means", "apriori", "luật kết hợp", "đánh giá", "overfitting"),
        [
            "Cách tránh overfitting bằng regularization và cross-validation?",
            "So sánh chỉ số Accuracy, Precision, Recall và F1-score khi dữ liệu lệch?",
            "K-Means nhạy cảm với khởi tạo tâm cụm như thế nào?",
        ],
    ),
    (
        ("kinh doanh", "doanh nghiệp", "tài chính", "finance", "business", "thị trường"),
        [
            "Những rủi ro và bài toán quản trị khi triển khai AI vào doanh nghiệp?",
            "Ví dụ ứng dụng AI trong dự báo nhu cầu, định giá hoặc quản trị rủi ro?",
            "Cách đo lường ROI và hiệu quả kinh doanh của mô hình học máy?",
        ],
    ),
]


def build_followup_questions(prompt, sources=None, history_questions=None, answer=""):
    """Build 3 context-aware, distinct Vietnamese follow-up suggestions (pure stdlib)."""
    from ui.catalog import CATALOG

    hay = f"{prompt or ''} {(answer or '')[:400]}".casefold()
    existing = {str(prompt or "").strip().casefold()}
    for q in history_questions or []:
        if q:
            existing.add(str(q).strip().casefold())

    candidates = []
    for keywords, suggestions in _FOLLOWUP_CLUSTERS:
        if any(kw in hay for kw in keywords):
            candidates.extend(suggestions)

    for src in sources or []:
        did = src.get("doc_id")
        sq = (CATALOG.get(did) or {}).get("sample_question")
        if sq:
            candidates.append(sq)

    candidates.extend([
        "Tóm tắt ý chính thành bảng so sánh ngắn gọn để ôn thi?",
        "Cho một ví dụ bài tập minh họa chi tiết kèm lời giải từng bước?",
        "Những lỗi sai sinh viên hay gặp nhất khi học chủ đề này?",
    ])

    picked = []
    seen = set(existing)
    for cand in candidates:
        norm = cand.strip().casefold()
        if not cand or norm in seen:
            continue
        seen.add(norm)
        picked.append(cand.strip())
        if len(picked) == 3:
            break
    return picked