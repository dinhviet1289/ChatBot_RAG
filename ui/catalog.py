"""Danh mục 44 học liệu đã nạp chỉ mục FAISS của ELC3008.

Module thuần (chỉ stdlib), không import Streamlit/LangChain để unit test chạy
được trong môi trường tối giản. doc_id khớp đúng ``manifest.json`` đi kèm
``data/faiss_index_huggingface`` (44 tài liệu, 15.510 chunks); 13 tài liệu có
link Google Drive (giữ nguyên từ ``app.py``), 31 tài liệu còn lại chỉ có trong
chỉ mục. Tiêu đề hiển thị đã làm sạch từ tiêu đề thô trong chỉ mục.
"""

# 13 link Google Drive giữ nguyên văn từ app.py (không sửa URL).
DRIVE_LINKS = {
    "elc3008_ch01": "https://drive.google.com/file/d/1OuFuDhWsIcK4IsV9o7mW02uQfX1AO3fz/view",
    "elc3008_ch02": "https://drive.google.com/file/d/13JG6IubVp0y6a5jEymnxIcUIICyORelA/view",
    "elc3008_ch03": "https://drive.google.com/file/d/1lC2LfXcP0t4E7acxMl56AMXazhvB7gHS/view",
    "elc3008_ch04": "https://drive.google.com/file/d/10GLMg4zfZeKC6LykmgsXTwOYXql_2bMt/view",
    "elc3008_ch05": "https://drive.google.com/file/d/1iAyf50-bbhlDs4f_HjaOPwyhV25kXZgc/view",
    "elc3008_ch06": "https://drive.google.com/file/d/195mzC9tyMAXfAHEGpnPpGi_Gj9i8ZOSA/view",
    "cs188_ch01_search": "https://drive.google.com/file/d/1GlZCchn7FYAdqI7t0dgP1SEm3h30NoOJ/view",
    "cs188_ch02_csps": "https://drive.google.com/file/d/1tcnUk-VTUCybBRoo2kVizoBOMZ9I4r4t/view",
    "cs188_ch03_games": "https://drive.google.com/file/d/1MQKM3silMKER0Hl14S0Zc2GEk1RY_-4D/view",
    "udl_ch15_gans": "https://drive.google.com/file/d/10zOfwOLlxt34Ir_EuJSYkAiyCqKAW9B8/view",
    "udl_ch17_vaes": "https://drive.google.com/file/d/1Q0XeFQPZ6q07qlAPevYNQyCaA9LmV3R6/view",
    "build_a_large_language_model_from_scratc": "https://drive.google.com/file/d/16kEpjYRsEHpP4LZ9xT2cOcUxKExDTtkz/view",
    "hands_on_large_language_models_language_": "https://drive.google.com/file/d/1rvB05bbu9rsoThY7ps3ksZ1RHQ4tk7VL/view",
}

# 8 nhóm học phần theo đúng thứ tự hiển thị của trang "Kho tài liệu".
CATEGORY_ORDER = [
    "Bài giảng ELC3008",
    "UC Berkeley CS188",
    "Deep Learning (UDL)",
    "Large Language Models",
    "Sách AI & ML (English)",
    "Bài giảng Học máy (Tiếng Việt)",
    "AI trong kinh doanh",
    "Giáo trình Tiếng Việt & Cơ sở ML",
]

# Nhóm dự phòng cho doc_id lạ chưa khai báo trong CATALOG.
FALLBACK_CATEGORY = "Tài liệu khác"

# Nhãn ngôn ngữ sạch ở tầng UI (manifest.json của 26 doc mới ghi mặc định "vi").
_EN_DOC_IDS = frozenset({
    "cs188_ch01_search",
    "cs188_ch02_csps",
    "cs188_ch03_games",
    "udl_ch15_gans",
    "udl_ch17_vaes",
    "build_a_large_language_model_from_scratc",
    "hands_on_large_language_models_language_",
    "artificial_intelligence_a_modern_approac",
    "aima_4e_solutions",
    "hands_on_machine_learning_with_scikit_le",
    "introduction_to_machine_learning_with_py",
    "ml_algorithms_bonaccorso",
    "ai_and_machine_learning_for_coders_a_pro",
    "ai_finance_hilpisch",
    "analytical_skills_for_ai_and_data_scienc",
    "ai_engineering_for_true_epub_chip_huyen_",
    "ai_powered_developer_build_software_with",
    "aib_ch01",
    "aib_ch02",
    "aib_ch03",
})


def doc_lang(doc_id):
    """Nhãn ngôn ngữ hiển thị ở UI ('EN' hoặc 'VI'), không sửa manifest gốc."""
    return "EN" if doc_id in _EN_DOC_IDS else "VI"

# Câu từ chối chuẩn của prompt (rule 3/10 trong rag_engine_web.TEMPLATE).
_REFUSAL_MARKER = "không đề cập"

# (doc_id, title, category, sample_question) — đủ 44 doc_id trong chỉ mục.
_ENTRIES = [
    ("elc3008_ch01", "Slide ELC3008 — Chương 1", "Bài giảng ELC3008",
     "Chương 1 trình bày những khái niệm mở đầu nào về trí tuệ nhân tạo trong kinh doanh?"),
    ("elc3008_ch02", "Slide ELC3008 — Chương 2", "Bài giảng ELC3008",
     "Mô hình PEAS gồm những thành phần nào? Cho ví dụ."),
    ("elc3008_ch03", "Slide ELC3008 — Chương 3", "Bài giảng ELC3008",
     "Agent giải quyết vấn đề hoạt động ra sao? So sánh DFS và BFS."),
    ("elc3008_ch04", "Slide ELC3008 — Chương 4", "Bài giảng ELC3008",
     "Tìm kiếm cục bộ và tối ưu hóa khác gì tìm kiếm cổ điển? Giải thích hill climbing."),
    ("elc3008_ch05", "Slide ELC3008 — Chương 5", "Bài giảng ELC3008",
     "Lý thuyết trò chơi (game theory) gồm những khái niệm nào trong chương 5?"),
    ("elc3008_ch06", "Slide ELC3008 — Chương 6", "Bài giảng ELC3008",
     "Bài toán thỏa mãn ràng buộc (CSP) được định nghĩa thế nào? Cho ví dụ."),
    ("cs188_ch01_search", "CS188 Berkeley — Chương 1: Tìm kiếm", "UC Berkeley CS188",
     "So sánh chi tiết thuật toán tìm kiếm DFS và BFS trong AI."),
    ("cs188_ch02_csps", "CS188 Berkeley — Chương 2: CSP", "UC Berkeley CS188",
     "CSP là gì? Giải thích backtracking search và heuristic MRV, LCV."),
    ("cs188_ch03_games", "CS188 Berkeley — Chương 3: Trò chơi đối kháng", "UC Berkeley CS188",
     "Giải thích thuật toán Minimax và cắt tỉa Alpha-Beta trong trò chơi đối kháng."),
    ("udl_ch15_gans", "Understanding Deep Learning — Chương 15: GAN", "Deep Learning (UDL)",
     "Giải thích hàm mục tiêu của mạng GAN và vai trò của generator, discriminator."),
    ("udl_ch17_vaes", "Understanding Deep Learning — Chương 17: VAE", "Deep Learning (UDL)",
     "VAE hoạt động như thế nào? Giải thích biến tiềm ẩn và hàm ELBO."),
    ("build_a_large_language_model_from_scratc", "Build a Large Language Model (From Scratch)",
     "Large Language Models",
     "Quy trình tự xây dựng LLM từ đầu gồm những bước nào (tokenizer, attention, transformer)?"),
    ("hands_on_large_language_models_language_", "Hands-On Large Language Models",
     "Large Language Models",
     "Embedding trong mô hình ngôn ngữ lớn là gì và được ứng dụng ra sao?"),
    ("gt_tri_tue_nhan_tao", "Giáo trình Trí tuệ nhân tạo", "Giáo trình Tiếng Việt & Cơ sở ML",
     "Giáo trình Trí tuệ nhân tạo giới thiệu những khái niệm nền tảng nào?"),
    ("ttnt_nguyen_quang_hoan", "Trí tuệ nhân tạo — Nguyễn Quang Hoan", "Giáo trình Tiếng Việt & Cơ sở ML",
     "Sách Trí tuệ nhân tạo của Nguyễn Quang Hoan trình bày các phương pháp tìm kiếm nào?"),
    ("ttnt_tu_minh_phuong", "Trí tuệ nhân tạo — Từ Minh Phương", "Giáo trình Tiếng Việt & Cơ sở ML",
     "Tìm kiếm theo chiều sâu (DFS) được mô tả thế nào trong sách của Từ Minh Phương?"),
    ("it3190_tai_lieu_doc", "IT3190 — Tài liệu đọc", "Giáo trình Tiếng Việt & Cơ sở ML",
     "Tài liệu đọc IT3190 bổ trợ những kiến thức nào về trí tuệ nhân tạo?"),
    ("ml_co_ban", "Machine Learning cơ bản", "Giáo trình Tiếng Việt & Cơ sở ML",
     "Phân biệt học có giám sát và học không giám sát trong machine learning cơ bản."),

    # --- 26 học liệu mới (benchmark đầy đủ 44 file) ---
    # Sách AI & ML tiếng Anh (8).
    ("artificial_intelligence_a_modern_approac",
     "Artificial Intelligence: A Modern Approach (Russell & Norvig)", "Sách AI & ML (English)",
     "Tác tử hợp lý (rational agent) và các loại môi trường được định nghĩa thế nào trong AIMA?"),
    ("aima_4e_solutions", "AIMA 4e — Lời giải bài tập (Russell & Norvig)", "Sách AI & ML (English)",
     "Hướng dẫn giải bài tập về tác tử thông minh và môi trường trong AIMA 4e?"),
    ("hands_on_machine_learning_with_scikit_le",
     "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow", "Sách AI & ML (English)",
     "Quy trình huấn luyện mô hình với Scikit-Learn và pipeline tiền xử lý dữ liệu?"),
    ("introduction_to_machine_learning_with_py",
     "Introduction to Machine Learning with Python", "Sách AI & ML (English)",
     "Các thuật toán học có giám sát cơ bản được giới thiệu thế nào trong Python?"),
    ("ml_algorithms_bonaccorso", "Machine Learning Algorithms — Bonaccorso", "Sách AI & ML (English)",
     "Các thuật toán học máy chính được phân loại và so sánh ra sao?"),
    ("ai_and_machine_learning_for_coders_a_pro",
     "AI and Machine Learning for Coders", "Sách AI & ML (English)",
     "Quy trình xây dựng mô hình học máy dành cho lập trình viên gồm những bước nào?"),
    ("ai_finance_hilpisch", "AI in Finance — Yves Hilpisch", "Sách AI & ML (English)",
     "Học máy được ứng dụng trong tài chính (định giá, quản trị rủi ro) như thế nào?"),
    ("analytical_skills_for_ai_and_data_scienc",
     "Analytical Skills for AI and Data Science", "Sách AI & ML (English)",
     "Kỹ năng phân tích cần thiết cho khoa học dữ liệu và AI gồm những gì?"),

    # Bài giảng Học máy tiếng Việt L01–L11 (11).
    ("l01_gioi_thieu", "Học máy — Bài 1: Giới thiệu", "Bài giảng Học máy (Tiếng Việt)",
     "Phân biệt học máy (machine learning) và khai phá dữ liệu (data mining) thế nào?"),
    ("l02_tien_xu_ly", "Học máy — Bài 2: Tiền xử lý dữ liệu", "Bài giảng Học máy (Tiếng Việt)",
     "Các bước tiền xử lý dữ liệu trước khi huấn luyện mô hình học máy?"),
    ("l03_danh_gia", "Học máy — Bài 3: Đánh giá mô hình", "Bài giảng Học máy (Tiếng Việt)",
     "Đánh giá hiệu năng mô hình học máy bằng thực nghiệm được thực hiện ra sao?"),
    ("l04_hoi_quy", "Học máy — Bài 4: Hồi quy", "Bài giảng Học máy (Tiếng Việt)",
     "Bài toán hồi quy (regression) và các độ đo đánh giá mô hình hồi quy?"),
    ("l05_phan_lop_knn", "Học máy — Bài 5: Phân lớp & KNN", "Bài giảng Học máy (Tiếng Việt)",
     "Thuật toán KNN và bài toán phân lớp (classification) được trình bày thế nào?"),
    ("l06_xac_suat", "Học máy — Bài 6: Phương pháp xác suất", "Bài giảng Học máy (Tiếng Việt)",
     "Phương pháp thống kê xác suất cho bài toán phân loại (Naive Bayes) hoạt động ra sao?"),
    ("l07_cay_quyet_dinh", "Học máy — Bài 7: Cây quyết định", "Bài giảng Học máy (Tiếng Việt)",
     "Cây quyết định chọn thuộc tính phân tách bằng độ lợi thông tin như thế nào?"),
    ("l08_mang_noron", "Học máy — Bài 8: Mạng nơ-ron", "Bài giảng Học máy (Tiếng Việt)",
     "Mạng nơ-ron nhân tạo (ANN) và thuật toán lan truyền ngược hoạt động ra sao?"),
    ("l09_svm", "Học máy — Bài 9: SVM", "Bài giảng Học máy (Tiếng Việt)",
     "Máy vectơ hỗ trợ (SVM) tìm siêu phẳng phân cách tối ưu như thế nào?"),
    ("l10_phan_cum", "Học máy — Bài 10: Phân cụm", "Bài giảng Học máy (Tiếng Việt)",
     "Các phương pháp phân cụm (K-means) và cách đánh giá chất lượng cụm?"),
    ("l11_luat_ket_hop", "Học máy — Bài 11: Luật kết hợp", "Bài giảng Học máy (Tiếng Việt)",
     "Khai phá luật kết hợp (association rule) và thuật toán Apriori hoạt động thế nào?"),

    # AI trong kinh doanh — bộ AIB (3).
    ("aib_ch01", "AI in Business — Chương 1", "AI trong kinh doanh",
     "Trí tuệ nhân tạo trong kinh doanh được định nghĩa và phân loại thế nào?"),
    ("aib_ch02", "AI in Business — Chương 2", "AI trong kinh doanh",
     "Các phương pháp xấp xỉ mối quan hệ giữa X và y trong học máy là gì?"),
    ("aib_ch03", "AI in Business — Chương 3", "AI trong kinh doanh",
     "AI tạo tín hiệu hỗ trợ quyết định thị trường và kinh doanh ra sao?"),

    # Bổ sung nhóm LLM (2) và giáo trình tiếng Việt (2).
    ("ai_engineering_for_true_epub_chip_huyen_", "AI Engineering — Chip Huyen", "Large Language Models",
     "Kỹ thuật prompt engineering và đánh giá mô hình ngôn ngữ lớn được trình bày thế nào?"),
    ("ai_powered_developer_build_software_with",
     "The AI-Powered Developer — ChatGPT & Copilot", "Large Language Models",
     "ChatGPT và Copilot hỗ trợ lập trình viên xây dựng phần mềm như thế nào?"),
    ("ai_for_student_vi", "AI for Students (Tiếng Việt)", "Giáo trình Tiếng Việt & Cơ sở ML",
     "Sinh viên nên bắt đầu học trí tuệ nhân tạo từ những khái niệm nền tảng nào?"),
    ("machine_learning_andrew_ng_pdf_ban_dich_",
     "Machine Learning Yearning — Andrew Ng (Bản dịch)", "Giáo trình Tiếng Việt & Cơ sở ML",
     "Chiến lược đặt mục tiêu và cải thiện mô hình học máy theo Andrew Ng gồm những gì?"),
]

CATALOG = {
    doc_id: {
        "title": title,
        "category": category,
        "url": DRIVE_LINKS.get(doc_id),  # None với 5 giáo trình không có link Drive
        "sample_question": sample_question,
    }
    for doc_id, title, category, sample_question in _ENTRIES
}


def trang_label(meta):
    """Nhãn trang ``"s"`` / ``"s-e"``; ``"?"`` khi thiếu cả page_start/page_end.

    Không raise KeyError/TypeError với metadata thiếu hoặc None.
    """
    start, end = meta.get("page_start"), meta.get("page_end")
    if start is None and end is None:
        return "?"
    if start is None:
        start = end
    if end is None:
        end = start
    return str(start) if start == end else f"{start}-{end}"


def summarize_docstore(docstore_dict):
    """Gộp ``db.docstore._dict`` theo doc_id -> danh sách đã sắp xếp.

    Mỗi phần tử: {doc_id, title, category, lang, url, sample_question, chunk_count,
    page_min, page_max}. doc_id lạ rơi vào nhóm "Tài liệu khác"; chunk thiếu
    page_start/page_end (None) vẫn được đếm, page_min/page_max = None khi
    không có trang nào. Sắp xếp theo CATEGORY_ORDER rồi tới tiêu đề.
    """
    groups = {}
    for doc in (docstore_dict or {}).values():
        meta = getattr(doc, "metadata", None) or {}
        doc_id = meta.get("doc_id") or "unknown"
        group = groups.setdefault(doc_id, {"chunk_count": 0, "pages": [], "doc_title": None})
        group["chunk_count"] += 1
        if not group["doc_title"] and meta.get("doc_title"):
            group["doc_title"] = meta["doc_title"]
        for key in ("page_start", "page_end"):
            value = meta.get(key)
            if isinstance(value, (int, float)):
                group["pages"].append(value)

    rows = []
    for doc_id, group in groups.items():
        known = CATALOG.get(doc_id)
        pages = group["pages"]
        rows.append({
            "doc_id": doc_id,
            "title": known["title"] if known else (group["doc_title"] or doc_id),
            "category": known["category"] if known else FALLBACK_CATEGORY,
            "lang": doc_lang(doc_id),
            "url": known["url"] if known else DRIVE_LINKS.get(doc_id),
            "sample_question": known["sample_question"] if known else None,
            "chunk_count": group["chunk_count"],
            "page_min": min(pages) if pages else None,
            "page_max": max(pages) if pages else None,
        })

    order = {category: i for i, category in enumerate(CATEGORY_ORDER)}
    rows.sort(key=lambda r: (order.get(r["category"], len(CATEGORY_ORDER)), r["title"], r["doc_id"]))
    return rows


def should_show_sources(answer):
    """False khi câu trả lời rỗng/chỉ khoảng trắng hoặc là lời từ chối.

    Khớp "không đề cập" không phân biệt hoa thường (vd "KHÔNG ĐỀ CẬP") để
    không hiển thị trích dẫn gây hiểu nhầm ở lượt từ chối.
    """
    if not answer or not str(answer).strip():
        return False
    return _REFUSAL_MARKER not in str(answer).lower()


def serialize_sources(docs):
    """Chuyển Document truy hồi -> dict nguồn trích dẫn bền vững cho state.

    Mỗi phần tử: {doc_id, doc_title, trang, url, snippet, rerank_score};
    snippet giữ nguyên page_content thô (việc làm sạch/escape thuộc ui.theme).
    Cùng một tài liệu + trang chỉ hiện một lần (nhiều chunk của cùng trang thường
    lọt vào top-k), giữ snippet của chunk có điểm rerank cao nhất.
    """
    sources = []
    for doc in docs or []:
        meta = getattr(doc, "metadata", None) or {}
        doc_id = meta.get("doc_id") or "?"
        known = CATALOG.get(doc_id) or {}
        sources.append({
            "doc_id": doc_id,
            "doc_title": known.get("title") or meta.get("doc_title") or doc_id,
            "trang": trang_label(meta),
            "url": DRIVE_LINKS.get(doc_id),
            "snippet": getattr(doc, "page_content", "") or "",
            "rerank_score": meta.get("rerank_score"),
        })

    merged = {}
    for src in sources:
        best = merged.get((src["doc_id"], src["trang"]))
        if best is None or _score(src) > _score(best):
            merged[(src["doc_id"], src["trang"])] = src
    return list(merged.values())


def _score(src):
    return src["rerank_score"] if src["rerank_score"] is not None else -1.0


def sources_from_inspection(selected):
    """Chiếu các item ``selected`` của inspect_retrieval thành nguồn trích dẫn.

    Cùng quy tắc gộp như serialize_sources: mỗi (doc_id, trang) một nguồn, giữ
    snippet có điểm rerank cao nhất cùng các trường xếp hạng (initial_rank,
    final_rank, rank_delta) để hiển thị trong hộp nguồn trích dẫn.
    """
    merged = {}
    for item in selected or []:
        src = {k: item.get(k) for k in
               ("doc_id", "doc_title", "trang", "url", "snippet", "rerank_score",
                "initial_rank", "final_rank", "rank_delta")}
        key = (src["doc_id"], src["trang"])
        best = merged.get(key)
        if best is None or _score(src) > _score(best):
            merged[key] = src
    return list(merged.values())


def format_selected_context(selected):
    """Định dạng các mục selected kèm nhãn [Tên tài liệu (doc_id), trang X-Y] khớp Rule 8."""
    parts = []
    for item in selected or []:
        doc_id = item.get("doc_id") or "Unknown"
        known = CATALOG.get(doc_id) or {}
        name = known.get("title") or item.get("doc_title") or doc_id
        trang = str(item.get("trang") or "?").strip() or "?"
        snippet = item.get("snippet") or ""
        parts.append(f"[{name} ({doc_id}), trang {trang}]\n{snippet}")
    return "\n\n".join(parts)