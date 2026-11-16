# Thiết kế Giao diện Figma Indigo & Trang Đánh giá Thực nghiệm RAG — ELC3008

- **Ngày:** 2026-11-15
- **Dự án:** ELC3008 — Trợ lý học tập AI (Khoa Thương mại Điện tử · ĐH Kinh tế — ĐHĐN)
- **Repo mục tiêu:** `D:\Workspace\Qocode\AI\chatbox-rag` (`dinhviet1289/ChatBot_RAG` -> `https://chatbotragdue.streamlit.app/`)

---

## 1. Mục tiêu & Lý do nâng cấp

Sản phẩm chuẩn bị cho buổi **bảo vệ đồ án môn học**. Pipeline lõi (BM25 + FAISS + Cohere Rerank + Groq LLM) đã chạy ổn định trên Streamlit Cloud, nhưng cần giải quyết 2 vấn đề lớn:

1. **Làm mới toàn bộ giao diện (`app.py`):**
   - Thay giao diện ~120 dòng đơn điệu hiện tại bằng thiết kế sáng, hiện đại theo mẫu **Figma AI Chat Web UI (`1373386017446585785`)** kết hợp trải nghiệm màn hình chào từ **`Dphenomenal101/rag-chatbot`**.
   - Sửa triệt để lỗi **mất khung Nguồn trích dẫn của câu cũ khi hỏi câu mới**, bổ sung **streaming câu trả lời**, nút **`↻ Tạo lại câu trả lời`**, và quản lý nhiều đoạn hội thoại trong phiên.
2. **Tích hợp trang Thực nghiệm & Đánh giá RAG ngay trên cùng 1 link Web:**
   - Hội đồng chỉ mở **1 URL duy nhất** trên máy chiếu là xem được cả Chatbot, Kho 18 học liệu, Báo cáo điểm Quality Gate/RAGAS (khớp 100% logic `eval_rag.py` trong `D:\AI_Tested4.ipynb`), và công cụ **Live Pipeline Inspector** mổ xẻ trực tiếp kết quả truy hồi trước/sau Cohere Rerank.

---

## 2. Ràng buộc kỹ thuật & Các hướng đã cân nhắc

- **Giới hạn hạ tầng:** Streamlit Community Cloud Free chỉ có **~1GB RAM**. Không cài thêm thư viện nặng ngoài `requirements.txt` hiện tại; không chạy batch `ragas` trực tiếp trên web Cloud; chỉ khởi tạo bộ model/index **đúng 1 lần duy nhất** qua `@st.cache_resource`.
- **Các hướng tiếp cận:**
  1. **Hướng A — Đa trang qua Sidebar + Hệ CSS Indigo chuẩn Figma + Tách module `ui/` & `views/` *(Đã chọn)***: Đỉnh màn hình thoáng đãng đúng bố cục Figma, chỉ thực thi trang đang mở, code tách file sạch đẹp, tốn 0MB RAM phụ trội.
  2. **Hướng B — Giữ 1 file `app.py` dùng `st.tabs` ngang ở đầu trang *(Loại)***: Chiếm chỗ đầu trang và chạy ngầm cả 3 tab mỗi lần gửi tin.
  3. **Hướng C — Nhúng toàn bộ giao diện qua iframe HTML/JS *(Loại)***: Mất `st.chat_input` gốc và streaming tự nhiên của Streamlit.

---

## 3. Cấu trúc thư mục dự kiến

```text
chatbox-rag/
├── app.py                        # Entrypoint: cấu hình trang, nạp RAG cache, Sidebar & chuyển trang
├── rag_engine_web.py             # Giữ get_rag() cũ + thêm get_rag_bundle() & inspect_retrieval()
├── ui/
│   ├── __init__.py
│   ├── theme.py                  # CSS biến màu Indigo Figma, card, pill, biểu đồ cột tròn HTML/CSS
│   └── catalog.py                # Danh mục 18 tài liệu (5 nhóm học phần, link Drive, câu hỏi gợi ý)
├── views/
│   ├── __init__.py
│   ├── chat.py                   # Trang 1: Trợ lý Chat (Overview 2x2 cards, Streaming, Citations, Regenerate)
│   ├── documents.py              # Trang 2: Kho 18 tài liệu (Thống kê chunks thực tế + Lọc nhóm)
│   └── evaluation.py             # Trang 3: Đánh giá RAG (Quality Gate/RAGAS Dashboard + Live Inspector)
├── data/
│   ├── faiss_index_huggingface/  # Giữ nguyên index.faiss + index.pkl
│   └── eval_results.json         # Dữ liệu báo cáo Eval (mặc định gắn cờ is_sample=true tới khi nạp số thật)
├── scripts/
│   └── export_eval_to_web.py     # Script/hàm dùng trên Colab gom eval_report.json + eval_runs.jsonl sang web
├── tests/
│   └── test_ui_and_eval_logic.py # Unit tests kiểm tra parser eval, phân loại PASS/FAIL, catalog & inspector
├── requirements.txt              # Giữ nguyên 100% không thêm package
└── README.md                     # Cập nhật tài liệu kiến trúc UI & quy trình cập nhật điểm Eval từ Colab
```

---

## 4. Chi tiết thiết kế các thành phần

### 4.1. Bảng màu & Sidebar (Figma Indigo)
- Sidebar `#FFFFFF`, nền chính `#F4F6FB`, màu nhấn `linear-gradient(135deg, #5B6EF5 0%, #6A5AE0 100%)`, pill active `#EBF0FF`, hộp trích dẫn `#EEF3FF`.
- Sidebar gồm: Logo `ELC3008` -> nút pill `＋ Đoạn chat mới` -> Menu 3 trang (`Trợ lý Chat`, `Kho tài liệu`, `Đánh giá RAG`) -> Danh sách `Cuộc trò chuyện` + `Xóa tất cả` -> Thẻ trạng thái hệ thống ở đáy.

### 4.2. Trang 1 — Trợ lý Chat (`views/chat.py`)
- Empty state: Thẻ chào + lưới 2x2 thẻ câu hỏi mẫu 2 dòng, tự ẩn khi bắt đầu hội thoại.
- Active chat: Streaming chữ qua `answer_chain.stream(...)` trong Card trắng bo góc 18px; lưu cố định mảng `sources` vào `st.session_state.conversations` để không bao giờ mất nguồn trích dẫn cũ; hiển thị hộp trích dẫn `#EEF3FF` (`-> Tên tài liệu · Trang X-Y` + điểm `Cohere` + expander xem trích đoạn gốc); nút pill `↻ Tạo lại câu trả lời` dưới câu cuối.

### 4.3. Trang 2 — Kho tài liệu (`views/documents.py`)
- Hiển thị đầy đủ **18 tài liệu thực tế** có trong `data/faiss_index_huggingface/index.pkl` (13 cuốn có link Google Drive + 5 giáo trình tiếng Việt: `gt_tri_tue_nhan_tao`, `ttnt_nguyen_quang_hoan`, `ttnt_tu_minh_phuong`, `it3190_tai_lieu_doc`, `ml_co_ban`), chia thành **5 nhóm học phần**, đếm động số chunks & trang từ `db.docstore._dict`, kèm nút mở Drive (nếu có link) và nút hỏi thử ngay tài liệu đó.

### 4.4. Trang 3 — Đánh giá RAG (`views/evaluation.py`)
- **Tab A (Quality Gate & RAGAS Benchmark — khớp `eval_rag.py`):**
  - Đủ bộ chỉ số từ `eval_rag.py`: `recall_at_k` (`>= 0.85`), `mrr` (`>= 0.70`), `precision_at_k`, `faithfulness` (`>= 0.85`), `answer_relevancy` (`>= 0.80`), `hallucination_rate` (`< 0.05`), `context_precision`, `context_recall`.
  - Hỗ trợ đọc trực tiếp 1 bản `eval_report.json` (+ `eval_runs.jsonl`) từ Colab hoặc so sánh 2 cấu hình (*Tắt Rerank* vs *Cohere Rerank*) từ `data/eval_results.json`, kèm uploader `.json`/`.jsonl`/`.csv` trên giao diện.
  - Khởi tạo mẫu có cờ `"is_sample": true` hiển thị cảnh báo rõ ràng cho tới khi người dùng nạp kết quả thật từ Colab.
  - Biểu đồ cột đầu tròn màu pastel bằng HTML/CSS thuần giống ảnh Figma + Bảng lọc câu hỏi (`passed = False` khi `recall_at_k == 0` hoặc `faithfulness < 0.85` hoặc `answer_relevancy < 0.70`).
- **Tab B (Live Pipeline Inspector):**
  - Gọi `inspect_retrieval(bundle, question)` đo thời gian từng tầng và đối chiếu Top-5 sau Cohere Rerank với 20 ứng viên Hybrid ban đầu kèm huy hiệu tăng/giảm thứ hạng (`Δ Rank`).
