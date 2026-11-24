# ELC3008 — Trợ lý học tập AI (RAG Chatbot)

Trợ lý hỏi đáp học liệu môn **Trí tuệ nhân tạo (ELC3008 — ĐH Kinh tế, ĐH Đà Nẵng)** xây dựng theo kiến trúc **Retrieval-Augmented Generation (RAG)**, triển khai trực tiếp trên **Streamlit Community Cloud**.

- **Web Live:** https://chatbotragdue.streamlit.app/
- **Kho học liệu:** 44 tài liệu PDF (24 tiếng Việt, 20 tiếng Anh) — 15.510 chunks (`size=1500`, `overlap=300`).
- **Đánh giá định lượng (RAGAS):** Chạy riêng trên Google Colab (không đóng gói vào repo web để giữ nhẹ RAM).

---

## Kiến trúc Pipeline RAG

```text
Câu hỏi người dùng
  │
  ▼
[Stage 1] Hybrid Retrieval (N = 20 ứng viên)
  ├── BM25_df (trọng số 0.30) — lọc từ khóa đặc hiệu, bắt chuẩn thuật ngữ/ký hiệu
  └── FAISS   (trọng số 0.70) — tìm kiếm ngữ nghĩa (MiniLM 384d hoặc bge-m3 1024d)
  │
  ▼
[Stage 2] Cross-Encoder Rerank (giữ Top-K = 5)
  └── Cohere API (rerank-multilingual-v3.0) — không tốn RAM cloud; fallback an toàn nếu lỗi
  │
  ▼
[Stage 3] Sinh câu trả lời & Trích dẫn
  └── Groq `qwen/qwen3.8-27b` (tự xoay vòng model → key khi gặp 429/404) + KaTeX
```

### Điểm nổi bật kỹ thuật
- **LLM & Tự động xoay vòng hạn mức:** Mặc định dùng Groq `qwen/qwen3.8-27b` (đổi qua `LLM_MODEL` trong Secrets). Các model free được hỗ trợ gồm `qwen/qwen3.8-27b`, `openai/gpt-oss-20b` và `openai/gpt-oss-120b` (mỗi model có pool hạn mức riêng ~200K token/ngày). Khi gặp lỗi `404`/`429`, `stream_answer()` tự động thử lần lượt các model còn lại trên key hiện tại trước khi chuyển sang key kế tiếp trong `GROQ_API_KEYS`.
- **Reranker linh hoạt:** Hỗ trợ `RERANK_PROVIDER=off | cohere | local`. Trên Streamlit Cloud miễn phí (~1 GB RAM), khuyên dùng `cohere` qua API để tăng độ chính xác truy hồi mà không tốn RAM; `local` (`BAAI/bge-reranker-v2-m3`) chỉ dành cho máy/Colab > 2 GB RAM.
- **Hiển thị công thức toán chuẩn KaTeX:** Prompt kết hợp hàm `normalize_latex_for_streamlit()` chuẩn hóa các ký hiệu LaTeX về chuẩn `$$...$$` / `$...$` của Streamlit mà vẫn giữ nguyên code block và tiền tệ.

---

## Giao diện & Tính năng (`Scientific Observatory`)

`app.py` sử dụng bố cục `layout="wide"` với sidebar trái (thương hiệu **Trợ lý học tập DUE**, nút tạo cuộc hội thoại mới, điều hướng, danh sách hội thoại **mới nhất ở trên cùng**, thẻ trạng thái hệ thống và chỉ số **RAM app** thực tế):

1. **Màn hình hội thoại (`views/conversation.py` — trang mặc định):**
   - Cuộc hội thoại mới mở với hero tổng quan kho học liệu và **4 thẻ gợi ý 2×2**.
   - Trả lời streaming kèm dải telemetry từng lượt (tổng thời gian, `Stage-1 ms`, `Rerank ms`, số chunk chọn, model và slot key đang phục vụ).
   - Khung **`Nguồn trích dẫn & truy hồi`** nằm gọn trong `st.expander`, hiển thị huy hiệu hạng `#1..#5`, biến động hạng sau rerank (`#init → #final ▲ +delta`), điểm liên quan, link Google Drive (nếu có) và trích đoạn gốc.
2. **Kho tài liệu (`views/documents.py`):**
   - Thống kê KPI (44 học liệu, 15.510 chunks, số chiều embedding thực tế), thanh phân bổ theo 8 nhóm học phần, lọc theo nhóm, tìm kiếm và nút **Hỏi thử tài liệu này**.
3. **Kiểm tra truy hồi (`views/inspector.py`):**
   - Công cụ debug 2 tầng Stage-1 / Stage-2 (kèm bảng toàn bộ ứng viên Stage-1) cho câu hỏi bất kỳ mà **không gọi LLM** nên không tiêu tốn token hạn mức.

---

## Cấu trúc thư mục

```text
chatbox-rag/
├── app.py                        # Entrypoint Streamlit & điều hướng sidebar
├── rag_engine_web.py             # Hybrid BM25 + FAISS, Cohere rerank, Groq fallback rotation
├── requirements.txt              # Thư viện tối giản (torch CPU-only, không kèm ragas/plotly)
├── ui/
│   ├── catalog.py                # Danh mục 44 học liệu, link Drive, gộp nguồn trích dẫn
│   ├── state.py                  # Quản lý state đa hội thoại (pure Python)
│   └── theme.py                  # Theme Dark Indigo, CSS & renderer HTML/LaTeX an toàn
├── views/
│   ├── conversation.py           # Màn hình hội thoại chính
│   ├── documents.py              # Trang Kho tài liệu
│   └── inspector.py              # Trang Kiểm tra truy hồi
├── data/
│   ├── faiss_index_huggingface/  # Index MiniLM-L12-v2 (384d): index.faiss, index.pkl, manifest.json
│   └── faiss_index_bge_m3/       # Index BAAI/bge-m3 (1024d): index.faiss, index.pkl, manifest.json (cho Colab/máy >2.2GB RAM)
├── .streamlit/
│   ├── config.toml               # Cấu hình theme & server Streamlit
│   └── secrets.toml.example      # Mẫu biến môi trường / Secrets (không chứa key thật)
└── tests/                        # Bộ unit test thuần stdlib (không cần cài Streamlit)
```

---

## Chạy thử Local

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Điền GROQ_API_KEY (và COHERE_API_KEY nếu bật rerank) vào .streamlit/secrets.toml
streamlit run app.py
```

### Chạy Unit Tests

Toàn bộ logic pure-Python (`ui/`, `rag_engine_web.py` và cấu trúc `app.py`) được kiểm thử không phụ thuộc runtime Streamlit:

```bash
python -m unittest discover -s tests -v
```

---

## Triển khai lên Streamlit Community Cloud

### 1. Chuẩn bị FAISS index
Đảm bảo thư mục `data/faiss_index_huggingface/` có đủ **3 file** (`index.faiss`, `index.pkl`, `manifest.json`). Bộ đầy đủ 44 tài liệu / 15.510 chunks (~45 MB) nhỏ hơn giới hạn 100 MB của GitHub nên có thể commit trực tiếp (nếu vượt 100 MB cần dùng Git LFS).

### 2. Tạo app & cấu hình Secrets
1. Truy cập https://share.streamlit.io → đăng nhập GitHub → **Create app**.
2. Chọn repository, branch `main`, main file `app.py` (khuyến nghị **Python 3.12** trong *Advanced settings*).
3. Mở **Settings → Secrets** và dán cấu hình:

```toml
GROQ_API_KEY = "gsk_your_primary_key"
GROQ_API_KEYS = ["gsk_your_primary_key", "gsk_backup_key_1", "gsk_backup_key_2"]

# Bật Rerank qua Cohere API (khuyên dùng trên cloud free)
RERANK_PROVIDER = "cohere"
COHERE_API_KEY = "your_cohere_trial_key"
```

> **Lưu ý vận hành:**
> - Sau khi sửa Secrets, nhấn **Save changes** rồi **Manage app → Reboot** để tiến trình app nạp lại cấu hình.
> - Khi push code thay đổi trong `ui/`, `views/` hoặc `rag_engine_web.py`, Streamlit Cloud chỉ tự đọc lại `app.py`; hãy **Reboot** app để áp dụng đầy đủ các module mới.
> - Nhiều API key tạo trong *cùng một* tài khoản Groq dùng chung pool hạn mức; để tăng hạn mức thực tế, cần dùng key từ các tài khoản khác nhau.
> - Tuyệt đối không commit `.streamlit/secrets.toml` hoặc dán key thật vào mã nguồn (file này đã nằm trong `.gitignore`).

---

## Tùy chọn nâng cấp Embedding (`BAAI/bge-m3`, 1024d) & Giới hạn RAM

| Model Embedding | Số chiều | RAM ước tính | Ghi chú |
|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` *(mặc định)* | 384d | ~470 MB | An toàn cho Streamlit Cloud free (~1 GB RAM) |
| `BAAI/bge-m3` | 1024d | > 2.2 GB | Nâng **Hit@5 từ 0.434 → 0.697 (+61%)** trên kho song ngữ Việt–Anh; cần gói RAM > 2.2 GB |

Để chuyển sang `bge-m3`:
1. Kiểm tra `data/faiss_index_bge_m3/` đã có đủ `index.faiss`, `index.pkl` và `manifest.json` (đã kèm sẵn trong repo; có thể gỡ khỏi branch deploy nếu muốn thu gọn clone khi chỉ chạy Free Tier).
2. Thêm `EMBED_MODEL = "BAAI/bge-m3"` vào Secrets (code tự động trỏ sang `data/faiss_index_bge_m3` và bật `normalize_embeddings=True`).
3. **Manage app → Reboot** và kiểm tra KPI *Không gian embedding* tại trang **Kho tài liệu** hiển thị `1024d · BAAI/bge-m3`.

---

## Khóa phiên bản thư viện (khi build Cloud gặp xung đột)

`requirements.txt` đang giữ phiên bản `langchain-*` linh hoạt. Nếu quá trình build trên Cloud gặp lỗi xung đột phụ thuộc, chạy đoạn sau trong môi trường Colab đang chạy ổn định để lấy version cố định:

```python
import importlib.metadata as md
for p in ["torch", "streamlit", "langchain-classic", "langchain-community",
          "langchain-core", "langchain-groq", "langchain-huggingface",
          "faiss-cpu", "rank-bm25", "sentence-transformers", "cohere"]:
    try:
        print(f"{p}=={md.version(p)}")
    except Exception:
        print(f"# {p}: not installed")
```
