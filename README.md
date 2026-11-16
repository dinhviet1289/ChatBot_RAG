# ELC3008 — Trợ lý học tập AI (RAG Chatbot)

Chatbot hỏi đáp tài liệu môn Trí tuệ nhân tạo, chạy thẳng trên **Streamlit Community Cloud**
từ repo GitHub này — không cần Colab/ngrok.

- **Web:** `app.py` (Streamlit) + `rag_engine_web.py` (RAG engine).
- **Retrieval:** hybrid BM25 + FAISS (embedding `paraphrase-multilingual-MiniLM-L12-v2`).
- **LLM:** Groq `openai/gpt-oss-20b` (đổi bằng `LLM_MODEL` trong Secrets). Key free gọi được
  `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b` — mỗi model một pool hạn mức
  riêng. Hai model Llama cũ của Groq đã chuyển sang Enterprise → **404 trên key free**.
- **Reranker:** mặc định TẮT (`RERANK_PROVIDER=off`). Trên cloud free nên bật
  `RERANK_PROVIDER=cohere` — rerank qua API, **không tốn RAM**. Bản local `bge-reranker-v2-m3`
  (`RERANK_PROVIDER=local`) chỉ cho Colab/máy >2GB RAM; pipeline đầy đủ vẫn giữ trong notebook để eval.
- **Eval (RAGAS):** chạy trong Colab, KHÔNG nằm trong repo web này.

## Giao diện 3 trang (Figma Indigo)

`app.py` là shell `layout="wide"` với sidebar trái (brand, nút "Đoạn chat mới",
3 nút điều hướng, danh sách cuộc trò chuyện, thẻ trạng thái hệ thống gồm
**Rerank đang bật/tắt**). Ba view tách trong `views/`:

- `views/chat.py` — Trợ lý Chat: hero + 4 thẻ gợi ý, trả lời streaming,
  trích dẫn bền vững theo từng lượt, nút "Tạo lại câu trả lời".
- `views/documents.py` — Kho tài liệu: KPI (số học liệu / chunks / chiều embedding),
  lọc theo nhóm + tìm kiếm, thẻ từng tài liệu kèm link Drive và "Hỏi thử tài liệu này".
- `views/inspector.py` — Kiểm tra truy hồi: đo ms retrieve/rerank, Top-K kèm rank-delta,
  danh sách toàn bộ ứng viên stage-1 và tuỳ chọn sinh câu trả lời LLM để đối chiếu.

Logic thuần nằm trong `ui/` (`catalog`, `state`, `theme`) để unit-test
không cần Streamlit. Renderer HTML escape mọi text và chỉ nhận link `https://`.

> Đánh giá định lượng (Quality Gate + RAGAS) nằm trong Colab, không ở web app — xem
> `eval_rag.py` trong notebook.

---

## 1. Chuẩn bị FAISS index (bắt buộc)

Repo cần index đã build sẵn. Tải từ Google Drive của m:

```
/content/drive/MyDrive/ChatboxRAG/faiss_index_huggingface/
```

Copy **2 file** `index.faiss` và `index.pkl` vào thư mục:

```
data/faiss_index_huggingface/index.faiss
data/faiss_index_huggingface/index.pkl
```

> Index này nhỏ (vài MB) nên commit thẳng lên GitHub được. Nếu >100MB thì dùng Git LFS.

## 2. Đẩy lên GitHub

```bash
cd chatbox-rag
git init
git add .
git commit -m "ELC3008 RAG chatbot for Streamlit Cloud"
git branch -M main
git remote add origin https://github.com/<ten-m>/<ten-repo>.git
git push -u origin main
```

**Lưu ý bảo mật:** `.streamlit/secrets.toml` đã nằm trong `.gitignore` — KHÔNG commit key.
Chỉ commit `secrets.toml.example` (file mẫu, không chứa key thật).

## 3. Nối Streamlit Community Cloud

1. Vào https://share.streamlit.io → đăng nhập bằng GitHub.
2. **Create app** → chọn repo này, branch `main`, file `app.py`.
3. Mở **Advanced settings → Secrets**, dán:
   ```toml
   GROQ_API_KEY = "gsk_key_that_cua_m"
   ```
4. **Deploy**. Lần đầu build vài phút (tải torch CPU + model embedding).

Xong — app chạy tại `https://<ten-app>.streamlit.app`, tự chạy lại mỗi khi m push code.

### Bật rerank Cohere (khuyên dùng trên cloud free)

Rerank tăng chất lượng truy hồi mà không tốn RAM (model chạy trên server Cohere):

1. Tạo **Trial API key** miễn phí tại https://dashboard.cohere.com/api-keys.
2. App settings → **Secrets**, thêm:
   ```toml
   RERANK_PROVIDER = "cohere"
   COHERE_API_KEY = "key_cua_m"
   ```
3. Save → app tự chạy lại. Mặc định dùng `rerank-multilingual-v3.0` (hỗ trợ tiếng Việt);
   muốn đổi thì thêm `RERANK_MODEL = "rerank-v3.5"` (hoặc `rerank-v4.0-fast`).

Nếu Cohere lỗi/rate-limit, app tự fallback về thứ tự truy hồi gốc nên vẫn trả lời được.

## 4. Chạy thử local (tuỳ chọn)

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # rồi điền key thật
streamlit run app.py
```

---

## Khoá version (nếu build lỗi)

`requirements.txt` đang để version langchain "thả nổi" cho khớp bản mới nhất. Nếu Cloud
build báo lỗi xung đột version, chạy **1 cell này trong Colab** (nơi đã chạy tốt) rồi gửi
output để khoá chính xác:

```python
import importlib.metadata as md
for p in ["torch","streamlit","langchain","langchain-classic","langchain-community",
          "langchain-core","langchain-groq","langchain-huggingface",
          "faiss-cpu","rank-bm25","sentence-transformers"]:
    try: print(f"{p}=={md.version(p)}")
    except Exception: print(f"# {p}: not installed")
```

Dán các dòng `pkg==version` vào `requirements.txt` (giữ nguyên dòng `--extra-index-url`
và đổi `torch==...` thành bản `+cpu` tương ứng nếu có).

## Giới hạn RAM (đọc kỹ)

Streamlit Cloud free ~1GB RAM. Bản này đã bỏ reranker để nhẹ nhất, nhưng model embedding
vẫn ~470MB. Nếu app bị **OOM / restart liên tục**:
- Đừng đặt `RERANK_PROVIDER=local` trên free (model reranker ~2GB). Dùng `cohere` (API) hoặc `off`.
- Cân nhắc nâng cấp gói trả phí, hoặc chạy web trên máy/Colab như cũ.