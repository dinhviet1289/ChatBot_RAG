# ELC3008 — Trợ lý học tập AI (RAG Chatbot)

Chatbot hỏi đáp tài liệu môn Trí tuệ nhân tạo, chạy thẳng trên **Streamlit Community Cloud**
từ repo GitHub này — không cần Colab/ngrok.

- **Web:** `app.py` (Streamlit) + `rag_engine_web.py` (RAG engine).
- **Retrieval:** hybrid BM25 + FAISS (embedding `paraphrase-multilingual-MiniLM-L12-v2`).
- **LLM:** Groq `openai/gpt-oss-120b`.
- **Reranker:** TẮT mặc định trên cloud free (`USE_RERANKER=false`) cho vừa ~1GB RAM.
  Pipeline đầy đủ (có reranker `bge-reranker-v2-m3`) vẫn giữ trong notebook Colab để eval.
- **Eval (RAGAS):** chạy trong Colab, KHÔNG nằm trong repo web này.

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
- Giữ `USE_RERANKER=false` (đừng bật trên free).
- Cân nhắc nâng cấp gói trả phí, hoặc chạy web trên máy/Colab như cũ.