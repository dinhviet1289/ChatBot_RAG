import copy
import logging
import os
import re
import time
import unicodedata
from pathlib import Path

# Chỉ import stdlib ở cấp module: mọi thư viện bên thứ ba (streamlit, faiss,
# langchain_*, cohere, sentence_transformers) được import lazy bên trong hàm
# để `import rag_engine_web` chạy được cả trong môi trường unit-test tối giản.

logger = logging.getLogger(__name__)


def _secret(key, default=None):
    """Đọc từ Streamlit Secrets (cloud/local), fallback về biến môi trường.

    Import streamlit lazy trong try/except: chưa cài Streamlit, chưa khởi tạo
    runtime, hoặc thiếu key -> không raise, trả về env/default.
    """
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = _secret("INDEX_PATH", str(BASE_DIR / "data" / "faiss_index_huggingface"))
# Key Groq của bản deploy đã đốt trần pool ngày trên gpt-oss-* (TPD trôi theo 24h),
# qwen là model free còn chỗ; đổi lại qua LLM_MODEL trong Secrets khi cần.
LLM_MODEL = _secret("LLM_MODEL", "qwen/qwen3.8-27b")
# Ba model Groq còn mở cho tài khoản free (đo 2026-11-16); mỗi model một pool
# hạn mức riêng nên model này cạn thì model khác vẫn còn chỗ.
GROQ_FREE_MODELS = ("qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b")
RETRIEVE_N = int(_secret("RETRIEVE_N", 20))
TOP_K = int(_secret("RERANK_TOP_K", 5))
# BM25 bỏ từ khoá xuất hiện trong hơn tỷ lệ này số chunk (xem select_query_terms).
BM25_MAX_DF_RATIO = float(_secret("BM25_MAX_DF_RATIO", 0.10))
EMBED_MODEL = _secret("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Rerank: "off" (nhẹ RAM nhất) | "cohere" (gọi API, không tốn RAM cloud) |
# "local" (CrossEncoder bge-reranker-v2-m3, chỉ cho Colab/máy >2GB RAM).
RERANK_PROVIDER = str(_secret("RERANK_PROVIDER", "off")).strip().lower()
_default_rerank_model = ("rerank-multilingual-v3.0" if RERANK_PROVIDER == "cohere"
                         else "BAAI/bge-reranker-v2-m3")
RERANK_MODEL = _secret("RERANK_MODEL", _default_rerank_model)
COHERE_API_KEY = _secret("COHERE_API_KEY")

_groq_key = _secret("GROQ_API_KEY")
if _groq_key:
    os.environ["GROQ_API_KEY"] = _groq_key

TEMPLATE = """
Bạn là trợ lý học tập môn Trí tuệ nhân tạo.

NHIỆM VỤ:
Giải đáp câu hỏi của sinh viên dựa trên tài liệu được cung cấp.

QUY TẮC:
1. Chỉ sử dụng thông tin trong NGỮ CẢNH.
2. Không tự ý bổ sung kiến thức bên ngoài, không đoán.
3. Nếu NGỮ CẢNH không có thông tin để trả lời, chỉ nói ngắn gọn (bằng ngôn ngữ của câu hỏi)
   rằng tài liệu môn học không đề cập nội dung này.
4. Nếu NGỮ CẢNH chỉ có một phần thông tin, hãy trả lời phần có và nói rõ phần nào NGỮ CẢNH chưa đủ để trả lời.
   KHÔNG kết luận tài liệu môn học thiếu nội dung đó; với câu hỏi tóm tắt nhiều chương, gợi ý hỏi từng chương để xem phần còn lại.
5. Giải thích dễ hiểu, phù hợp với sinh viên.
6. Có thể sử dụng ví dụ nếu ví dụ đó có trong tài liệu.
7. Trả lời bằng ngôn ngữ của câu hỏi.
8. Sau mỗi nội dung có sử dụng thông tin tài liệu, ghi nguồn theo định dạng (Tên tài liệu, trang X-Y),
   chép đúng theo nhãn trong ngoặc vuông ở đầu mỗi đoạn NGỮ CẢNH.
   Dòng "<!-- page: N -->" (nếu có) cho biết nội dung ngay sau nó thuộc trang N.
9. Công thức toán phải viết bằng plain text dễ đọc, KHÔNG dùng LaTeX hay cặp $...$ / $$...$$.
   Ví dụ: sigmoid(x) = 1 / (1 + exp(-x)); softmax(x_i) = exp(x_i) / tong_cac_j(exp(x_j)); P(A|B) = P(B|A) * P(A) / P(B).
   Dùng ký tự thường, dấu ngoặc và phép toán rõ ràng để sinh viên đọc hiểu ngay.
10. Phạm vi môn là Trí tuệ nhân tạo. Nếu câu hỏi hỏi kiến thức chung ngoài môn (địa lý, thủ đô, lịch sử...),
    KHÔNG trả lời dù NGỮ CẢNH tình cờ chứa câu chữ giống đáp án (vd: "Paris là thủ đô nước Pháp" chỉ là ví dụ
    logic mệnh đề, không phải nội dung môn). Khi đó từ chối ngắn gọn rằng tài liệu môn học
    không đề cập nội dung này (câu hỏi ngoài phạm vi môn).

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
"""


def page_label(meta):
    start, end = meta.get("page_start"), meta.get("page_end")
    if start is None:
        return "trang ?"
    return f"trang {start}" if start == end else f"trang {start}-{end}"


def format_docs(docs):
    parts = []
    for d in docs:
        m = d.metadata
        name = m.get("doc_title") or m.get("doc_id", "Unknown")
        parts.append(f"[{name} ({m.get('doc_id', 'Unknown')}), {page_label(m)}]\n{d.page_content}")
    return "\n\n".join(parts)


def describe_llm_error(exc):
    """Thay traceback khi gọi mô hình trả lời thất bại (hết hạn mức, mạng, lỗi API).

    Không chép lại thông điệp gốc: body của Groq có thể chứa URL hoặc fragment
    nhạy cảm, còn người dùng cần biết cách xử lý chứ không cần stack trace.
    """
    if "RateLimit" in type(exc).__name__:
        return ("Hạn mức gọi mô hình miễn phí đã chạm trần (~8.000 token/phút hoặc "
                "~200.000 token/ngày). App đã tự thử các model free còn lại mà vẫn hết. "
                "Đợi khoảng một phút rồi bấm **Tạo lại câu trả lời**; nếu vẫn lỗi thì trần "
                "theo ngày đã cạn sạch ở mọi model — đổi `GROQ_API_KEY` hoặc `LLM_MODEL` "
                "trong Secrets (mỗi model, mỗi tài khoản có hạn mức riêng).")
    if "NotFound" in type(exc).__name__:
        return ("Groq không cho tài khoản free dùng model này (404). "
                "Đổi `LLM_MODEL` trong Secrets sang một model còn hạn mức free: "
                "`qwen/qwen3.8-27b` (mặc định), `openai/gpt-oss-20b` hoặc `openai/gpt-oss-120b`.")
    return f"Không gọi được mô hình trả lời ({type(exc).__name__}). Thử lại sau ít phút."


def model_chain(configured):
    """Thứ tự model nên thử: model đã cấu hình trước, rồi các model free còn lại."""
    return [configured] + [m for m in GROQ_FREE_MODELS if m != configured]


def _model_unusable(exc):
    name = type(exc).__name__
    return "RateLimit" in name or "NotFound" in name


def stream_answer(make_chain, models, inputs, report=None):
    """Stream câu trả lời, tự chuyển model khi model trước cạn hạn mức hoặc bị 404.

    Chỉ retry trước khi chữ đầu tiên kịp lên màn hình; đã phun chữ rồi mà lỗi thì
    ném thẳng, vì gọi lại sẽ nhân đôi câu trả lời người dùng đang đọc dở.
    """
    last_error = None
    for index, model in enumerate(models):
        started = False
        try:
            for chunk in make_chain(model).stream(inputs):
                if not started:
                    started = True
                    if report:
                        report(model, index > 0)
                yield chunk
            return
        except Exception as exc:
            if started or not _model_unusable(exc):
                raise
            last_error = exc
    if last_error is not None:
        raise last_error


_BM25_STOPWORDS = frozenset({
    "là", "gì", "và", "của", "có", "các", "những", "nào", "cho", "về",
    "trong", "một", "được", "với", "như", "thế",
})


def tokenize_bm25(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"([a-z])([A-Z]{2,})", r"\1 \2", text)
    return [word for word in re.findall(r"[^\W_]+", text.casefold())
            if word not in _BM25_STOPWORDS]


def build_doc_freq(tokens_per_doc):
    """Đếm số chunk chứa mỗi từ (một từ lặp trong cùng chunk chỉ tính một lần)."""
    df = {}
    for tokens in tokens_per_doc:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    return df


def select_query_terms(query, doc_freq, total_docs, max_df_ratio=BM25_MAX_DF_RATIO):
    """Chỉ giữ các từ khoá đặc hiệu khi truy vấn BM25 (không dùng khi index chunk).

    Điểm BM25 là TỔNG idf của các từ khớp, nên vài âm tiết Việt thông dụng
    ("mô", "hình", "thành", "phần", "ví", "dụ") cộng lại vượt một từ hiếm nhưng
    quyết định ("peas"): chunk chứa PEAS rớt từ hạng 1 xuống 426 và không vào được
    danh sách ứng viên, tầng rerank mất cơ hội sửa. Từ xuất hiện trong hơn
    max_df_ratio số chunk bị bỏ; nếu câu hỏi chỉ toàn từ thông dụng thì giữ nguyên.
    """
    tokens = tokenize_bm25(query)
    limit = max_df_ratio * total_docs
    specific = [term for term in tokens if doc_freq.get(term, 0) <= limit]
    return specific or tokens


def _dedup(docs):
    uniq = {}
    for d in docs:
        m = d.metadata
        key = (m.get("doc_id"), m.get("start_index"), m.get("page_start"), m.get("page_end"))
        uniq.setdefault(key, d)
    return list(uniq.values())


class RerankRetriever:
    """Tầng 1: hybrid lấy N ứng viên. Tầng 2: cross-encoder giữ top-k."""
    def __init__(self, base, reranker, k):
        self.base = base
        self.reranker = reranker
        self.k = k

    def invoke(self, question):
        candidates = _dedup(self.base.invoke(question))
        if not candidates:
            return []
        scores = self.reranker.predict([(question, d.page_content) for d in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)[:self.k]
        out = []
        for d, sc in ranked:
            # Copy nông Document + metadata để KHÔNG mutate Document dùng chung
            # trong db.docstore (rerank_score chỉ tồn tại trên bản sao trả về).
            d = copy.copy(d)
            d.metadata = dict(d.metadata)
            d.metadata["rerank_score"] = round(float(sc), 4)
            out.append(d)
        return out


class CohereReranker:
    """Rerank qua Cohere API -> không tốn RAM cloud. predict() cùng chữ ký với CrossEncoder.
    API lỗi (mạng/rate-limit/key) -> giữ thứ tự truy hồi gốc để chatbot vẫn trả lời được."""
    def __init__(self, api_key, model):
        import cohere
        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model

    def predict(self, pairs):
        query = pairs[0][0]
        docs = [text for _, text in pairs]
        try:
            resp = self.client.rerank(model=self.model, query=query,
                                      documents=docs, top_n=len(docs))
            scores = [0.0] * len(docs)
            for r in resp.results:
                scores[r.index] = float(r.relevance_score)
            return scores
        except Exception as e:
            logger.warning("Cohere rerank lỗi, giữ thứ tự truy hồi gốc: %s", e)
            return [float(len(docs) - i) for i in range(len(docs))]


class TopKRetriever:
    """Không reranker: hybrid lấy N ứng viên rồi giữ top-k (nhẹ RAM cho cloud free)."""
    def __init__(self, base, k):
        self.base = base
        self.k = k

    def invoke(self, question):
        return _dedup(self.base.invoke(question))[:self.k]


def get_rag_bundle():
    """Nạp toàn bộ pipeline RAG đúng một lần (dùng qua @st.cache_resource ở app.py).

    Trả về {db, ensemble, reranker, retriever, answer_chain, config};
    reranker = None khi RERANK_PROVIDER = "off". Mọi import bên thứ ba nằm
    trong hàm để module vẫn import được chỉ với stdlib.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # Dựng lại BM25 từ docstore đã lưu để web tự chạy, không cần chunk lại
    chunks = list(db.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(chunks, preprocess_func=tokenize_bm25)
    # from_documents đã dùng preprocess_func để index; từ lúc truy vấn trở đi nó
    # chỉ chạy trên câu hỏi -> thay bằng bộ lọc từ khoá đặc hiệu riêng tầng BM25
    # (tầng vector vẫn nhận nguyên câu hỏi để giữ ngữ nghĩa).
    doc_freq = build_doc_freq(tokenize_bm25(d.page_content) for d in chunks)
    bm25.preprocess_func = lambda question: select_query_terms(question, doc_freq, len(chunks))
    bm25.k = RETRIEVE_N
    vector_ret = db.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVE_N})
    ensemble = EnsembleRetriever(retrievers=[bm25, vector_ret], weights=[0.3, 0.7])

    reranker = None
    if RERANK_PROVIDER == "cohere":
        if not COHERE_API_KEY:
            raise ValueError("RERANK_PROVIDER=cohere nhưng thiếu COHERE_API_KEY trong Secrets.")
        reranker = CohereReranker(COHERE_API_KEY, RERANK_MODEL)
        retriever = RerankRetriever(ensemble, reranker, TOP_K)
    elif RERANK_PROVIDER == "local":
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANK_MODEL, max_length=1024)
        retriever = RerankRetriever(ensemble, reranker, TOP_K)
    else:
        retriever = TopKRetriever(ensemble, TOP_K)

    prompt_template = ChatPromptTemplate.from_template(TEMPLATE)

    def make_answer_chain(model):
        return prompt_template | ChatGroq(model=model, temperature=0) | StrOutputParser()

    answer_chain = make_answer_chain(LLM_MODEL)

    config = {
        "top_k": TOP_K,
        "retrieve_n": RETRIEVE_N,
        "rerank_provider": RERANK_PROVIDER,
        "rerank_model": RERANK_MODEL if RERANK_PROVIDER in ("cohere", "local") else None,
        "embed_model": EMBED_MODEL,
        "llm_model": LLM_MODEL,
        "index_path": str(INDEX_PATH),
    }
    return {"db": db, "ensemble": ensemble, "reranker": reranker,
            "retriever": retriever, "answer_chain": answer_chain,
            "make_answer_chain": make_answer_chain, "config": config}


def get_rag():
    """Backward compat: giữ chữ ký cũ (retriever, answer_chain) cho app hiện tại."""
    bundle = get_rag_bundle()
    return bundle["retriever"], bundle["answer_chain"]


def inspect_retrieval(bundle, question):
    """Mổ xẻ 2 tầng truy hồi cho trang Kiểm tra truy hồi (views/inspector.py).

    Hàm thuần: chỉ dùng bundle + ui.catalog + time, không Streamlit/LLM,
    không mutate Document hay metadata gốc trong docstore. Mỗi candidate/selected
    gồm {doc_id, doc_title, trang, url, snippet, initial_rank, final_rank,
    rank_delta, rerank_score}; rank_delta = initial_rank - final_rank (dương =
    được rerank đẩy lên). selected = top_k sau rerank (hoặc đầu danh sách khi tắt).
    """
    from ui.catalog import CATALOG, DRIVE_LINKS, trang_label

    config = bundle.get("config") or {}
    top_k = int(config.get("top_k", TOP_K))
    provider = str(config.get("rerank_provider", RERANK_PROVIDER)).strip().lower()
    reranker = bundle.get("reranker")
    rerank_enabled = reranker is not None and provider not in ("off", "none", "")

    t0 = time.perf_counter()
    candidates = _dedup(bundle["ensemble"].invoke(question))
    retrieve_ms = (time.perf_counter() - t0) * 1000.0

    items = []
    for rank, doc in enumerate(candidates, 1):
        meta = getattr(doc, "metadata", None) or {}
        doc_id = meta.get("doc_id") or "?"
        known = CATALOG.get(doc_id) or {}
        items.append({
            "doc_id": doc_id,
            "doc_title": meta.get("doc_title") or known.get("title") or doc_id,
            "trang": trang_label(meta),
            "url": DRIVE_LINKS.get(doc_id),
            "snippet": getattr(doc, "page_content", "") or "",
            "initial_rank": rank,
            "final_rank": rank,
            "rank_delta": 0,
            "rerank_score": None,
        })

    rerank_ms = 0.0
    order = list(range(len(candidates)))
    if rerank_enabled and candidates:
        t1 = time.perf_counter()
        scores = [float(s) for s in reranker.predict([(question, d.page_content) for d in candidates])]
        rerank_ms = (time.perf_counter() - t1) * 1000.0
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        for final_rank, idx in enumerate(order, 1):
            item = items[idx]
            item["final_rank"] = final_rank
            item["rank_delta"] = item["initial_rank"] - final_rank
            item["rerank_score"] = round(scores[idx], 4)

    selected = [items[i] for i in order[:top_k]]
    return {
        "question": question,
        "retrieve_ms": round(retrieve_ms, 2),
        "rerank_ms": round(rerank_ms, 2),
        "rerank_enabled": rerank_enabled,
        "total_candidates": len(candidates),
        "candidates": items,
        "selected": selected,
    }