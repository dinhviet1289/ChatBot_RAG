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


def _bool_secret(key, default):
    """Secret bool, chấp nhận true/1/yes/on; dùng cho EMBED_NORMALIZE."""
    v = _secret(key)
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


BASE_DIR = Path(__file__).resolve().parent
EMBED_MODEL = _secret("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# Embedder đa ngôn ngữ (bge-m3/e5/gte) dùng cosine: index build với
# normalize_embeddings=True nên query cũng phải normalize. MiniLM index cũ build L2
# không chuẩn hóa -> giữ default False, hành vi deploy hiện tại không đổi.
_CROSS_LINGUAL_HINTS = ("bge-m3", "bge_large", "multilingual-e5", "e5-", "gte-")


def resolve_embed_defaults(embed_model):
    """Suy ra (normalize_embeddings, default_index_dir) theo tên model (không phân biệt hoa thường)."""
    low = str(embed_model or "").lower()
    normalize = any(h in low for h in _CROSS_LINGUAL_HINTS)
    index_dir = "faiss_index_bge_m3" if "bge-m3" in low else "faiss_index_huggingface"
    return normalize, index_dir


_default_norm, _DEFAULT_INDEX_DIR = resolve_embed_defaults(EMBED_MODEL)
EMBED_NORMALIZE = _bool_secret("EMBED_NORMALIZE", _default_norm)
# Thư mục index mặc định tự khớp model (bge-m3 -> 1024d, khác -> MiniLM 384d);
# override qua Secret INDEX_PATH khi cần.
INDEX_PATH = _secret("INDEX_PATH", str(BASE_DIR / "data" / _DEFAULT_INDEX_DIR))
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

# Rerank: "off" (nhẹ RAM nhất) | "cohere" (gọi API, không tốn RAM cloud) |
# "local" (CrossEncoder bge-reranker-v2-m3, chỉ cho Colab/máy >2GB RAM).
RERANK_PROVIDER = str(_secret("RERANK_PROVIDER", "off")).strip().lower()
_default_rerank_model = ("rerank-multilingual-v3.0" if RERANK_PROVIDER == "cohere"
                         else "BAAI/bge-reranker-v2-m3")
RERANK_MODEL = _secret("RERANK_MODEL", _default_rerank_model)
COHERE_API_KEY = _secret("COHERE_API_KEY")

_groq_key = _secret("GROQ_API_KEY")


def parse_keys(primary, extra):
    """Gộp key chính với GROQ_API_KEYS (TOML array hoặc chuỗi ngăn phẩy), bỏ trùng.

    Key chỉ tới từ Secrets/env, không bao giờ nằm trong repo. Trùng thì bỏ vì hai
    key giống nhau dùng chung một pool hạn mức, thử lại chỉ tốn thời gian.
    """
    candidates = [primary]
    if isinstance(extra, str):
        candidates += extra.split(",")
    elif extra:
        candidates += list(extra)
    keys = []
    for raw in candidates:
        key = (raw or "").strip()
        if key and key not in keys:
            keys.append(key)
    return keys


GROQ_KEYS = parse_keys(_groq_key, _secret("GROQ_API_KEYS", []))

TEMPLATE = r"""
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
9. Công thức toán phải viết bằng LaTeX chuẩn để hiển thị rõ nét bằng KaTeX trên Streamlit:
   - Công thức quan trọng hoặc đứng riêng dòng: đặt trong cặp $$ trên dòng riêng, ví dụ:
     $$
     \pi_\theta(s, a) = \frac{{\exp(\beta \hat{{Q}}_\theta(s, a))}}{{\sum_{{a'}} \exp(\beta \hat{{Q}}_\theta(s, a'))}}
     $$
   - Biến hoặc biểu thức ngắn trong câu: đặt trong cặp $...$ KHÔNG có khoảng trắng sát dấu $ (vd: $\alpha$, $f(n) = g(n) + h(n)$, $O(b^d)$).
   - KHÔNG dùng \(...\), \[...\], KHÔNG đặt công thức trong code block, và KHÔNG viết công thức phức tạp kiểu plain text rời rạc (như π_θ / Σ_a').
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

QUIZ_TEMPLATE = r"""
Bạn là giảng viên môn Trí tuệ nhân tạo (ELC3008).

NHIỆM VỤ:
Tạo đúng {num_questions} câu hỏi trắc nghiệm khách quan (4 lựa chọn A, B, C, D, chỉ 1 đáp án đúng)
dựa HOÀN TOÀN vào các đoạn NGỮ CẢNH dưới đây.

CHỦ ĐỀ TRỌNG TÂM:
{topic_hint}

QUY TẮC:
1. Mỗi câu hỏi phải kiểm tra kiến thức thực sự trong NGỮ CẢNH (không tự bịa kiến thức ngoài tài liệu).
2. Trình bày câu hỏi và 4 phương án A, B, C, D rõ ràng bằng tiếng Việt (giữ thuật ngữ chuyên ngành tiếng Anh nếu cần).
3. 4 phương án nhiễu phải hợp lý, độ dài tương đương, chỉ có đúng 1 đáp án đúng.
4. Phần "explanation" giải thích ngắn gọn vì sao đáp án đó đúng.
5. "source_idx" là số nguyên (1, 2, ...) khớp với nhãn [Nguồn X | ...] chứa thông tin của câu hỏi.
6. CHỈ trả về JSON hợp lệ dạng mảng (không kèm markdown fence hay chữ thừa ngoài JSON):
[
  {{
    "question": "Nội dung câu hỏi?",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "answer": "A",
    "explanation": "Giải thích ngắn gọn dựa trên tài liệu.",
    "source_idx": 1
  }}
]

NGỮ CẢNH:
{context}
"""


def describe_llm_error(exc):
    """Thay traceback khi gọi mô hình trả lời thất bại (hết hạn mức, mạng, lỗi API).

    Không chép lại thông điệp gốc: body của Groq có thể chứa URL hoặc fragment
    nhạy cảm, còn người dùng cần biết cách xử lý chứ không cần stack trace.
    """
    if "RateLimit" in type(exc).__name__:
        return ("Hạn mức gọi mô hình miễn phí đã chạm trần (~8.000 token/phút hoặc "
                "~200.000 token/ngày). App đã tự thử mọi model free và mọi key trong "
                "Secrets mà vẫn hết. Đợi khoảng một phút rồi bấm **Tạo lại câu trả lời**; "
                "nếu vẫn lỗi thì trần theo ngày đã cạn sạch — thêm key của tài khoản Groq "
                "khác vào `GROQ_API_KEYS` trong Secrets (mỗi model, mỗi tài khoản có hạn "
                "mức riêng).")
    if "NotFound" in type(exc).__name__:
        return ("Groq không cho tài khoản free dùng model này (404). "
                "Đổi `LLM_MODEL` trong Secrets sang một model còn hạn mức free: "
                "`qwen/qwen3.8-27b` (mặc định), `openai/gpt-oss-20b` hoặc `openai/gpt-oss-120b`.")
    return f"Không gọi được mô hình trả lời ({type(exc).__name__}). Thử lại sau ít phút."


def model_chain(configured):
    """Thứ tự model nên thử: model đã cấu hình trước, rồi các model free còn lại."""
    return [configured] + [m for m in GROQ_FREE_MODELS if m != configured]


def rotation_pairs(keys, models):
    """Các cặp (model, key) nên thử: hết mọi model của key này mới sang key kế.

    Mỗi tài khoản Groq có pool riêng theo model, nên đổi model trước (rẻ nhất),
    chỉ đổi key khi cả 3 model của key đó đều cạn.
    """
    pairs = []
    for key in parse_keys(None, keys):
        pairs.extend((model, key) for model in models)
    return pairs


def _model_unusable(exc):
    name = type(exc).__name__
    return "RateLimit" in name or "NotFound" in name


def stream_answer(make_chain, pairs, inputs, report=None):
    """Stream câu trả lời, tự chuyển model rồi key khi cặp trước cạn hạn mức/404.

    Chỉ retry trước khi chữ đầu tiên kịp lên màn hình; đã phun chữ rồi mà lỗi thì
    ném thẳng, vì gọi lại sẽ nhân đôi câu trả lời người dùng đang đọc dở.
    Request bị từ chối vì hạn mức không tốn token, nên quét nhiều cặp vẫn rẻ.
    """
    if not pairs:
        raise ValueError("Thiếu GROQ_API_KEY trong Secrets.")
    last_error = None
    for index, (model, key) in enumerate(pairs):
        started = False
        try:
            for chunk in make_chain(model, key).stream(inputs):
                if not started:
                    started = True
                    if report:
                        report(model, key, index > 0)
                yield chunk
            return
        except Exception as exc:
            if started or not _model_unusable(exc):
                raise
            last_error = exc
    if last_error is not None:
        raise last_error


def invoke_text(make_chain, pairs, inputs, report=None):
    """Gọi chain đồng bộ (non-stream) với fallback qua nhiều model/key khi cạn hạn mức."""
    if not pairs:
        raise ValueError("Thiếu GROQ_API_KEY trong Secrets.")
    last_error = None
    for index, (model, key) in enumerate(pairs):
        try:
            out = make_chain(model, key).invoke(inputs)
            if report:
                report(model, key, index > 0)
            return out
        except Exception as exc:
            if not _model_unusable(exc):
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


_BM25_QUERY_STOPWORDS = frozenset({
    "mô", "hình", "gồm", "thành", "phần", "ví", "dụ",
    "hãy", "nêu", "biết", "giải", "thích", "sánh", "so", "biệt",
    "trình", "bày", "hoạt", "động", "ra", "sao", "dưới", "dạng",
    "khái", "niệm", "định", "nghĩa", "vai", "trò", "chi", "tiết",
    "thuật", "toán", "bài",
})


def select_query_terms(query, doc_freq, total_docs, max_df_ratio=BM25_MAX_DF_RATIO):
    """Chỉ giữ các từ khoá đặc hiệu khi truy vấn BM25 (không dùng khi index chunk).

    Trên kho học liệu song ngữ Việt–Anh (44 tài liệu), các từ đệm đặt câu tiếng Việt
    ("gồm", "thành", "phần", "ví", "dụ"...) chỉ xuất hiện ở nửa tiếng Việt nên tỷ lệ
    df toàn cục < 10%, nhưng cộng 4–5 âm tiết lại vẫn đè bẹp từ khoá hiếm quyết định
    ("peas" chỉ có 14/15.510 chunk tiếng Anh). Ta lọc từ đệm câu hỏi trước, sau đó nếu
    câu chứa từ neo thật hiếm (df <= 0.5% kho) thì loại các từ đi kèm phổ biến gấp >8 lần.
    """
    tokens = tokenize_bm25(query)
    limit = max_df_ratio * total_docs
    specific = [term for term in tokens if doc_freq.get(term, 0) <= limit]
    pool = [term for term in specific if term not in _BM25_QUERY_STOPWORDS] or specific or tokens
    present = [doc_freq[term] for term in pool if doc_freq.get(term, 0) > 0]
    if present and total_docs > 0:
        min_df = min(present)
        if min_df <= max(2, int(0.005 * total_docs)):
            cap = max(min_df * 8, int(0.015 * total_docs))
            anchored = [term for term in pool if doc_freq.get(term, 0) <= cap]
            if anchored:
                return anchored
    return pool


def _dedup(docs):
    uniq = {}
    for d in docs:
        m = d.metadata
        key = (m.get("doc_id"), m.get("start_index"), m.get("page_start"), m.get("page_end"))
        uniq.setdefault(key, d)
    return list(uniq.values())


class CohereReranker:
    """Rerank qua Cohere API -> không tốn RAM cloud. predict() cùng chữ ký với CrossEncoder.
    API lỗi (mạng/rate-limit/key) -> trả về None để tầng truy hồi giữ thứ tự gốc trung thực."""
    def __init__(self, api_key, model):
        import cohere
        self.client = cohere.ClientV2(api_key=api_key, timeout=15)
        self.model = model

    def predict(self, pairs):
        if not pairs:
            return []
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
            return None


def get_rag_bundle():
    """Nạp toàn bộ pipeline RAG đúng một lần (dùng qua @st.cache_resource ở app.py).

    Trả về {db, ensemble, reranker, make_answer_chain, groq_keys, doc_summary, config};
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
    from ui.catalog import summarize_docstore

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": EMBED_NORMALIZE},
    )
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    index_dim = int(getattr(db.index, "d", 0) or 0)
    embed_dim_fn = getattr(getattr(embeddings, "_client", None), "get_sentence_embedding_dimension", None)
    model_dim = int(embed_dim_fn() or 0) if callable(embed_dim_fn) else 0
    if index_dim and model_dim and index_dim != model_dim:
        raise ValueError(
            f"Lệch chiều vector: model {EMBED_MODEL} ({model_dim}d) != index {INDEX_PATH} ({index_dim}d)."
        )

    # Dựng lại BM25 từ docstore đã lưu để web tự chạy, không cần chunk lại
    chunks = list(db.docstore._dict.values())
    total_chunks = len(chunks)
    bm25 = BM25Retriever.from_documents(chunks, preprocess_func=tokenize_bm25)
    # from_documents đã điền tần suất từng chunk vào vectorizer.doc_freqs -> dùng lại
    # thay vì tokenize lại toàn bộ kho lần thứ hai; khi truy vấn đổi sang bộ lọc từ đặc hiệu.
    doc_freq = build_doc_freq(bm25.vectorizer.doc_freqs)
    bm25.preprocess_func = lambda question: select_query_terms(question, doc_freq, total_chunks)
    bm25.k = RETRIEVE_N
    vector_ret = db.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVE_N})
    ensemble = EnsembleRetriever(retrievers=[bm25, vector_ret], weights=[0.3, 0.7])

    reranker = None
    if RERANK_PROVIDER == "cohere":
        if not COHERE_API_KEY:
            raise ValueError("RERANK_PROVIDER=cohere nhưng thiếu COHERE_API_KEY trong Secrets.")
        reranker = CohereReranker(COHERE_API_KEY, RERANK_MODEL)
    elif RERANK_PROVIDER == "local":
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANK_MODEL, max_length=1024)

    prompt_template = ChatPromptTemplate.from_template(TEMPLATE)
    quiz_prompt_template = ChatPromptTemplate.from_template(QUIZ_TEMPLATE)

    def make_answer_chain(model, key):
        return prompt_template | ChatGroq(model=model, api_key=key, temperature=0) | StrOutputParser()

    def make_quiz_chain(model, key):
        return quiz_prompt_template | ChatGroq(model=model, api_key=key, temperature=0.3) | StrOutputParser()

    doc_summary = summarize_docstore(db.docstore._dict)
    config = {
        "top_k": TOP_K,
        "retrieve_n": RETRIEVE_N,
        "rerank_provider": RERANK_PROVIDER,
        "rerank_model": RERANK_MODEL if RERANK_PROVIDER in ("cohere", "local") else None,
        "embed_model": EMBED_MODEL,
        "embed_normalize": EMBED_NORMALIZE,
        "embed_dim": index_dim,
        "llm_model": LLM_MODEL,
        "groq_key_count": len(GROQ_KEYS),
        "index_path": str(INDEX_PATH),
    }
    return {
        "db": db,
        "ensemble": ensemble,
        "reranker": reranker,
        "make_answer_chain": make_answer_chain,
        "make_quiz_chain": make_quiz_chain,
        "groq_keys": list(GROQ_KEYS),
        "doc_summary": doc_summary,
        "config": config,
    }


def inspect_retrieval(bundle, question):
    """Mổ xẻ 2 tầng truy hồi cho trang Kiểm tra truy hồi (views/inspector.py).

    Hàm thuần: chỉ dùng bundle + ui.catalog + time, không Streamlit/LLM,
    không mutate Document hay metadata gốc trong docstore. Mỗi candidate/selected
    gồm {doc_id, doc_title, trang, url, snippet, initial_rank, final_rank,
    rank_delta, rerank_score}; rank_delta = initial_rank - final_rank (dương =
    được rerank đẩy lên). selected = top_k sau rerank (hoặc đầu danh sách khi tắt/lỗi).
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
            "doc_title": known.get("title") or meta.get("doc_title") or doc_id,
            "trang": trang_label(meta),
            "url": DRIVE_LINKS.get(doc_id),
            "snippet": getattr(doc, "page_content", "") or "",
            "initial_rank": rank,
            "final_rank": rank,
            "rank_delta": 0,
            "rerank_score": None,
        })

    rerank_ms = 0.0
    rerank_fallback = False
    order = list(range(len(candidates)))
    if rerank_enabled and candidates:
        t1 = time.perf_counter()
        try:
            raw_scores = reranker.predict([(question, d.page_content) for d in candidates])
        except Exception as e:
            logger.warning("Rerank lỗi, giữ thứ tự truy hồi gốc: %s", e)
            raw_scores = None
        rerank_ms = (time.perf_counter() - t1) * 1000.0
        if raw_scores is None or len(raw_scores) != len(candidates):
            rerank_enabled = False
            rerank_fallback = True
        else:
            scores = [float(s) for s in raw_scores]
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
        "rerank_fallback": rerank_fallback,
        "rerank_provider": provider,
        "total_candidates": len(candidates),
        "candidates": items,
        "selected": selected,
    }