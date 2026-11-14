import logging
import os
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


def _secret(key, default=None):
    """Đọc từ Streamlit Secrets (cloud/local), fallback về biến môi trường."""
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = _secret("INDEX_PATH", str(BASE_DIR / "data" / "faiss_index_huggingface"))
LLM_MODEL = _secret("LLM_MODEL", "openai/gpt-oss-120b")
RETRIEVE_N = int(_secret("RETRIEVE_N", 20))
TOP_K = int(_secret("RERANK_TOP_K", 5))
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


def get_rag():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # Dựng lại BM25 từ docstore đã lưu để web tự chạy, không cần chunk lại
    bm25 = BM25Retriever.from_documents(list(db.docstore._dict.values()),
                                        preprocess_func=lambda t: t.lower().split())
    bm25.k = RETRIEVE_N
    vector_ret = db.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVE_N})
    ensemble = EnsembleRetriever(retrievers=[bm25, vector_ret], weights=[0.3, 0.7])

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

    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    answer_chain = ChatPromptTemplate.from_template(TEMPLATE) | llm | StrOutputParser()
    return retriever, answer_chain