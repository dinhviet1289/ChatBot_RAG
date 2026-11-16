import streamlit as st
from rag_engine_web import get_rag, format_docs
from ui.theme import inject_global_css

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


st.set_page_config(page_title="ELC3008 | Trợ lý học tập AI",
                   page_icon=":material/school:", layout="centered")

inject_global_css()

SAMPLE = ["LLM là gì? Kiến trúc gồm những thành phần nào?",
          "Mô hình PEAS gồm những thành phần nào? Cho ví dụ.",
          "So sánh thuật toán DFS và BFS."]

with st.sidebar:
    st.subheader("ELC3008")
    st.markdown("**Trí tuệ nhân tạo trong kinh doanh**")
    st.caption("Khoa Thương mại Điện tử · Trường ĐH Kinh tế (DUE)")
    st.divider()
    st.markdown("**Câu hỏi gợi ý**")
    for q in SAMPLE:
        if st.button(q, width="stretch"):
            st.session_state.pending = q
    st.divider()
    if st.button("Xóa lịch sử đoạn chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

st.title("Trợ lý học tập ELC3008")
st.caption("Hỏi đáp khái niệm, thuật toán và trích dẫn trực tiếp từ tài liệu đã nạp.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
        "content": "Xin chào! Bạn muốn ôn tập khái niệm hay phần bài giảng nào?"}]

@st.cache_resource
def load_rag():
    return get_rag()

try:
    retriever, answer_chain = load_rag()
except Exception as e:
    st.error(f"Chưa nạp được vector index: {e}")
    st.info("Kiểm tra thư mục data/faiss_index_huggingface đã có index.faiss + index.pkl chưa, "
            "và GROQ_API_KEY đã khai báo trong Secrets chưa.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Nhập câu hỏi của bạn...")
if st.session_state.get("pending"):
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu tài liệu..."):
            try:
                # retrieve MỘT lần rồi tự ráp context: chain.invoke(question) sẽ truy vấn lần thứ hai
                docs = retriever.invoke(prompt)
                answer = answer_chain.invoke({"context": format_docs(docs), "question": prompt})
                st.markdown(answer)
                if "không đề cập" not in answer:
                    with st.expander(f"Nguồn trích dẫn ({len(docs)})"):
                        for i, d in enumerate(docs, 1):
                            m = d.metadata
                            s, e = m.get("page_start", 0), m.get("page_end", 0)
                            trang = str(s) if s == e else f"{s}-{e}"
                            doc_id = m.get("doc_id", "?")
                            title = m.get("doc_title") or doc_id
                            url = DRIVE_LINKS.get(doc_id)
                            label = f"[{title} ({doc_id})]({url})" if url else f"{title} ({doc_id})"
                            st.markdown(f"**{i}. {label}** · Trang {trang}")
                            st.caption(d.page_content.strip()[:200] + "...")
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as ex:
                st.error(f"Đã có lỗi khi xử lý: {ex}")