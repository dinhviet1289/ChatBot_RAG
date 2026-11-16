import streamlit as st

from rag_engine_web import get_rag_bundle
from ui.state import (
    clear_all_conversations,
    create_new_conversation,
    delete_conversation,
    init_state,
)
from ui.theme import inject_global_css
from views.documents import render_documents_view
from views.inspector import render_inspector_view

st.set_page_config(page_title="ELC3008 | Trợ lý học tập AI",
                   page_icon=":material/school:", layout="wide")
inject_global_css()
init_state(st.session_state)


@st.cache_resource
def load_bundle():
    return get_rag_bundle()


try:
    bundle = load_bundle()
except Exception as e:
    st.error(f"Chưa nạp được vector index: {e}")
    st.info("Kiểm tra data/faiss_index_huggingface (index.faiss + index.pkl) "
            "và GROQ_API_KEY trong Secrets.")
    st.stop()

state = st.session_state
NAV = [("documents", ":material/folder_open:", "Kho tài liệu"),
       ("inspector", ":material/forum:", "Hỏi đáp & Truy hồi")]

with st.sidebar:
    st.markdown("### ELC3008")
    st.caption("Trí tuệ nhân tạo trong kinh doanh · DUE")
    if st.button(":material/add: Cuộc hội thoại mới", width="stretch", type="primary"):
        create_new_conversation(state)
        state["active_page"] = "inspector"
        st.rerun()
    for key, icon, label in NAV:
        active = state["active_page"] == key
        if st.button(f"{icon} {label}", key=f"nav_{key}", width="stretch",
                     type="primary" if active else "secondary"):
            state["active_page"] = key
            st.rerun()
    st.divider()
    st.caption("Cuộc hội thoại")
    for conv in state["conversations"]:
        c1, c2 = st.columns([0.85, 0.15])
        with c1:
            if st.button(conv["title"], key=f"conv_{conv['id']}", width="stretch"):
                state["active_conv_id"] = conv["id"]
                state["active_page"] = "inspector"
                st.rerun()
        with c2:
            if st.button(":material/delete:", key=f"del_{conv['id']}"):
                delete_conversation(state, conv["id"])
                st.rerun()
    if st.button("Xóa tất cả", width="stretch"):
        clear_all_conversations(state)
        st.rerun()
    st.divider()
    cfg = bundle["config"]
    with st.container(border=True):
        st.caption("Hệ thống")
        provider = cfg.get("rerank_provider", "off")
        if provider == "off":
            st.markdown(":material/close: Rerank **TẮT**")
        else:
            st.markdown(f":material/check_circle: Rerank **{provider}** · {cfg.get('rerank_model', '')}")
        st.caption(f"Top-k {cfg.get('top_k')} · retrieve {cfg.get('retrieve_n')}")
        st.caption(f"LLM {cfg.get('llm_model')} · {len(cfg.get('groq_keys') or [])} key")
        st.caption(f"Embed {cfg.get('embed_model')}")

page = state["active_page"]
if page == "inspector":
    render_inspector_view(bundle)
else:
    render_documents_view(bundle)