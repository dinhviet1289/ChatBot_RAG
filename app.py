import streamlit as st

from rag_engine_web import get_rag_bundle
from ui.state import (
    clear_all_conversations,
    create_new_conversation,
    delete_conversation,
    init_state,
)
from ui.theme import inject_global_css, render_system_card_html
from views.conversation import render_conversation_view
from views.documents import render_documents_view
from views.inspector import render_inspector_view


def _ram_mb():
    """RAM thực của process app (MB) đọc từ /proc/self/status trên container Linux."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        return None


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
       ("inspector", ":material/science:", "Kiểm tra truy hồi")]

# Sidebar chạy TRƯỚC dispatch ở cuối file: handler chỉ cần đổi state, lần chạy
# hiện tại sẽ tự render trang mới (không gọi rerun thủ công để tránh nháy đôi).
with st.sidebar:
    st.markdown(
        '<div class="rag-brand">'
        '<span class="rag-brand-kicker">ELC3008 · AI LAB</span>'
        '<div class="rag-brand-title">Trợ lý học tập DUE</div>'
        '<div class="rag-brand-sub">Trí tuệ nhân tạo trong kinh doanh · ĐH Kinh tế</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button(":material/add: Cuộc hội thoại mới", width="stretch", type="primary"):
        create_new_conversation(state)
        state["active_page"] = "conversation"
    for key, icon, label in NAV:
        active = state["active_page"] == key
        if st.button(f"{icon} {label}", key=f"nav_{key}", width="stretch",
                     type="primary" if active else "secondary"):
            state["active_page"] = key
    st.divider()
    st.caption("Cuộc hội thoại")
    for conv in state["conversations"]:
        c1, c2 = st.columns([0.85, 0.15])
        with c1:
            opened = (conv["id"] == state["active_conv_id"]
                      and state["active_page"] == "conversation")
            if st.button(conv["title"], key=f"conv_{conv['id']}", width="stretch",
                         type="primary" if opened else "secondary"):
                state["active_conv_id"] = conv["id"]
                state["active_page"] = "conversation"
        with c2:
            if st.button(":material/delete:", key=f"del_{conv['id']}"):
                delete_conversation(state, conv["id"])
    if st.button("Xóa tất cả", width="stretch"):
        clear_all_conversations(state)
    st.divider()
    st.caption("Hệ thống")
    st.markdown(render_system_card_html(bundle["config"]), unsafe_allow_html=True)
    ram = _ram_mb()
    st.caption(f"RAM app: {ram:.0f} MB" if ram is not None else "RAM app: —")

page = state["active_page"]
if page == "conversation":
    render_conversation_view(bundle)
elif page == "inspector":
    render_inspector_view(bundle)
else:
    render_documents_view(bundle)
