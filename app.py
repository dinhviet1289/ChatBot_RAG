import logging

import streamlit as st

from rag_engine_web import INDEX_PATH, get_rag_bundle
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

logger = logging.getLogger(__name__)


def _ram_mb():
    """RAM thực của process app (MB) đọc từ /proc/self/status trên container Linux."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        return None


def _on_new_conv():
    create_new_conversation(st.session_state)
    st.session_state["active_page"] = "conversation"


def _on_nav(page_key):
    st.session_state["active_page"] = page_key


def _on_select_conv(conv_id):
    st.session_state["active_conv_id"] = conv_id
    st.session_state["active_page"] = "conversation"


def _on_delete_conv(conv_id):
    delete_conversation(st.session_state, conv_id)


def _on_clear_all():
    clear_all_conversations(st.session_state)


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
    logger.exception("Không nạp được bundle RAG")
    st.error(f"Chưa nạp được vector index ({type(e).__name__}).")
    st.info(f"Kiểm tra `{INDEX_PATH}` (index.faiss + index.pkl) và cấu hình Secrets.")
    st.stop()

state = st.session_state
NAV = [("documents", ":material/folder_open:", "Kho tài liệu"),
       ("inspector", ":material/science:", "Kiểm tra truy hồi")]

# Sidebar dùng on_click callback (chạy TRƯỚC khi render nút ở rerun mới) nên trạng thái
# active cập nhật tức thì ở lần click đầu tiên mà không cần gọi rerun thủ công.
with st.sidebar:
    st.markdown(
        '<div class="rag-brand">'
        '<span class="rag-brand-kicker">ELC3008 · AI LAB</span>'
        '<div class="rag-brand-title">Trợ lý học tập DUE</div>'
        '<div class="rag-brand-sub">Trí tuệ nhân tạo trong kinh doanh · ĐH Kinh tế</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.button(":material/add: Cuộc hội thoại mới", key="new_conv",
              width="stretch", type="primary", on_click=_on_new_conv)
    for key, icon, label in NAV:
        active = state["active_page"] == key
        st.button(f"{icon} {label}", key=f"nav_{key}", width="stretch",
                  type="primary" if active else "secondary",
                  on_click=_on_nav, args=(key,))
    st.divider()
    st.caption("Gần đây")
    for conv in state["conversations"]:
        c1, c2 = st.columns([0.84, 0.16], vertical_alignment="center")
        with c1:
            opened = (conv["id"] == state["active_conv_id"]
                      and state["active_page"] == "conversation")
            st.button(conv["title"], key=f"conv_{conv['id']}", width="stretch",
                      type="primary" if opened else "secondary",
                      on_click=_on_select_conv, args=(conv["id"],))
        with c2:
            st.button(":material/delete:", key=f"del_{conv['id']}",
                      help="Xóa cuộc hội thoại",
                      on_click=_on_delete_conv, args=(conv["id"],))
    st.button(":material/delete_sweep: Xóa tất cả", key="clear_all",
              width="stretch", on_click=_on_clear_all)
    st.divider()
    st.caption("Hệ thống")
    st.markdown(render_system_card_html(bundle["config"]), unsafe_allow_html=True)
    ram = _ram_mb()
    ram_txt = f"{ram:.0f} MB" if ram is not None else "—"
    st.markdown(
        f'<div class="rag-ram-caption"><span>RAM app:</span><span>{ram_txt}</span></div>',
        unsafe_allow_html=True,
    )

page = state["active_page"]
if page == "conversation":
    render_conversation_view(bundle)
elif page == "inspector":
    render_inspector_view(bundle)
else:
    render_documents_view(bundle)
