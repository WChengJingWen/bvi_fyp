import streamlit as st
import json, time, os
from PIL import Image
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Robot Guide UI", layout="wide")

STATE_FILE = "/home/mustar/catkin_ws/src/bvi_fyp/src/tmp/ui_state.json"
CHAT_FILE  = "/home/mustar/catkin_ws/src/bvi_fyp/src/tmp/ui_chat.jsonl"

st_autorefresh(interval=300, key="ui_refresh")

# ---------- load latest state ----------
state = {"page": "status", "status": "detecting"}
last_update_age = None

if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        last_update_age = time.time() - state.get("_received_ts", time.time())
    except Exception as e:
        st.warning(f"Failed reading {STATE_FILE}: {e}")

page = state.get("page", "status")
status = state.get("status", "detecting")

# ---------- load chat log ----------
chat = []
if os.path.exists(CHAT_FILE):
    try:
        with open(CHAT_FILE, "r") as f:
            lines = f.readlines()[-80:]  # last 80 lines
        for ln in lines:
            chat.append(json.loads(ln))
    except Exception as e:
        st.warning(f"Failed reading {CHAT_FILE}: {e}")

# ---------- styling ----------
st.markdown(
    """
    <style>
    .big-status { font-size: 42px; font-weight: 800; }
    .sub-status { font-size: 18px; opacity: 0.85; }
    .chat-bubble { padding: 12px 14px; border-radius: 14px; margin: 8px 0; }
    .user {
        background: #e8f0fe;
        color: black;
    }
    .robot {
        background: #f1f3f4;
        color: #222222;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- header ----------
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("🤖 Campus Guide Robot")
with col2:
    st.write("**Page:**", page)
    st.write("**Status:**", status)
with col3:
    if last_update_age is None:
        st.caption("No state file yet")
    else:
        st.caption(f"Last update: {last_update_age:.1f}s ago")

st.divider()

# ---------- pages ----------
def render_status():
    destination = state.get("destination", "")
    status_text_map = {
        "detecting": "Detecting target user…",
        "approaching": "Approaching target user…",
        "navigating": "Navigating to destination…",
        "navigation_completed": "Navigation completed ✅",
        "idle": "Idle…"
    }
    main = status_text_map.get(status, status)
    st.markdown(f"<div class='big-status'>{main}</div>", unsafe_allow_html=True)
    if destination:
        st.markdown(f"<div class='sub-status'>Destination: <b>{destination}</b></div>", unsafe_allow_html=True)

def render_detection():
    st.markdown("<div class='big-status'>Target user detected ✅</div>", unsafe_allow_html=True)
    img_path = state.get("image_path", "/tmp/target_user.jpg")
    st.write("Snapshot:")
    try:
        img = Image.open(img_path)
        st.image(img, use_container_width=True, caption=img_path)
    except Exception as e:
        st.error(f"Cannot load image at {img_path}. Error: {e}")

def render_chat():
    st.markdown("<div class='big-status'>Conversation</div>", unsafe_allow_html=True)
    for m in chat[-50:]:
        role = m.get("role", "robot")
        text = m.get("text", "")
        css = "user" if role == "user" else "robot"
        label = "👤 User" if role == "user" else "🤖 Robot"
        st.markdown(f"<div class='chat-bubble {css}'><b>{label}:</b><br>{text}</div>",
                    unsafe_allow_html=True)

if page == "status":
    render_status()
elif page == "detection":
    render_detection()
elif page == "chat":
    render_chat()
else:
    st.error(f"Unknown page: {page}")
    render_status()
