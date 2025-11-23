# app.py — Multi-PDF RAG chat with login + signup + per-user history
import os
import json
import html
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from ingest import ingest_pdf
from retriever import retrieve, generate_answer, DEFAULT_INDEX_PATH

# -------------------------
# Config & path setup
# -------------------------
load_dotenv()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

INDEX_META_PATH = DEFAULT_INDEX_PATH + "_meta.json"
USERS_FILE = os.path.join(DATA_DIR, "users.json")


def get_secret(name: str, default: str | None = None):
    """Read from env vars, then Streamlit secrets, else default."""
    v = os.getenv(name)
    if v is not None:
        return v
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return default


# -------------------------
# User storage (signup / login)
# -------------------------
def load_users_from_disk() -> dict:
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def save_users_to_disk(users: dict) -> None:
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Error saving users:", e)


def init_users() -> dict:
    """
    Initialise users:
      1. If users.json has users, use those.
      2. Else, seed from APP_USERS_JSON or APP_USERNAME/APP_PASSWORD.
    """
    users = load_users_from_disk()
    if users:
        return users

    # Try APP_USERS_JSON first (JSON like {"uday": "pass", "alice":"123"})
    users_json = get_secret("APP_USERS_JSON", "")
    if users_json:
        try:
            users = json.loads(users_json)
        except Exception:
            users = {}

    if not users:
        # Fallback: single user from APP_USERNAME/APP_PASSWORD
        default_user = get_secret("APP_USERNAME", "uday")
        default_pwd = get_secret("APP_PASSWORD", "login123")
        users = {default_user: default_pwd}

    save_users_to_disk(users)
    return users


USERS = init_users()


# -------------------------
# Per-user history helpers
# -------------------------
def get_history_path(user: str) -> str:
    safe = (user or "default").replace("/", "_").replace("\\", "_")
    return os.path.join(DATA_DIR, f"history_{safe}.json")


def load_history_from_disk(user: str):
    path = get_history_path(user)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            return []
    return []


def save_history_to_disk(user: str, history):
    path = get_history_path(user)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print("Error saving history:", e)
        return False


# -------------------------
# Index helpers
# -------------------------
def get_indexed_sources():
    """Return sorted unique list of PDF sources from index meta file."""
    if not os.path.exists(INDEX_META_PATH):
        return []
    try:
        with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
            metas = json.load(f)
        sources = sorted({m.get("source", "unknown") for m in metas})
        return sources
    except Exception:
        return []


# -------------------------
# Auth (login + signup)
# -------------------------
def check_auth():
    """Username/password login with simple signup."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    # Already logged in
    if st.session_state.logged_in and st.session_state.current_user:
        return

    st.title("🔐UDAY'S PDF Q&A — Login / Sign up")

    tab_login, tab_signup = st.tabs(["Login", "Sign up"])

    # ---- LOGIN TAB ----
    with tab_login:
        st.subheader("Sign in")
        login_user = st.text_input("Username", key="login_username")
        login_pwd = st.text_input("Password", type="password", key="login_password")

        def do_login():
            global USERS
            if login_user in USERS and USERS[login_user] == login_pwd:
                st.session_state.logged_in = True
                st.session_state.current_user = login_user
                st.session_state.history = load_history_from_disk(login_user)
            else:
                st.error("Incorrect username or password.")

        if st.button("Login"):
            do_login()

    # ---- SIGNUP TAB ----
    with tab_signup:
        st.subheader("Create a new account")
        new_user = st.text_input("Choose a username", key="signup_username")
        new_pwd = st.text_input("Choose a password", type="password", key="signup_password")
        new_pwd2 = st.text_input("Confirm password", type="password", key="signup_confirm")

        def do_signup():
            global USERS
            if not new_user.strip():
                st.error("Username cannot be empty.")
                return
            if new_user in USERS:
                st.error("That username is already taken.")
                return
            if not new_pwd:
                st.error("Password cannot be empty.")
                return
            if new_pwd != new_pwd2:
                st.error("Passwords do not match.")
                return

            USERS[new_user] = new_pwd
            save_users_to_disk(USERS)
            st.success("Account created! You can now log in with your new credentials.")

        if st.button("Sign up"):
            do_signup()

    st.info("UDAY:ITS A DEMO DONT USE REAL PASSWORD")
    st.stop()


# -------------------------
# Streamlit page config & CSS
# -------------------------
st.set_page_config(page_title="PDF Q&A — RAG Chat (Multi-PDF)", layout="wide")

chat_css = """
<style>
.chat-container {max-width: 900px; margin: 0 auto;}
.msg-row {display: flex; margin: 0.4rem 0;}
.msg {
  padding: 0.6rem 0.9rem;
  border-radius: 1rem;
  max-width: 60%;
  font-size: 0.9rem;
}
.msg.user {
  margin-left: auto;
  background: linear-gradient(135deg, #3b82f6, #1d4ed8);
  color: white;
  border-bottom-right-radius: 0.2rem;
}
.msg.assistant {
  margin-right: auto;
  background: #111827;
  color: #e5e7eb;
  border-bottom-left-radius: 0.2rem;
}
.meta {
  font-size: 0.65rem;
  opacity: 0.7;
  margin-bottom: 0.25rem;
}
.small { font-size: 0.8rem; }
</style>
"""
st.markdown(chat_css, unsafe_allow_html=True)

# -------------------------
# Main app
# -------------------------
check_auth()
current_user = st.session_state.current_user or "default"

# initialise per-user history
if "history" not in st.session_state:
    st.session_state.history = load_history_from_disk(current_user)

# cached index sources
if "indexed_sources" not in st.session_state:
    st.session_state.indexed_sources = get_indexed_sources()

# -------------------------
# Sidebar: ingestion, filters, settings, history, logout
# -------------------------
with st.sidebar:
    st.header("Multi-PDF Ingestion")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    saved_paths = []
    if uploaded_files:
        for up in uploaded_files:
            save_path = os.path.join(DATA_DIR, up.name)
            with open(save_path, "wb") as f:
                f.write(up.getbuffer())
            saved_paths.append(save_path)
        st.success(f"Saved {len(saved_paths)} PDF(s) to data/")

    if st.button("Ingest all uploaded PDFs"):
        if not saved_paths and not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            to_ingest = saved_paths or [
                os.path.join(DATA_DIR, up.name) for up in uploaded_files
            ]
            with st.spinner("Ingesting PDFs..."):
                for p in to_ingest:
                    try:
                        ingest_pdf(p)
                    except Exception as e:
                        st.error(f"Failed to ingest {os.path.basename(p)}: {e}")
            st.session_state.indexed_sources = get_indexed_sources()
            st.success("Finished ingestion.")

    st.markdown("---")
    st.subheader("Filter PDFs for search")

    selected_sources = []
    if st.session_state.indexed_sources:
        for src in st.session_state.indexed_sources:
            key = f"src_{src}"
            checked = st.checkbox(src, value=True, key=key)
            if checked:
                selected_sources.append(src)
    else:
        st.caption("No PDFs ingested yet.")

    st.markdown("---")
    st.subheader("Retrieval settings")
    k = st.slider("Number of retrieved chunks (k)", 1, 8, 4)

    st.markdown("---")
    st.subheader("History controls")
    st.caption(f"Logged in as **{current_user}**")

    if st.button("Clear history (local)"):
        st.session_state.history = []
        st.success("Cleared in-memory history.")

    if st.button("Save history to disk"):
        ok = save_history_to_disk(current_user, st.session_state.history)
        st.success("Saved." if ok else "Save failed.")

    if st.button("Reload history from disk"):
        st.session_state.history = load_history_from_disk(current_user)
        st.rerun()

    if st.session_state.history:
        lines = []
        for item in st.session_state.history:
            ts = item.get("timestamp", "")
            if item["role"] == "user":
                lines.append(f"[USER {ts}] {item.get('query','')}")
            else:
                lines.append(f"[ASSISTANT {ts}] {item.get('answer','')}")
        txt = "\n".join(lines)
        st.download_button(
            "Download chat (TXT)",
            txt,
            file_name=f"chat_{current_user}.txt",
        )

    # ------------- LOGOUT BUTTON -------------
    st.markdown("---")
    if st.button("🚪 Logout"):
        # Clear session state safely
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# -------------------------
# Main chat area
# -------------------------
st.title("📄 PDF Q&A — RAG Chat (Multi-PDF)")
st.write(
    "Upload multiple PDFs, ingest them, choose which ones to search, "
    "and chat with an assistant grounded in your documents."
)

st.markdown("---")

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
last_assistant_entry = None

for entry in st.session_state.history:
    role = entry.get("role")
    ts = entry.get("timestamp", "")
    tsstr = ts if ts else ""

    if role == "user":
        q = html.escape(entry.get("query", ""))
        user_html = (
            f"<div class='msg-row'>"
            f"<div class='msg user'>"
            f"<div class='meta'>You • {tsstr}</div>"
            f"<div class='small'>{q}</div>"
            f"</div></div>"
        )
        st.markdown(user_html, unsafe_allow_html=True)
    elif role == "assistant":
        a = entry.get("answer", "") or ""
        a = a.replace("\n", "<br>")
        asst_html = (
            f"<div class='msg-row'>"
            f"<div class='msg assistant'>"
            f"<div class='meta'>Assistant • {tsstr}</div>"
            f"<div class='small'>{a}</div>"
            f"</div></div>"
        )
        st.markdown(asst_html, unsafe_allow_html=True)
        last_assistant_entry = entry

st.markdown("</div>", unsafe_allow_html=True)

# Show sources/snippets for last answer
if last_assistant_entry and last_assistant_entry.get("retrieved"):
    with st.expander("Sources used"):
        used = []
        for r in last_assistant_entry["retrieved"]:
            used.append(f"{r.get('source')} (page {r.get('page')})")
        used = sorted(set(used))
        for u in used:
            st.markdown(f"- {u}")

    with st.expander("Retrieved snippets (show/hide)"):
        for r in last_assistant_entry["retrieved"]:
            st.markdown(
                f"**{r.get('source')} — page {r.get('page')} (chunk {r.get('chunk_idx')})**"
            )
                # snippet
            st.write(r.get("text", "")[:800] + "...")

st.markdown("---")

# -------------------------
# Query input with safe clear flag
# -------------------------
st.subheader("Enter your question here")

if "clear_query" not in st.session_state:
    st.session_state.clear_query = False

# Clear input BEFORE creating widget if flag set
if st.session_state.clear_query:
    st.session_state.query_input = ""
    st.session_state.clear_query = False

query = st.text_input(" ", key="query_input", label_visibility="collapsed")

col_send, col_clear = st.columns([1, 1])
with col_send:
    send_clicked = st.button("Send")
with col_clear:
    clear_clicked = st.button("Clear input")

if clear_clicked:
    st.session_state.clear_query = True
    st.rerun()

# -------------------------
# Send logic
# -------------------------
if send_clicked:
    if not query.strip():
        st.warning("Please enter a question before sending.")
    else:
        if not st.session_state.indexed_sources:
            st.error("No PDFs indexed. Please ingest at least one PDF first.")
        elif not selected_sources:
            st.error("Select at least one PDF in the sidebar to search.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                try:
                    retrieved = retrieve(
                        query,
                        k=k,
                        index_path=DEFAULT_INDEX_PATH,
                        allowed_sources=selected_sources,
                    )
                    if not retrieved:
                        st.warning(
                            "No relevant chunks found. "
                            "Try increasing k or selecting more PDFs."
                        )
                    else:
                        answer = generate_answer(query, retrieved)
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        st.session_state.history.append(
                            {
                                "role": "user",
                                "query": query,
                                "timestamp": ts,
                                "selected_sources": selected_sources,
                            }
                        )
                        st.session_state.history.append(
                            {
                                "role": "assistant",
                                "answer": answer,
                                "timestamp": ts,
                                "retrieved": retrieved,
                            }
                        )

                        save_history_to_disk(current_user, st.session_state.history)
                        st.session_state.clear_query = True
                        st.rerun()
                except FileNotFoundError:
                    st.error("No index found. Please ingest at least one PDF first.")
                except Exception as e:
                    st.error(f"Error retrieving/generating: {e}")
