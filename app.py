# app.py — RAG PDF Chat with multi-PDF ingestion + PDF filters + login
import streamlit as st
import os
import json
import html
from datetime import datetime
from dotenv import load_dotenv

from ingest import ingest_multiple_pdfs
from retriever import retrieve, generate_answer

# -------------------
# Paths & basic config
# -------------------
DATA_DIR = "data"
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
INDEX_META_PATH = os.path.join(DATA_DIR, "faiss_index_meta.json")
os.makedirs(DATA_DIR, exist_ok=True)

# load .env for local dev
load_dotenv()

def get_secret(name: str, default: str | None = None):
    """Read from env vars, then from Streamlit secrets, else default."""
    # 1) normal environment variables / .env
    val = os.getenv(name)
    if val is not None:
        return val
    # 2) Streamlit Cloud secrets
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    # 3) fallback
    return default


DATA_DIR = "data"
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
INDEX_META_PATH = os.path.join(DATA_DIR, "faiss_index_meta.json")
os.makedirs(DATA_DIR, exist_ok=True)

APP_USERNAME = get_secret("APP_USERNAME", "uday")
APP_PASSWORD = get_secret("APP_PASSWORD", "secret123")


st.set_page_config(page_title="PDF Q&A — RAG Chat", layout="wide")

# -------------------
# Chat UI CSS
# -------------------
chat_css = """
<style>
.chat-wrapper {
    max-width: 900px;
    margin: 0 auto;
    padding: 0.5rem 0 3rem 0;
}

/* Row that holds a single message */
.msg-row {
    display: flex;
    width: 100%;
    margin: 8px 0;
}

/* Assistant row: align to LEFT inside wrapper */
.msg-row.assistant {
    justify-content: flex-start;
}

/* User row: align to RIGHT inside wrapper */
.msg-row.user {
    justify-content: flex-end;
}

/* Message bubble */
.msg {
    padding: 10px 14px;
    border-radius: 18px;
    max-width: 75%;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    font-size: 14px;
    white-space: pre-wrap;
    word-break: break-word;
    text-align: left;
    line-height: 1.4;
}

/* Assistant bubble style (left) */
.msg.assistant {
    background: #f5f5f7;
    color: #111;
    border-top-left-radius: 6px;
    margin-right: auto;
}

/* User bubble style (right) */
.msg.user {
    background: linear-gradient(135deg, #4f9cff, #4175ff);
    color: #ffffff;
    border-top-right-radius: 6px;
    margin-left: auto;
}

/* Metadata (who + time) */
.meta {
    font-size: 11px;
    color: #888;
    margin-bottom: 4px;
}

/* Slight spacing on expanders */
.stExpander {
    margin-top: 0.25rem;
    margin-bottom: 0.25rem;
}
</style>
"""
st.markdown(chat_css, unsafe_allow_html=True)

# -------------------
# Helpers
# -------------------
def check_login():
    """Simple username/password login using session_state."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return

    st.title("🔐 UDAY'S PDF Q&A Login")

    def try_login():
        user = st.session_state.get("login_username", "")
        pwd = st.session_state.get("login_password", "")
        if user == APP_USERNAME and pwd == APP_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.login_username = ""
            st.session_state.login_password = ""
           
        else:
            st.error("Incorrect username or password.")

    st.text_input("Username", key="login_username")
    st.text_input("Password", type="password", key="login_password")
    st.button("Login", on_click=try_login)

    # stop rest of app until logged in
    st.stop()


def load_history_from_disk():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            return []
    return []


def save_history_to_disk(history):
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print("Error saving history:", e)
        return False


def get_index_sources(meta_path=INDEX_META_PATH):
    """Return list of unique PDF sources currently in the index."""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metas = json.load(f)
        sources = sorted({m.get("source", "unknown") for m in metas})
        return sources
    except Exception:
        return []


# -------------------
# Login gate
# -------------------
check_login()

# -------------------
# Session state init
# -------------------
if "history" not in st.session_state:
    st.session_state.history = load_history_from_disk()

if "last_ingested" not in st.session_state:
    st.session_state.last_ingested = []

if "k" not in st.session_state:
    st.session_state.k = 4

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

if "selected_sources" not in st.session_state:
    st.session_state.selected_sources = []


# -------------------
# Sidebar
# -------------------
with st.sidebar:
    st.header("Multi-PDF Ingestion")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    saved_paths = []
    if uploaded_files:
        for uploaded in uploaded_files:
            save_path = os.path.join(DATA_DIR, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            saved_paths.append(save_path)
        st.success(f"Saved {len(saved_paths)} PDF(s) to {DATA_DIR}/")

        if st.button("Ingest all uploaded PDFs"):
            with st.spinner("Ingesting PDFs..."):
                try:
                    count_pdfs, count_chunks = ingest_multiple_pdfs(saved_paths)
                    st.success(f"Ingested {count_pdfs} PDF(s) with {count_chunks} chunks.")
                    st.session_state.last_ingested = [os.path.basename(p) for p in saved_paths]
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    # --- PDF filter checkboxes ---
    st.markdown("---")
    st.header("Filter PDFs for search")

    available_sources = get_index_sources()
    if available_sources:
        selected = []
        for src in available_sources:
            key = f"src_{src}"
            default_value = (
                (not st.session_state.selected_sources)
                or (src in st.session_state.selected_sources)
            )
            checked = st.checkbox(src, value=default_value, key=key)
            if checked:
                selected.append(src)

        # Save exactly what the user selected (can be empty)
        st.session_state.selected_sources = selected

        if not selected:
            st.caption("No PDFs selected – questions will not run until you tick at least one.")
    else:
        st.write("No PDFs indexed yet.")

    st.markdown("---")
    st.header("Retrieval settings")
    st.session_state.k = st.slider(
        "Number of retrieved chunks (k)",
        min_value=1,
        max_value=12,
        value=st.session_state.k,
    )
    st.caption("Model & API configured via .env / GROQ_API_KEY, etc.")

    st.markdown("---")
    st.header("History controls")
    if st.session_state.history:
        st.write(f"Messages: {len(st.session_state.history)}")
        if st.button("Clear history (and disk)"):
            st.session_state.history = []
            save_history_to_disk(st.session_state.history)
            st.success("History cleared.")
    else:
        st.write("No messages yet.")

    if st.session_state.history:
        txt_export = "\n\n".join(
            (
                f"[{datetime.fromtimestamp(h.get('ts', 0)).strftime('%Y-%m-%d %H:%M:%S')}] "
                + (
                    "YOU: " + h.get("query", "")
                    if h.get("role") == "user"
                    else "ASSISTANT: " + (h.get("answer") or "")
                )
            )
            for h in st.session_state.history
        )
        st.download_button("Download chat (TXT)", data=txt_export, file_name="conversation.txt")
        st.download_button(
            "Download chat (JSON)",
            data=json.dumps(st.session_state.history, ensure_ascii=False, indent=2),
            file_name="conversation.json",
        )

    if st.button("Save history to disk"):
        ok = save_history_to_disk(st.session_state.history)
        st.success("Saved." if ok else "Save failed.")

    if st.button("Reload history from disk"):
        st.session_state.history = load_history_from_disk()
        st.rerun()

    st.markdown("---")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.caption("Keep API keys out of the repo. Use environment variables / hosting secrets.")


# -------------------
# Main layout
# -------------------
col_main, col_right = st.columns([3, 1])

with col_main:
    st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)
    st.title("📄 PDF Q&A — RAG Chat (Multi-PDF)")
    st.write("Upload multiple PDFs, ingest them, filter which ones to search, and ask questions across them.")

    # ---- Render chat history ----
    for entry in st.session_state.history:
        role = entry.get("role")
        ts = entry.get("ts")
        ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""

        if role == "user":
            text_raw = entry.get("query", "") or ""
            text_safe = html.escape(text_raw).replace("\n", "<br>")
            msg_row_class = "msg-row user"
            msg_class = "msg user"
            who = "You"
        else:
            text_raw = entry.get("answer", "") or ""
            text_safe = html.escape(str(text_raw)).replace("\n", "<br>")
            msg_row_class = "msg-row assistant"
            msg_class = "msg assistant"
            who = "Assistant"

        html_block = f"""
        <div class="{msg_row_class}">
            <div>
                <div class="meta">{who} • {ts_str}</div>
                <div class="{msg_class}">
                    {text_safe}
                </div>
            </div>
        </div>
        """
        st.markdown(html_block, unsafe_allow_html=True)

        # For assistant messages, show sources + retrieved chunks under expanders
        if role == "assistant":
            sources = entry.get("sources") or []
            if sources:
                with st.expander("Sources used"):
                    for s in sources:
                        st.write(f"- {s}")
            retrieved = entry.get("retrieved") or []
            if retrieved:
                with st.expander("Retrieved snippets (show/hide)"):
                    for r in retrieved:
                        st.write(
                            f"**{r.get('source')}** — page {r.get('page')} (chunk {r.get('chunk_idx')})"
                        )
                        txt = r.get("text", "") or ""
                        preview = txt[:800] + ("..." if len(txt) > 800 else "")
                        st.write(preview)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---- Input handling ----
    if st.session_state.clear_input:
        st.session_state.query_input = ""
        st.session_state.clear_input = False

    query = st.text_input("Enter your question here", key="query_input")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    send_clicked = c1.button("Send")
    clear_input_clicked = c2.button("Clear input")
    save_hist_clicked = c3.button("Save history")
    clear_hist_clicked = c4.button("Clear history (local only)")

    # --------------------------
    # Send button behavior
    # --------------------------
    if send_clicked and query and query.strip():
        user_q = query.strip()

        # --- Safety check: ensure at least 1 PDF ingested ---
        available_sources_now = get_index_sources()
        if not available_sources_now:
            st.error("❗ No PDFs indexed yet. Please upload and ingest at least one PDF.")
            st.stop()

        # --- Safety check: ensure at least 1 PDF is selected in checkboxes ---
        allowed = st.session_state.get("selected_sources") or []
        if not allowed:
            st.error("❗ No PDFs selected. Please tick at least one PDF under 'Filter PDFs for search' in the sidebar.")
            st.stop()

        # Append user message to history
        st.session_state.history.append({
            "role": "user",
            "query": user_q,
            "answer": None,
            "sources": None,
            "retrieved": None,
            "ts": datetime.now().timestamp(),
        })

        with st.spinner("Retrieving relevant chunks and generating answer..."):
            try:
                retrieved = retrieve(user_q, k=st.session_state.k, allowed_sources=allowed)

                # If nothing retrieved from selected PDFs
                if not retrieved:
                    st.error("⚠ No relevant content found in selected PDFs. Try selecting more PDFs or re-ingesting.")
                    st.stop()

                answer_text = generate_answer(user_q, retrieved)

                # Extract simple "Sources:" line if present
                sources = []
                for line in answer_text.splitlines():
                    if line.strip().lower().startswith("sources:"):
                        parts = line.split(":", 1)[1]
                        sources = [p.strip() for p in parts.split(",") if p.strip()]
                        break

                # Append assistant response
                st.session_state.history.append({
                    "role": "assistant",
                    "query": user_q,
                    "answer": answer_text,
                    "sources": sources,
                    "retrieved": retrieved,
                    "ts": datetime.now().timestamp(),
                })

                # Save automatically
                save_history_to_disk(st.session_state.history)

                # request clear on next run
                st.session_state.clear_input = True
                st.rerun()

            except Exception as e:
                st.error(f"Error retrieving/generating: {e}")

    # Clear input button
    if clear_input_clicked:
        st.session_state.clear_input = True
        st.rerun()

    # Save history button
    if save_hist_clicked:
        ok = save_history_to_disk(st.session_state.history)
        st.success("History saved." if ok else "Save failed.")

    # Clear history (local only) button
    if clear_hist_clicked:
        st.session_state.history = []
        save_history_to_disk(st.session_state.history)
        st.rerun()

# -------------------
# Right column: info
# -------------------
with col_right:
    st.markdown("### Ingested files")
    last_files = st.session_state.get("last_ingested", [])
    if last_files:
        for f in last_files:
            st.write(f"• {f}")
    else:
        st.write("No PDFs ingested yet.")

    st.markdown("---")
    st.markdown("### Tips")
    st.write("- Upload and ingest multiple PDFs from the sidebar.")
    st.write("- Use the checkboxes to choose which PDFs to search.")
    st.write("- Increase k to retrieve more chunks (for longer documents).")
    st.write("- Export your chat from the sidebar history controls.")
