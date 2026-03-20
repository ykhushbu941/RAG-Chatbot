"""
Wattmonk RAG Chatbot — Streamlit Frontend
Connects to the FastAPI backend for RAG-powered responses.
"""

import streamlit as st
import httpx
import os
import time

# ─── Config ───────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Wattmonk AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit default elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* App background */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1c2e 0%, #1e3a5f 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #fbbf24 !important;
}

/* Main header area */
.main-header {
    background: linear-gradient(135deg, #0f1c2e 0%, #1e3a5f 60%, #0f2444 100%);
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
    display: flex;
    align-items: center;
    gap: 16px;
}

/* Chat message containers */
.user-message {
    background: linear-gradient(135deg, #1e3a5f, #2563eb);
    color: white;
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    margin-left: 15%;
    box-shadow: 0 2px 12px rgba(37, 99, 235, 0.25);
    font-size: 15px;
    line-height: 1.6;
}

.assistant-message {
    background: white;
    color: #1e293b;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    margin-right: 15%;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
    font-size: 15px;
    line-height: 1.7;
}

/* Source badges */
.badge-wattmonk {
    background: #fef3c7;
    border: 1px solid #f59e0b;
    color: #92400e;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.badge-nec {
    background: #dbeafe;
    border: 1px solid #3b82f6;
    color: #1e3a8a;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.badge-both {
    background: #ede9fe;
    border: 1px solid #7c3aed;
    color: #4c1d95;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.badge-general {
    background: #d1fae5;
    border: 1px solid #10b981;
    color: #064e3b;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* Suggestion buttons */
.stButton button {
    background: white !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 20px !important;
    color: #334155 !important;
    font-size: 13px !important;
    padding: 6px 14px !important;
    transition: all 0.15s !important;
}
.stButton button:hover {
    background: #f59e0b !important;
    border-color: #f59e0b !important;
    color: #000 !important;
}

/* Chat input */
.stChatInput > div {
    border-radius: 12px !important;
    border: 1.5px solid #e2e8f0 !important;
}
.stChatInput > div:focus-within {
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.15) !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Expander */
.streamlit-expanderHeader {
    background: #f8fafc !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Status indicators */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Chunk card */
.chunk-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    color: #475569;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "last_intent" not in st.session_state:
    st.session_state.last_intent = "—"
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = 0
if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_badge_html(intent: str) -> str:
    labels = {
        "wattmonk": ("badge-wattmonk", "⚡ Wattmonk Info"),
        "nec":      ("badge-nec",      "📋 NEC 2017"),
        "both":     ("badge-both",     "📚 Wattmonk + NEC"),
        "general":  ("badge-general",  "🌞 General Knowledge"),
    }
    cls, label = labels.get(intent, ("badge-general", "💬 General"))
    return f'<span class="{cls}">{label}</span>'


def call_api(message: str, history: list) -> dict:
    """Call the FastAPI backend."""
    payload = {
        "message": message,
        "history": [{"role": m["role"], "content": m["content"]} for m in history],
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{API_URL}/chat", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        return {"error": f"Cannot connect to API at {API_URL}. Make sure the backend is running."}
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


def check_api_health() -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{API_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Wattmonk AI")
    st.markdown("*RAG-Powered Solar Assistant*")
    st.divider()

    # API status
    api_healthy = check_api_health()
    if api_healthy:
        st.markdown('<span class="status-dot"></span> **API Connected**', unsafe_allow_html=True)
    else:
        st.error("⚠️ API Offline")
        st.markdown(f"Expected at: `{API_URL}`")

    st.divider()

    # Session stats
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        st.metric("Chunks Used", st.session_state.last_chunks)

    st.markdown(f"**Last Intent:** `{st.session_state.last_intent}`")

    st.divider()

    # Knowledge sources
    st.markdown("### 📚 Knowledge Base")
    with st.expander("⚡ Wattmonk Info", expanded=False):
        st.markdown("""
- Company overview & history  
- All 6 core services  
- Turnaround times & stats  
- Zippy technology  
- Culture & benefits
        """)
    with st.expander("📋 NEC 2017 (NFPA 70)", expanded=False):
        st.markdown("""
- Article 690: Full PV requirements  
- Section 690.7: Max voltage  
- Section 690.8: Circuit sizing  
- Section 690.9: Overcurrent protection  
- Articles 250, 705, 691
        """)
    with st.expander("🌞 General Solar", expanded=False):
        st.markdown("""
- Installation workflow (9 steps)  
- 15+ industry acronyms  
- Project types & components  
        """)

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")
    api_url_input = st.text_input("API URL", value=API_URL, key="api_url")
    if api_url_input != API_URL:
        API_URL = api_url_input

    show_chunks = st.toggle("Show retrieved chunks", value=True)
    show_debug = st.toggle("Show debug info", value=False)

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.last_intent = "—"
        st.session_state.last_chunks = 0
        st.session_state.retrieved_chunks = []
        st.rerun()

# ─── Main Content ─────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <div style="font-size:40px">⚡</div>
    <div>
        <h1 style="color:white;margin:0;font-size:26px;font-weight:700;">Wattmonk AI Assistant</h1>
        <p style="color:#94a3b8;margin:0;font-size:13px;font-family:'JetBrains Mono',monospace;">
            RAG-Powered · NEC 2017 · Solar Engineering · FastAPI + Streamlit
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Suggested questions (only shown when chat is empty)
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking:")
    suggestions = [
        "What services does Wattmonk offer?",
        "What is NEC Article 690?",
        "How fast are Wattmonk's permit plan sets?",
        "Max voltage for residential PV systems?",
        "Who founded Wattmonk and when?",
        "NEC requirements for overcurrent protection in PV?",
    ]
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state["pending_message"] = suggestion
                st.rerun()

    st.divider()
    st.markdown("""
    <div style="text-align:center;color:#94a3b8;font-size:14px;padding:20px;">
        I can answer questions about <strong>Wattmonk</strong> company info, 
        <strong>NEC 2017</strong> solar PV code requirements, and general solar industry topics.
    </div>
    """, unsafe_allow_html=True)

# ─── Chat History Display ─────────────────────────────────────────────────────

chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">⚡ {msg["content"]}</div>', unsafe_allow_html=True)
            if "intent" in msg:
                badge = get_badge_html(msg["intent"])
                chunks_info = f'<span style="color:#94a3b8;font-size:11px;margin-left:8px;">· {msg.get("chunks_used",0)} chunks retrieved</span>'
                st.markdown(f'<div style="margin:-4px 0 12px 0">{badge}{chunks_info}</div>', unsafe_allow_html=True)

# ─── Retrieved Chunks Display ─────────────────────────────────────────────────

if show_chunks and st.session_state.retrieved_chunks:
    with st.expander(f"🔍 Retrieved Chunks ({len(st.session_state.retrieved_chunks)} used for last response)", expanded=False):
        for i, chunk in enumerate(st.session_state.retrieved_chunks):
            source_emoji = {"wattmonk": "⚡", "nec": "📋", "general": "🌞"}.get(chunk["source"], "📄")
            st.markdown(f"""
            <div class="chunk-card">
                <strong>{source_emoji} {chunk['source'].upper()} — Score: {chunk['score']:.0f}</strong><br/>
                {chunk['preview']}
            </div>
            """, unsafe_allow_html=True)

# ─── Chat Input ───────────────────────────────────────────────────────────────

# Handle pending message from suggestion buttons
pending = st.session_state.pop("pending_message", None)

prompt = st.chat_input("Ask about Wattmonk, NEC 2017, or general solar topics...")

user_input = pending or prompt

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.total_queries += 1

    # Show user message immediately
    st.markdown(f'<div class="user-message">👤 {user_input}</div>', unsafe_allow_html=True)

    # Call API with spinner
    with st.spinner("⚡ Searching knowledge base and generating response..."):
        result = call_api(user_input, st.session_state.messages[:-1])

    if "error" in result:
        error_msg = f"❌ Error: {result['error']}"
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "intent": "error",
            "chunks_used": 0,
        })
        st.error(error_msg)
    else:
        # Store assistant response
        assistant_msg = {
            "role": "assistant",
            "content": result["reply"],
            "intent": result["intent"],
            "chunks_used": result["chunks_used"],
        }
        st.session_state.messages.append(assistant_msg)

        # Update session stats
        st.session_state.last_intent = result["intent"]
        st.session_state.last_chunks = result["chunks_used"]
        st.session_state.retrieved_chunks = result.get("retrieved_chunks", [])

        # Show response
        st.markdown(f'<div class="assistant-message">⚡ {result["reply"]}</div>', unsafe_allow_html=True)
        badge = get_badge_html(result["intent"])
        chunks_info = f'<span style="color:#94a3b8;font-size:11px;margin-left:8px;">· {result["chunks_used"]} chunks retrieved</span>'
        st.markdown(f'<div style="margin:-4px 0 12px 0">{badge}{chunks_info}</div>', unsafe_allow_html=True)

        # Debug info
        if show_debug:
            with st.expander("🐛 Debug Info", expanded=False):
                st.json({
                    "intent": result["intent"],
                    "source_label": result["source_label"],
                    "chunks_used": result["chunks_used"],
                    "model": "claude-sonnet-4-20250514",
                })

    st.rerun()


#py -m streamlit run app.py
#py -m uvicorn main:app --reload --port 8000