"""Forge AI — Streamlit Chat Interface.

Interactive chat UI for the ML assistant with file upload,
tool visualization, and conversation history. Integrates with the FastAPI backend.
"""

import os
import sys
import uuid
import tempfile
from pathlib import Path
import requests

import streamlit as st
from dotenv import load_dotenv

# Allow running without `pip install -e .` (dev convenience).
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists():
    src_str = str(_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

load_dotenv()

# Get API URL from environment or default to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")

LOGO_PATH = _ROOT / "assets" / "logo.png"

_page_icon = "🔨"
if LOGO_PATH.exists():
    try:  # Optional: use the logo as favicon when Pillow is available.
        from PIL import Image

        _page_icon = Image.open(LOGO_PATH)
    except Exception:
        _page_icon = "🔨"

st.set_page_config(
    page_title="Forge AI — ML Assistant",
    page_icon=_page_icon,
    layout="wide",
)

# ─── Initialization ──────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Sidebar: Configuration ─────────────────────────────────────────────────

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.title("🔨 Forge AI")

    provider = st.selectbox(
        "LLM Provider",
        ["groq", "gemini", "openai"],
        help="All use OpenAI-compatible SDK. Groq & Gemini are free.",
    )

    model_options = {
        "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "gemini": ["gemini-2.0-flash", "gemini-1.5-flash"],
        "openai": ["gpt-4o-mini", "gpt-4o"],
    }
    model = st.selectbox("Model", model_options.get(provider, []))

    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    st.markdown("---")

    # File upload
    uploaded_file = st.file_uploader(
        "📂 Upload a CSV dataset",
        type=["csv"],
        help="Upload a dataset for the agent to analyze",
    )

    dataset_path = None
    if uploaded_file:
        with st.spinner("Uploading file to backend..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                upload_res = requests.post(f"{API_URL}/upload", files=files)
                upload_res.raise_for_status()
                dataset_path = upload_res.json()["file_path"]
                st.success(f"✅ Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Failed to upload dataset: {e}")

    st.markdown("---")
    st.markdown(
        "**Available Tools:**\n"
        "- 🔧 Dataset Analyzer\n"
        "- 📚 RAG (ML Docs)\n"
        "- 🔨 Code Generator\n"
        "- 📊 Model Evaluator\n"
        "- 🧹 Preprocessing Advisor\n"
        "- 🔍 Concept Explainer\n"
        "- 📈 Experiment Tracker (MLflow)"
    )

    if st.button("🗑️ Clear conversation"):
        try:
            requests.delete(f"{API_URL}/chat/{st.session_state.session_id}")
        except Exception as e:
            st.error(f"Failed to clear session on backend: {e}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# ─── Chat Interface ──────────────────────────────────────────────────────────

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show tool calls if present
        if "steps" in message and message["steps"]:
            with st.expander("🔧 Tool calls", expanded=False):
                for step in message["steps"]:
                    st.markdown(f"**{step['tool']}** → {step['output'][:300]}...")

# Chat input
if prompt := st.chat_input("Ask me about your ML workflow..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "session_id": st.session_state.session_id,
                    "query": prompt,
                    "provider": provider,
                    "model": model,
                    "temperature": temperature,
                    "file_path": dataset_path
                }
                api_res = requests.post(f"{API_URL}/chat", json=payload)
                api_res.raise_for_status()
                
                result = api_res.json()
                response = result.get("output", "")
                steps = result.get("steps", [])

                st.markdown(response)

                if steps:
                    with st.expander("🔧 Tool calls", expanded=False):
                        for step in steps:
                            st.markdown(
                                f"**{step['tool']}** → {step['output'][:300]}..."
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "steps": steps,
                })

            except requests.exceptions.HTTPError as e:
                error_msg = f"HTTP Error: {e.response.text if e.response else e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ {error_msg}",
                    "steps": [],
                })
            except Exception as e:
                st.error(f"Error communicating with backend: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {e}",
                    "steps": [],
                })
