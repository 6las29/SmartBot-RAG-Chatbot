# Entry point: This is the chat interface
import os
import tempfile

import streamlit as st

from rag_chain import build_rag_chain
from vector_store import get_embedding_model, get_vector_store

# ── Page Configuration ────────────────────────────────────────────────────────
# st.set_page_config() must be the very first Streamlit call in the script.

st.set_page_config(
    page_title="SmartBot",
    page_icon="🤖",
    layout="centered",
)


# ── Cached Resource: Embedding Model ──────────────────────────────────────────
# @st.cache_resource tells Streamlit to run this function exactly once for the
# entire server process lifetime, then reuse the cached return value on every
# subsequent call. Without this, the embedding model would be reloaded from
# disk on every user interaction — a multi-second delay each time.
#
# We wrap get_embedding_model() here rather than in vector_store.py because
# caching is a UI-layer concern. vector_store.py stays pure and framework-agnostic.


@st.cache_resource
def load_embedding_model():
    return get_embedding_model()


# ── Session State Initialization ──────────────────────────────────────────────
# These guards run on every re-execution of the script. On the first run,
# the keys don't exist yet so they get initialized. On every subsequent run,
# the keys exist so the guard is skipped and the existing values survive.
#
# Think of this block as declaring your application's "memory slots".

if "messages" not in st.session_state:
    # The full conversation history, stored as a list of dicts with
    # {"role": "user" | "assistant", "content": "..."} — the standard
    # format for chat UIs and also what most LLM APIs expect.
    st.session_state.messages = []

if "chain" not in st.session_state:
    # The assembled RAG chain. None until a PDF has been successfully ingested.
    st.session_state.chain = None

if "ingested_file" not in st.session_state:
    # Tracks the name of the last ingested file. Used to detect when the user
    # uploads a different PDF so we can rebuild the chain for the new document.
    st.session_state.ingested_file = None


# ── UI Layout ─────────────────────────────────────────────────────────────────

st.title("🤖 Your wonderful SmartBot as a EPITA's Student")
st.caption("Upload your course PDF and let's dive into it.")

# The sidebar is a natural home for configuration controls — it's always visible
# but doesn't compete with the main content area where the conversation lives.
with st.sidebar:
    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        label="Upload a PDF",
        type=["pdf"],
        # help text appears as a tooltip on the upload widget.
        help="Upload the PDF you want to chat with.",
    )

    # Show a status indicator so the user knows what state the system is in.
    if st.session_state.chain is not None:
        st.success(f"Ready: **{st.session_state.ingested_file}**")
    else:
        st.info("Upload a PDF to get started.")

    st.divider()

    # A reset button lets the user start a fresh conversation without
    # refreshing the browser tab. Clearing session state keys triggers
    # a re-run, which re-initializes everything to defaults via the guards above.
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = None
        st.session_state.ingested_file = None
        st.rerun()


# ── PDF Ingestion ─────────────────────────────────────────────────────────────
# This block runs only when the user has uploaded a file AND it's different
# from the one already ingested. The second condition prevents re-ingestion
# on every re-run (which would happen on every keypress otherwise).

if uploaded_file and uploaded_file.name != st.session_state.ingested_file:

    with st.spinner(f"Treating **{uploaded_file.name}**… this may take a moment."):

        # PyPDFLoader requires a file path on disk, not an in-memory buffer.
        # We write the uploaded bytes to a named temporary file, use its path
        # for ingestion, then delete it — the vectors are now in ChromaDB,
        # so the temp file is no longer needed.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            embedding_model = load_embedding_model()

            # get_vector_store() runs the full Load → Split → Embed → Store
            # pipeline (or loads from cache if the same PDF was already ingested).
            vector_store = get_vector_store(tmp_path, embedding_model)
            if vector_store is not None:
                # build_rag_chain() wires the retriever, prompt, and LLM together
                # into a single callable pipeline and stores it in session state.
                st.session_state.chain = build_rag_chain(vector_store)
                st.session_state.ingested_file = uploaded_file.name

            # Reset conversation history when a new document is loaded
            # so old answers about a different PDF don't linger in the chat.
            st.session_state.messages = []

        finally:
            # Always clean up the temp file, even if ingestion raised an error.
            # The finally block runs regardless of whether an exception occurred.
            os.unlink(tmp_path)

    st.rerun()  # Refresh the UI to show the updated sidebar status indicator.


# ── Conversation History ───────────────────────────────────────────────────────
# On every re-run, we replay the full conversation history from session state
# into the chat UI. Streamlit renders each message as a styled chat bubble.
# This is what creates the illusion of a persistent conversation even though
# the script re-executes completely on every interaction.

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── Chat Input & Response Generation ──────────────────────────────────────────
# st.chat_input() renders a fixed input bar at the bottom of the page.
# It returns the submitted string when the user presses Enter, or None otherwise.
# When it returns None (i.e., the user hasn't submitted anything this re-run),
# the entire block below is skipped — Streamlit's conditional rendering at work.

if prompt := st.chat_input(
    placeholder="Ask a question about your document…",
    disabled=st.session_state.chain is None,  # Grey out until PDF is ingested.
):
    # Guard: don't process input if no chain is ready (belt-and-suspenders check
    # in addition to the disabled= parameter above).
    if st.session_state.chain is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    # Immediately append and render the user's message so the UI feels responsive.
    # The user sees their own message appear before the assistant starts answering.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the assistant's response token by token.
    # st.write_stream() accepts any Python iterator or generator and writes
    # each yielded value to the chat bubble as it arrives.
    # chain.stream() returns a generator that yields string tokens from the LLM
    # as they are produced — no waiting for the full response before display.
    with st.chat_message("assistant"):
        response = st.write_stream(st.session_state.chain.stream(prompt))

    # Once streaming is complete, persist the full response to session state
    # so it appears correctly on the next re-run.
    st.session_state.messages.append({"role": "assistant", "content": response})
