# ChromaDB setup and PDF ingestion

# this is a pipeline in 4 steps

# first step is loading produces Document object one per page holding metadata and row text
# second step is chunk plitting and naturally produces chunks
# third step is embedding chunk : from text to geometry produces vectors
# fourth step is database persistance

import hashlib

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config


# ── Stage 0: Embedding Model
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads and returns the sentence-transformer embedding model.

    HuggingFaceEmbeddings is LangChain's wrapper around the sentence-transformers
    library. On first call it downloads the model weights from HuggingFace Hub
    (~90MB for MiniLM) and caches them locally in ~/.cache/huggingface/.
    Subsequent calls load from cache and are near-instant.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        # encode_kwargs controls the embedding computation itself.
        # normalize_embeddings=True scales every vector to unit length (L2 norm = 1).
        # This makes cosine similarity equivalent to dot product, which is what
        # ChromaDB uses internally. Skipping this can silently degrade search quality.
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Stage 1: Load pdf file
def load_pdf(file_path: str) -> list:
    """
    Loads a PDF from disk and returns a list of LangChain Document objects,
    one per page. Each Document carries:
      - page_content (str): the raw extracted text of that page.
      - metadata (dict): automatically populated with 'source' (file path)
        and 'page' (0-indexed page number).

    We store the file path (not the uploaded file object) because PyPDFLoader
    reads from disk. In app.py, we'll save the uploaded bytes to a temp file
    first, then pass that path here.
    """
    loader = PyPDFLoader(file_path)
    # .load() reads every page and returns the list of Document objects.
    documents = loader.load()
    return documents


# ── Stage 2: Splitting into chunks: this is here the chunking happens
def split_documents(documents: list) -> list:
    """
    Takes the list of page-level Documents and breaks them into smaller,
    overlapping chunks suitable for embedding.

    Why RecursiveCharacterTextSplitter specifically?
    It tries separators in priority order: ["\\n\\n", "\\n", " ", ""]
    meaning it first tries to split on paragraph breaks, then line breaks,
    then spaces, and only resorts to mid-word splitting as a last resort.
    This preserves linguistic boundaries and produces much better embeddings
    than a naive fixed-size slice.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    return chunks


# ── Stage 3 and 4: Embedding each chunk and produce their corresponding vector
# and store the produce vector
def _compute_file_hash(file_path: str) -> str:
    """
    Computes a SHA-256 hash of the file's raw bytes.

    This is our duplicate detection mechanism. Before ingesting a PDF, we hash
    its content and check whether any document in ChromaDB already carries that
    hash in its metadata. If yes, we skip re-ingestion entirely.

    SHA-256 produces a 64-character hexadecimal string that is unique to the
    file's exact content — rename the file and the hash stays the same;
    change a single character and the hash changes completely.
    """

    hasher = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# The hole pipeline function
def ingest_pdf(file_path: str, embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """
    Full ingestion pipeline: Load → Split → Embed → Store.
    Returns a Chroma vector store object ready to be used as a retriever.

    Includes duplicate detection: if the same PDF (same content hash) has
    already been ingested, this function loads the existing collection instead
    of re-embedding everything.
    """

    # ── Duplicate detection ───────────────────────────────────────────────────
    # We connect to the existing ChromaDB (or create it if first run) and check
    # whether this file's hash is already present in the stored metadata.
    file_hash = _compute_file_hash(file_path)

    # PersistentClient connects to (or creates) the on-disk ChromaDB store.
    # This is the line that physically creates the chroma_db/ directory.
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

    # get_or_create_collection is idempotent: safe to call whether the
    # collection already exists or not.
    collection = chroma_client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME
    )

    # Query the collection's metadata to see if this hash was stored before.
    # We use a where filter — ChromaDB's equivalent of a SQL WHERE clause.
    existing = collection.get(where={"file_hash": file_hash}, limit=1)

    if existing["ids"]:
        # This exact PDF has already been ingested. Skip the pipeline and
        # reconnect to the existing collection directly.
        print(f"[SmartBot] PDF already ingested (hash match). Loading from cache.")
        return _load_existing_store(embedding_model)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"[SmartBot] Loading PDF: {file_path}")
    documents = load_pdf(file_path)
    print(f"[SmartBot] Loaded {len(documents)} pages.")

    # ── Split ─────────────────────────────────────────────────────────────────
    chunks = split_documents(documents)
    print(f"[SmartBot] Split into {len(chunks)} chunks.")

    # Attach the file hash to every chunk's metadata. This is how we'll detect
    # duplicates on future runs. The metadata travels with the chunk into
    # ChromaDB and is stored alongside the vector.
    for chunk in chunks:
        chunk.metadata["file_hash"] = file_hash

    # ── Embed + Store ─────────────────────────────────────────────────────────
    # Chroma.from_documents() is the high-level LangChain API that:
    #   1. Calls embedding_model.embed_documents() on every chunk's text.
    #   2. Passes the resulting vectors + texts + metadata to ChromaDB.
    #   3. Returns a Chroma object that wraps the collection for LangChain use.
    #
    # persist_directory tells it to write to disk (not just keep in memory).
    # collection_name must match what we set in config so everything talks
    # to the same collection.
    print(f"[SmartBot] Embedding {len(chunks)} chunks — this may take a moment...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME,
    )
    print(
        f"[SmartBot] Ingestion complete. Vectors saved to {config.CHROMA_PERSIST_DIR}."
    )
    return vector_store


def _load_existing_store(embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """
    Reconnects to an already-populated ChromaDB collection without re-embedding.
    Called either by ingest_pdf() (when duplicate detected) or directly by
    app.py on subsequent runs where the user hasn't uploaded a new PDF yet.

    The embedding model must be passed in because ChromaDB doesn't store the
    model itself — only the vectors it produced. The model is needed to embed
    new queries at search time so they can be compared to the stored vectors.
    """
    return Chroma(
        persist_directory=config.CHROMA_PERSIST_DIR,
        embedding_function=embedding_model,
        collection_name=config.CHROMA_COLLECTION_NAME,
    )


# ── Public Interface ───────────────────────────────────────────────────────────
# This is the only function app.py needs to call. It encapsulates the decision:
# "should I ingest this PDF or load what's already there?"


def get_vector_store(
    file_path: str | None, embedding_model: HuggingFaceEmbeddings
) -> Chroma | None:
    """
    Entry point for app.py.

    - If file_path is provided: run the full ingestion pipeline (with duplicate
      guard) and return the resulting vector store.
    - If file_path is None: try to load an existing store (for cases where the
      app restarts but ChromaDB already has data from a previous session).
    - Returns None if no file was provided and no existing store is found,
      signalling to app.py that the user needs to upload a PDF first.
    """
    if file_path:
        return ingest_pdf(file_path, embedding_model)

    # No file provided — try to reconnect to whatever is already on disk.
    try:
        store = _load_existing_store(embedding_model)
        # Peek inside: if the collection is empty, treat it as non-existent.
        if store._collection.count() == 0:
            return None
        print("[SmartBot] Reconnected to existing vector store.")
        return store
    except Exception:
        # ChromaDB raises if the collection doesn't exist at all.
        return None
