# Constants with the models names, chunk sizes

# config.py — SmartBot Central Configuration
# ─────────────────────────────────────────────────────────────────────────────
# This file is the single source of truth for every tunable parameter in the
# project.

import os

from dotenv import load_dotenv

# load_dotenv() reads your .env file and injects its key=value pairs into
# os.environ, making them available via os.getenv(). This way, your secrets
# (API keys) never live in the source code — only in .env, which is gitignored.

load_dotenv()

# The LLM provider in my case here is openai
LLM_PROVIDER = "openai"

# The key needed to access the api
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# The model to use with the question
OPENAI_MODEL_NAME = "gpt-4o-mini"

# The setting off temperature is here to assure that the model is determinst.
# This means that the answers must be close enough to the provided document
# no freestyle answer from the AI model itself.

LLM_TEMPERATURE = 0.0

# embeddings is the numerical representation of text. This implies that
# semantically close enough sentences end up close to each other. It is
# implement in this project by sentence-transformers that do the job well
# enough.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunk size and chunk overlap
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# The database folder on the folder disk
CHROMA_PERSIST_DIR = "./chroma_db"

# the collection here is related to the sql table
CHROMA_COLLECTION_NAME = "smartbot_docs"

# The retrieval windows that determine the number of chunks that will be
# retrieve on a question to the chatbot. The sweet spot is generarally 4
RETRIEVER_K = 4
