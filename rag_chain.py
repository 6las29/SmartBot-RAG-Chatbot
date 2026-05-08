# LangChain RAG logic LLM chain
# rag_chain.py — Retrieval-Augmented Generation Chain
# The chain here is : get-question() split it into two parallels road,
# First road vectorize the question and return the 4 closer one
# second road associate the question into a string that represent the context;
# That context is associated with the 4 chunks and get passed to the LLM
# The LLM produces a result and send it back;
# That result is unwrapped a the raw string is pass to the app.py as response.

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import config


def get_retriever(vector_store: Chroma):
    """
    Wraps the CHroma vector into a LangChain object
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_K},
    )


def get_prompt() -> PromptTemplate:
    """
    This is here I build the question, prompt template have two placeholders:
    context and question.
    """

    template = """You are a helpful assistant specialized in answering questions \
    about documents. Use ONLY the following context extracted from the document to \
    answer the question. If the answer is not contained in the context, say clearly \
    that you don't know based on the provided document. Do not use any outside knowledge.
 
    Keep your answer concise and grounded in the provided text.
 
    Context:
    {context}
 
    Question: {question}
 
    Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )


def get_llm():
    """
    Instantiates the language model based on the provider set in config.py.
    """

    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=config.GROQ_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
            streaming=True,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: '{config.LLM_PROVIDER}'. "
            "Set LLM_PROVIDER to 'groq' in config.py."
        )


def _format_docs(docs: list) -> str:
    """
    Converts a list of LangChain Document objects into a single text block
    that can be injected into the {context} placeholder of the prompt.

    Each chunk is separated by a double newline so the LLM can visually
    distinguish where one excerpt ends and the next begins. We also prepend
    the source page number for each chunk so that, if we ever want to add
    source attribution to the UI, the information is already in the context.
    """
    formatted_chunks = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        # Page numbers from PyPDFLoader are 0-indexed, so we add 1 for humans.
        formatted_chunks.append(f"[Page {int(page) + 1}]\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)


def build_rag_chain(vector_store: Chroma):
    """
    Assembles and return the full chain using LangChain Expression Language.
    The chain is construct with pipe operators that redirect the output of the
    last operation into the input of the next.
    """

    retriever = get_retriever(vector_store)
    prompt = get_prompt()
    llm = get_llm()

    chain = (
        # Stage 1 — Retrieve and format context, pass question through unchanged.
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        # Stage 2 — Fill the prompt template with context and question.
        | prompt
        # Stage 3 — Send the assembled prompt to the LLM.
        | llm
        # Stage 4 — Extract the plain text string from the LLM's response object.
        | StrOutputParser()
    )

    return chain
