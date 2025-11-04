# vectorstore.py
import os
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai")


def _get_embeddings():
    """
    Choose an embedding backend.
    Priority:
    1. OPENAI (needs OPENAI_API_KEY)
    2. Google Gemini embeddings (needs GOOGLE_API_KEY)
    """
    if EMBEDDING_MODEL == "openai":
        return OpenAIEmbeddings()  # uses text-embedding-3-large by default in 2025 lc wrappers :contentReference[oaicite:5]{index=5}
    else:
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # :contentReference[oaicite:6]{index=6}


def get_or_create_vectorstore() -> Chroma:
    """
    Create a persistent Chroma vectorstore on disk.
    If empty, seed it with a few demo docs so the pipeline actually returns something. :contentReference[oaicite:7]{index=7}
    """
    embeddings = _get_embeddings()
    vs = Chroma(
        collection_name="agentic-ai",
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
    )

    # seed only if empty
    if vs._collection.count() == 0:
        seed_docs: List[Document] = [
            Document(
                page_content=(
                    "Agentic AI refers to LLM-based systems that can plan, call tools, "
                    "and collaborate with other specialized agents to complete tasks."
                ),
                metadata={"source": "seed", "topic": "agentic-ai"},
            ),
            Document(
                page_content=(
                    "Multi-agent workflows often follow retrieve -> validate -> synthesize. "
                    "Retrieval pulls context, validation filters hallucinations, synthesis writes the answer."
                ),
                metadata={"source": "seed", "topic": "pipelines"},
            ),
            Document(
                page_content=(
                    "LangGraph adds stateful DAG-style orchestration on top of LangChain components "
                    "for long-running autonomous agents."
                ),
                metadata={"source": "seed", "topic": "langgraph"},
            ),
        ]
        vs.add_documents(seed_docs)
        vs.persist()

    return vs


def get_retriever():
    vs = get_or_create_vectorstore()
    return vs.as_retriever(search_kwargs={"k": 4})
