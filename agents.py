# agents.py
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from vectorstore import get_retriever


def get_llm():
    """
    Simple LLM factory: chooses OpenAI first, else Gemini. :contentReference[oaicite:8]{index=8}
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        # gpt-4.5 or o4-mini — adjust to your account
        return ChatOpenAI(model="gpt-4.5-mini", temperature=temperature)
    elif os.getenv("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature)
    else:
        raise RuntimeError("No LLM provider configured. Set OPENAI_API_KEY or GOOGLE_API_KEY.")


# 1. RETRIEVAL AGENT -----------------------------------------------------------

def retrieval_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input  : { 'query': str }
    Output : { 'query': str, 'retrieved_docs': [str] }
    """
    query = state["query"]
    retriever = get_retriever()
    docs: List[Document] = retriever.get_relevant_documents(query)
    state["retrieved_docs"] = [d.page_content for d in docs]
    return state


# 2. VALIDATION AGENT ----------------------------------------------------------

def validation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take retrieved docs, ask the LLM to drop contradictory / irrelevant ones.
    Output: 'validated_docs': [str]
    """
    llm = get_llm()

    docs = state.get("retrieved_docs", [])
    if not docs:
        state["validated_docs"] = []
        return state

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict fact-checking assistant. "
                "You will be given a user query and a set of context passages. "
                "Return ONLY the passages that are (1) on-topic and (2) non-contradictory. "
                "Return them as a numbered list.",
            ),
            ("user", "User query: {query}\n\nContext passages:\n{contexts}\n"),
        ]
    )

    joined_contexts = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(docs, start=1))
    chain = prompt | llm
    result = chain.invoke({"query": state["query"], "contexts": joined_contexts})
    validated_text = result.content

    # very lightweight parsing: keep lines that look like a passage
    validated_list = [
        line for line in validated_text.splitlines()
        if line.strip() and not line.strip().lower().startswith("passages:")
    ]
    state["validated_docs"] = validated_list
    return state


# 3. SYNTHESIS AGENT -----------------------------------------------------------

def synthesis_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take validated docs + original query, produce final answer.
    """
    llm = get_llm()

    validated = state.get("validated_docs") or state.get("retrieved_docs") or []
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior research assistant. Use ONLY the provided validated context "
                "to answer the user's question. If something is missing, say so, but still give the best answer.",
            ),
            (
                "user",
                "User question: {query}\n\nValidated context:\n{context}\n\nWrite a structured answer.",
            ),
        ]
    )
    chain = prompt | llm
    result = chain.invoke(
        {
            "query": state["query"],
            "context": "\n".join(validated) if validated else "NO CONTEXT",
        }
    )
    state["answer"] = result.content
    return state
