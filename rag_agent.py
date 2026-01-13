
import os
from typing import Dict, Any, List

from tavily import TavilyClient

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------
# 1) Build retriever (FAISS)
# -------------------------
def build_retriever(kb_path: str = "data/kb.txt"):
    loader = TextLoader(kb_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 4})


# -------------------------
# 2) Tools (KB + Web)
# -------------------------
def kb_retrieve(retriever, query: str):
    # New LangChain: retriever.invoke(query)
    try:
        docs = retriever.invoke(query)
    except Exception:
        # Older versions fallback
        docs = retriever.get_relevant_documents(query)

    results = []
    for d in docs:
        results.append({
            "source": d.metadata.get("source", "data/kb.txt"),
            "content": d.page_content
        })
    return results


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_key:
        return [{"source": "tool:web_search", "content": "TAVILY_API_KEY missing. Web search unavailable."}]

    client = TavilyClient(api_key=tavily_key)
    res = client.search(query=query, max_results=max_results)

    out = []
    for r in res.get("results", []):
        out.append({
            "source": r.get("url", "unknown"),
            "content": f"{r.get('title','')}\n{r.get('content','')}"
        })
    return out if out else [{"source": "tool:web_search", "content": "No web results."}]


# -------------------------
# 3) Simple "agentic loop"
#    Decide -> Retrieve -> (optional web) -> Answer with sources
# -------------------------
def should_use_web(question: str) -> bool:
    # Règle simple débutant: web si question "latest/today/news/2026/current" ou "search web"
    q = question.lower()
    triggers = ["today", "latest", "news", "current", "2026", "search the web", "web", "recent"]
    return any(t in q for t in triggers)


def format_sources(items: List[Dict[str, str]]) -> str:
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"[{i}] {it['source']}")
    return "\n".join(lines) if lines else "No sources"


def answer_question(question: str, model: str = "llama-3.1-8b-instant") -> Dict[str, Any]:
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        return {"answer": "Missing GROQ_API_KEY in .env", "sources": [], "raw": None}

    retriever = build_retriever()

    # Step A: always retrieve from KB first
    kb_hits = kb_retrieve(retriever, question)

    # Step B: optionally use web tool
    web_hits = web_search(question) if should_use_web(question) else []

    # Build context for the LLM (keep it short)
    context_parts = []
    if kb_hits:
        context_parts.append("KB SOURCES:\n" + "\n\n".join(
            [f"Source: {x['source']}\n{x['content']}" for x in kb_hits]
        ))
    if web_hits:
        context_parts.append("WEB SOURCES:\n" + "\n\n".join(
            [f"Source: {x['source']}\n{x['content']}" for x in web_hits]
        ))

    context = "\n\n".join(context_parts) if context_parts else "No context found."

    llm = ChatGroq(model=model, temperature=0)

    system = (
        "You are a helpful assistant. Use ONLY the provided sources to answer. "
        "If the sources do not contain the answer, say you don't know. "
        "Finish with a short 'Sources:' list."
    )

    prompt = f"""{system}

Question: {question}

Sources content:
{context}

Write a clear beginner-friendly answer in 3-6 sentences.
Then add:
Sources:
- [1] ...
- [2] ...
(use the sources list numbers)
"""

    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)

        all_sources = kb_hits + web_hits
        return {"answer": text, "sources": all_sources, "raw": None}
    except Exception as e:
        return {"answer": f"Error while calling Groq: {e}", "sources": [], "raw": None}


# Local test
if __name__ == "__main__":
    print(answer_question("What is Agentic RAG?")["answer"])
