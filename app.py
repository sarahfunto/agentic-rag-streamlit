import os
import streamlit as st
from dotenv import load_dotenv

# 1) Load .env
load_dotenv()

# 2) (Optional) LangSmith env flags (safe even if you don't use them)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
if os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# 3) Import your RAG function
from rag_agent import answer_question

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Agentic RAG (D2W10)", layout="centered")
st.title("D2W10 Agentic RAG App")
st.caption("Beginner version: local KB retrieval + optional Tavily web search + Groq answer with sources.")

# Show keys status (helpful for debugging)
with st.expander("✅ Setup check (keys)"):
    st.write("GROQ_API_KEY:", "✅ Found" if os.getenv("GROQ_API_KEY") else "❌ Missing")
    st.write("TAVILY_API_KEY:", "✅ Found" if os.getenv("TAVILY_API_KEY") else "⚠️ Missing (web search disabled)")

# Try to load notebook as text (optional requirement)
nb_text = None
try:
    with open("agentic_rag.ipynb", "r", encoding="utf-8") as f:
        nb_text = f.read()
except Exception:
    nb_text = None

if nb_text:
    with st.expander("📓 agentic_rag.ipynb detected (preview)"):
        st.code(nb_text[:2500] + "\n...\n(Truncated)", language="json")
else:
    st.info("📓 agentic_rag.ipynb not found (it's ok for now). Add it later if required.")

# Input
question = st.text_input(
    "Ask a question",
    placeholder="e.g., What is Agentic RAG? / What is RAG? / Latest news about LangChain?"
)

col1, col2 = st.columns([1, 1])
with col1:
    submit = st.button("Submit", type="primary")
with col2:
    clear = st.button("Clear")

if clear:
    st.rerun()

# Run
if submit:
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Thinking..."):
            res = answer_question(question)

        st.subheader("Answer")
        st.write(res.get("answer", ""))

        sources = res.get("sources", [])
        if sources:
            with st.expander("Sources used"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**[{i}]** {s.get('source', 'unknown')}")

        # Simple tips
        st.divider()
        st.caption("Tip: include words like 'latest' or 'today' to trigger web search.")
