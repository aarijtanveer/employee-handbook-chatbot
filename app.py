import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import re
import os

# ----------------------------
# CONFIG
# ----------------------------
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Load Groq API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

# Initialize embeddings + Chroma
embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# ----------------------------
# Helper Functions
# ----------------------------
def clean_and_translate(query: str) -> str:
    """
    Clean and (if needed) translate Roman Urdu → English before retrieval.
    """
    query = query.strip()

    translation_prompt = f"""
    You are a translation assistant. If the following text is in Roman Urdu, translate it to clear English. 
    If it's already English, leave it as is. Preserve HR terminology.

    Text: "{query}"
    """

    try:
        response = llm.invoke([("user", translation_prompt)])
        translated = response.content.strip()
    except Exception as e:
        print(f"Translation failed, fallback to raw query: {e}")
        translated = query

    return translated


def search_docs(query: str, k: int = 5):
    """
    Retrieve top-k relevant chunks from Chroma.
    """
    results = db.similarity_search(query, k=k)
    return results


def build_answer(query: str, docs):
    """
    Build an answer using both the docs and the LLM.
    """
    if not docs:
        return "Sorry, I couldn’t find anything in the handbook for that."

    context_texts = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are an HR policy assistant. 
    Use ONLY the content below (from the Employee Handbook) to answer the question. 
    If the answer is not found, say "Sorry, I couldn’t find anything in the handbook about that."

    Handbook context:
    {context_texts}

    Question: {query}

    Final Answer:
    """

    response = llm.invoke([("user", prompt)])
    return response.content.strip()

# ----------------------------
# STREAMLIT APP
# ----------------------------
st.set_page_config(page_title="Employee Handbook Assistant", page_icon="📘")

st.title("📘 Employee Handbook Assistant")
st.write("Ask a question in English or Roman Urdu about HR policies.")

query = st.text_input("Your question:")

if query:
    refined_query = clean_and_translate(query)
    st.write(f"🔍 Interpreted Query: **{refined_query}**")

    docs = search_docs(refined_query, k=5)
    answer = build_answer(refined_query, docs)

    st.write("### Answer:")
    st.write(answer)

    # Show sources
    with st.expander("📂 Sources"):
        for i, d in enumerate(docs, 1):
            st.markdown(f"**Source {i}:** {d.metadata.get('source','unknown')}")
            snippet = re.sub(r"\s+", " ", d.page_content[:400])
            st.caption(snippet + "...")
