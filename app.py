import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import re

# ----------------------------
# CONFIG
# ----------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"
TOP_K = 6  # number of chunks to retrieve
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ----------------------------
# Translation / Cleaning
# ----------------------------
def clean_and_translate(query: str) -> str:
    """Translate Roman Urdu to English if detected."""
    translation_prompt = f"""
    You are a language assistant.
    The user may ask in English or Roman Urdu.
    1. If it's Roman Urdu, translate it into clear English.
    2. If it's already English, keep it as-is.
    User query: {query}
    """

    response = llm.invoke([("user", translation_prompt)])
    return response.content.strip()

# ----------------------------
# Keyword Booster
# ----------------------------
def boost_query(query: str) -> str:
    """Add common HR-related keywords to improve retrieval."""
    boost_terms = "leave, annual leave, casual leave, sick leave, vacation, notice period, resignation, working hours, salary, policy, entitlement"
    return f"{query}. Related terms: {boost_terms}"

# ----------------------------
# Retriever
# ----------------------------
def retrieve_docs(query: str):
    """Retrieve top chunks from Chroma."""
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return db.similarity_search(query, k=TOP_K)

# ----------------------------
# Answer Builder (improved)
# ----------------------------
def build_answer(query: str, docs):
    """Force the model to only answer from handbook content, prioritizing keywords."""
    if not docs:
        return "Sorry, I couldn’t find anything in the handbook for that."

    # Prioritize docs with relevant HR terms
    priority_keywords = ["leave", "annual leave", "quota", "entitlement", "notice period", "resignation"]
    sorted_docs = sorted(
        docs,
        key=lambda d: sum(kw in d.page_content.lower() for kw in priority_keywords),
        reverse=True
    )

    # Take top chunks
    context_texts = "\n\n".join([d.page_content for d in sorted_docs[:4]])

    # Stronger answering prompt
    prompt = f"""
    You are an HR assistant for employees.
    ONLY use the following Employee Handbook content to answer.
    - Quote exact figures (e.g., number of days, percentages, periods).
    - If the info is not present, reply: "Sorry, I couldn’t find anything in the handbook."
    - Do not guess.

    Handbook Content:
    {context_texts}

    Question: {query}

    Final Answer:
    """

    response = llm.invoke([("user", prompt)])
    return response.content.strip()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📘 Employee Handbook Assistant")
st.write("Ask a question in English or Roman Urdu about HR policies.")

user_q = st.text_input("Your question:")

if user_q:
    refined_query = clean_and_translate(user_q)
    boosted_query = boost_query(refined_query)

    st.write(f"🔍 Interpreted Query: {refined_query}")

    docs = retrieve_docs(boosted_query)
    answer = build_answer(refined_query, docs)

    st.write("### Answer:")
    st.write(answer)
