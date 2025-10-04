import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langdetect import detect
import re

# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"
TOP_K = 6

# Groq LLM (stable supported model)
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model="llama-3.1-8b-instant"
)

# ----------------------------
# Helper functions
# ----------------------------
def clean_and_translate(query: str) -> str:
    """Detect Roman Urdu/Urdu, translate to English for better retrieval."""
    try:
        lang = detect(query)
    except:
        lang = "en"

    if lang in ["ur", "ro"]:
        translation_prompt = ChatPromptTemplate.from_messages([
            ("system", "Translate Roman Urdu or Urdu to English, keeping meaning intact."),
            ("user", query)
        ])
        translated = llm.invoke(translation_prompt.format_messages())
        return translated.content.strip()
    else:
        return query

def boost_keywords(query: str) -> str:
    """Add HR-related keywords to improve retrieval hits."""
    extra_keywords = [
        "leave policy", "annual leave", "notice period",
        "probation", "resignation", "additional employment",
        "working hours", "benefits", "holidays"
    ]
    return query + " " + " ".join(extra_keywords)

def answer_question(query: str):
    cleaned = clean_and_translate(query)
    boosted = boost_keywords(cleaned)

    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    docs = db.similarity_search(boosted, k=TOP_K)
    if not docs:
        return cleaned, "Sorry, I couldn’t find anything in the handbook."

    context = "\n\n".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an HR assistant. Use the provided Employee Handbook context to answer questions accurately. If the context is not enough, say 'Sorry, I couldn’t find anything in the handbook.' Keep answers concise and relevant."),
        ("user", f"Context:\n{context}\n\nQuestion: {cleaned}\n\nAnswer:")
    ])

    response = llm.invoke(prompt.format_messages())
    return cleaned, response.content.strip()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📘 Employee Handbook Assistant")
st.write("Ask a question in English or Roman Urdu about HR policies.")

query = st.text_input("Your question:")

if query:
    with st.spinner("Thinking..."):
        refined_query, answer = answer_question(query)
    st.write(f"🔍 **Interpreted Query:** {refined_query}")
    st.write(f"### Answer:\n{answer}")
