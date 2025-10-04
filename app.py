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
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6

# ----------------------------
# Load embedding model and DB
# ----------------------------
embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# ----------------------------
# Init LLM (Groq)
# ----------------------------
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

# ----------------------------
# Preprocess Query (clean + Roman Urdu → English)
# ----------------------------
def clean_and_translate(query: str) -> str:
    """Cleans user query and translates Roman Urdu to English if needed."""
    query = query.strip()
    query = re.sub(r"\s+", " ", query)

    try:
        lang = detect(query)
    except Exception:
        lang = "en"

    if lang != "en":
        translation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a translator. Convert Roman Urdu text into clear English, keeping the meaning."),
            ("human", query)
        ])
        try:
            translated = llm.invoke(translation_prompt.format_messages())
            return translated.content.strip()
        except Exception:
            return query  # fallback if translation fails

    return query

# ----------------------------
# Retrieval
# ----------------------------
def retrieve_answer(query: str):
    """Search similar docs from Chroma."""
    try:
        docs = db.similarity_search(query, k=TOP_K)
    except Exception:
        docs = []
    return docs

# ----------------------------
# Main Q&A Pipeline
# ----------------------------
def answer_question(query: str):
    """Main pipeline: preprocess → retrieve → LLM answer."""
    refined_query = clean_and_translate(query)
    docs = retrieve_answer(refined_query)

    if not docs:
        return refined_query, "Sorry, I couldn’t find anything in the handbook."

    # Trim docs to avoid hitting Groq’s token limits
    trimmed_docs = [d.page_content[:800] for d in docs[:3]]
    context = "\n\n".join(trimmed_docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an HR assistant. Answer ONLY from the Employee Handbook. "
                   "If the answer is not present, say: 'Sorry, I couldn’t find anything in the handbook.'"),
        ("human", f"Employee Handbook Context:\n{context}\n\nUser Question: {refined_query}\n\nAnswer:")
    ])

    try:
        response = llm.invoke(prompt.format_messages())
        return refined_query, response.content
    except Exception as e:
        return refined_query, f"⚠️ Error while generating answer: {str(e)}"

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📘 Employee Handbook Assistant")
st.write("Ask a question in English or Roman Urdu about HR policies.")

query = st.text_input("Your question:")

if query:
    refined_query, answer = answer_question(query)
    st.write(f"🔍 **Interpreted Query:** {refined_query}")
    st.write(f"**Answer:**\n{answer}")
