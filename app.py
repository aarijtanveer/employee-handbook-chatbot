import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from deep_translator import GoogleTranslator
import re

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
    """Translate Roman Urdu → English using local translator."""
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(query)
    except Exception:
        translated = query
    return translated.strip()

def boost_query(query: str) -> str:
    """Boost queries by appending domain-specific keywords."""
    keywords = {
        "leave": ["leave", "annual leave", "casual leave", "sick leave", "holiday", "vacation"],
        "notice": ["notice period", "resignation", "exit policy"],
        "probation": ["probation", "confirmation", "joining rules"],
        "working hours": ["working hours", "timings", "shifts"],
        "other": ["benefits", "allowances", "disciplinary", "promotion", "increment"]
    }

    boosted = query
    for key, kws in keywords.items():
        if key in query.lower():
            boosted += " " + " ".join(kws)
    return boosted

def search_docs(query: str, k: int = 6):
    """Retrieve top-k relevant chunks from Chroma with keyword boosting."""
    boosted_query = boost_query(query)
    results = db.similarity_search(boosted_query, k=k)
    return results

def build_answer(query: str, docs):
    """Force the model to only answer from handbook content."""
    if not docs:
        return "Sorry, I couldn’t find anything in the handbook for that."

    context_texts = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are an HR assistant. 
    ONLY use the following Employee Handbook content to answer. 
    Do NOT guess. 
    If the answer is not found in the handbook, reply: "Sorry, I couldn’t find anything in the handbook."

    Handbook Content:
    {context_texts}

    Question: {query}

    Answer strictly based on handbook content:
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
    boosted_query = boost_query(refined_query)

    st.write(f"🔍 Interpreted Query: **{refined_query}**")
    if boosted_query != refined_query:
        st.caption(f"(Boosted for better retrieval: {boosted_query})")

    docs = search_docs(refined_query, k=6)
    answer = build_answer(refined_query, docs)

    st.write("### Answer:")
    st.write(answer)

    # Show sources
    with st.expander("📂 Sources"):
        for i, d in enumerate(docs, 1):
            st.markdown(f"**Source {i}:** {d.metadata.get('source','unknown')}")
            snippet = re.sub(r"\s+", " ", d.page_content[:400])
            st.caption(snippet + "...")
