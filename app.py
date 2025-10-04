import streamlit as st
import re
from langdetect import detect
from deep_translator import GoogleTranslator
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"

# Force embeddings to CPU (important for Streamlit Cloud)
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"}
)
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# Groq LLM for answering
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model="llama3-8b-8192"
)

# ----------------------------
# Keyword Booster
# ----------------------------
KEYWORD_BOOST = {
    "leave": ["leave entitlement", "annual leave", "vacation", "casual leave", "sick leave", "chutti"],
    "notice": ["notice period", "resignation", "exit", "contract termination"],
    "job": ["additional employment", "dual job", "outside work", "secondary employment"]
}

def boost_query(query: str) -> str:
    """Boost queries with HR-related synonyms."""
    q = query.lower()
    for key, synonyms in KEYWORD_BOOST.items():
        if key in q:
            query += " " + " ".join(synonyms)
    return query

# ----------------------------
# Translation & Cleaning
# ----------------------------
def clean_and_translate(query: str) -> str:
    """Detect language, translate Roman Urdu → English using GoogleTranslator."""
    try:
        lang = detect(query)
    except:
        lang = "en"

    if lang != "en":
        try:
            query = GoogleTranslator(source="auto", target="en").translate(query)
        except Exception:
            # fallback: leave as is
            pass

    query = re.sub(r"[^a-zA-Z0-9\s]", "", query)
    return query

# ----------------------------
# Retrieval + LLM Answer
# ----------------------------
def answer_question(query: str):
    cleaned = clean_and_translate(query)
    boosted = boost_query(cleaned)

    docs = vectordb.similarity_search(boosted, k=5)

    if not docs:
        return boosted, "Sorry, I couldn’t find anything in the handbook."

    context = "\n\n".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an HR assistant. Answer ONLY from the Employee Handbook text below. "
         "If the answer is not in the handbook, reply: 'Sorry, I couldn’t find anything in the handbook.'\n\n"
         f"Employee Handbook:\n{context}"),
        ("human", boosted)
    ])

    try:
        response = llm.invoke(prompt.format_messages())
        return boosted, response.content.strip()
    except Exception as e:
        return boosted, f"⚠️ Error while generating answer: {str(e)}"

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📘 Employee Handbook Assistant")
st.write("Ask a question in English or Roman Urdu about HR policies.")

query = st.text_input("Your question:")

if query:
    refined_query, answer = answer_question(query)
    st.markdown(f"🔍 **Interpreted Query:** {refined_query}")
    st.markdown(f"### Answer:\n{answer}")
