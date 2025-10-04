import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import re
from langdetect import detect

# ----------------------------
# CONFIG
# ----------------------------
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Force embeddings on CPU (fix for Streamlit Cloud)
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBED_MODEL, model_kwargs={"device": "cpu"}
)

# ✅ Use Groq stable model
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model="llama-3.1-70b-versatile"
)

# ----------------------------
# UTILS
# ----------------------------
def clean_and_translate(query: str) -> str:
    """Clean query, detect language, and translate Roman Urdu to English if needed."""
    cleaned = re.sub(r"[^\w\s]", "", query).strip()

    try:
        lang = detect(cleaned)
    except Exception:
        lang = "en"

    # If Urdu (or Roman Urdu detected as 'ur'), translate to English using LLM
    if lang == "ur":
        translation_prompt = ChatPromptTemplate.from_template(
            "Translate this Roman Urdu text into English clearly:\n\n{q}"
        )
        translated = llm.invoke(translation_prompt.format_messages(q=cleaned))
        return translated.content.strip()

    return cleaned


def boost_keywords(query: str) -> str:
    """Boost retrieval by adding synonyms/keywords for HR terms."""
    boosts = {
        "leave": ["vacation", "annual leave", "holidays", "chutti"],
        "notice": ["resignation", "exit period", "termination"],
        "job": ["employment", "work", "additional job", "dual employment"],
    }

    lower_q = query.lower()
    extra = []
    for key, kws in boosts.items():
        if key in lower_q:
            extra.extend(kws)

    if extra:
        query += " " + " ".join(extra)

    return query


def answer_question(query: str):
    """Main pipeline: clean, translate, boost, retrieve, and answer."""
    cleaned = clean_and_translate(query)
    boosted = boost_keywords(cleaned)

    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    docs = db.similarity_search(boosted, k=5)

    if not docs:
        return boosted, "Sorry, I couldn’t find anything in the handbook."

    context = "\n\n".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_template(
        """
        You are an HR assistant. Answer only using the context below (from Employee Handbook).
        If the answer is not found in the context, reply with:
        "Sorry, I couldn’t find anything in the handbook."

        Context:
        {context}

        Question:
        {question}
        """
    )

    response = llm.invoke(prompt.format_messages(context=context, question=boosted))
    return boosted, response.content.strip()


# ----------------------------
# STREAMLIT APP
# ----------------------------
st.title("📘 Employee Handbook Assistant")
st.write("Ask a question in English or Roman Urdu about HR policies.")

query = st.text_input("Your question:")

if query:
    with st.spinner("Thinking..."):
        try:
            refined_query, answer = answer_question(query)
            st.write(f"🔍 **Interpreted Query:** {refined_query}")
            st.write(f"\n**Answer:**\n{answer}")
        except Exception as e:
            st.error(f"⚠️ Error while generating answer: {e}")
