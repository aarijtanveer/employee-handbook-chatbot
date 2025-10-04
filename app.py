import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
import re

# ----------------------------
# Setup
# ----------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"

embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

# ----------------------------
# Preprocess Query (clean + translate Roman Urdu if needed)
# ----------------------------
def clean_and_translate(query: str) -> str:
    query = re.sub(r"\s+", " ", query.strip())

    detect_lang_prompt = [
        ("system", "You are a language detector. Detect if this query is English or Roman Urdu."),
        ("user", query),
    ]
    detection = llm.invoke(detect_lang_prompt).content.lower()

    if "roman urdu" in detection:
        translation_prompt = [
            ("system", "You are a translator. Translate Roman Urdu into clear English without changing meaning."),
            ("user", query),
        ]
        translated = llm.invoke(translation_prompt).content
        return translated.strip()

    return query

# ----------------------------
# Force Answers Only From Handbook
# ----------------------------
def build_answer(query: str, docs):
    if not docs:
        return "Sorry, I couldn’t find anything in the handbook for that."

    # Keyword boosting for relevance
    priority_keywords = [
        "leave", "annual leave", "casual leave", "sick leave",
        "entitlement", "notice period", "resignation", "probation",
        "working hours", "holidays", "overtime"
    ]
    sorted_docs = sorted(
        docs,
        key=lambda d: sum(kw in d.page_content.lower() for kw in priority_keywords),
        reverse=True
    )

    context_texts = "\n\n".join([d.page_content for d in sorted_docs[:4]])

    messages = [
        ("system", "You are an HR assistant. ONLY answer using the Employee Handbook context below. "
                   "If the answer is not in the handbook, reply exactly: "
                   "'Sorry, I couldn’t find anything in the handbook.' Do not guess. Do not use outside knowledge."),
        ("user", f"""
        ----------------------
        Employee Handbook Context:
        {context_texts}
        ----------------------

        Question: {query}
        """)
    ]

    response = llm.invoke(messages)
    return response.content.strip()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📘 Employee Handbook Assistant")
st.write("Ask a question in English or Roman Urdu about HR policies.")

query = st.text_input("Your question:")

if query:
    refined_query = clean_and_translate(query)
    st.write("🔍 **Interpreted Query:**", refined_query)

    docs = db.similarity_search(refined_query, k=6)
    answer = build_answer(refined_query, docs)

    st.write("### Answer:")
    st.write(answer)
