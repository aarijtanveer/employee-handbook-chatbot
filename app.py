import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import re

# ----------------------------
# Settings
# ----------------------------
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6

# Initialize embedding + DB
embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# Initialize LLM
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

# ----------------------------
# Synonym Expansion Dictionary
# ----------------------------
SYNONYMS = {
    "additional job": ["outside employment", "secondary employment", "moonlighting", "dual employment"],
    "chutti": ["leave", "annual leave", "vacation"],
    "notice period": ["resignation notice", "termination notice", "contract notice period"],
    "salary": ["pay", "compensation", "wages"],
}

def expand_with_synonyms(query: str) -> str:
    """Expand the query with known synonyms for better retrieval."""
    expanded = [query]
    q_lower = query.lower()
    for key, variants in SYNONYMS.items():
        if key in q_lower:
            expanded.extend(variants)
    return " OR ".join(expanded)


# ----------------------------
# Helper functions
# ----------------------------
def clean_and_translate(query: str) -> str:
    """Normalize user input (remove extra spaces, capitalize)."""
    q = re.sub(r"\s+", " ", query.strip())
    q = q[0].upper() + q[1:] if q else q
    return q

def retrieve_answer(query: str):
    """Search Chroma for relevant chunks."""
    expanded_query = expand_with_synonyms(query)
    docs = db.similarity_search(expanded_query, k=TOP_K)
    return docs


def answer_question(query: str):
    """Main pipeline: preprocess → retrieve → LLM answer."""
    refined_query = clean_and_translate(query)
    docs = retrieve_answer(refined_query)

    if not docs:
        return refined_query, "Sorry, I couldn’t find anything in the handbook."

    context = "\n\n".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an HR assistant. Answer ONLY from the Employee Handbook. "
                   "If the answer is not present, say: 'Sorry, I couldn’t find anything in the handbook.'"),
        ("human", f"Employee Handbook Context:\n{context}\n\nUser Question: {refined_query}\n\nAnswer:")
    ])

    response = llm.invoke(prompt.format_messages())
    return refined_query, response.content


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📘 Employee Handbook Assistant")
st.write("Ask a question in English or Roman Urdu about HR policies.")

query = st.text_input("Your question:")

if query:
    refined_query, answer = answer_question(query)

    st.markdown(f"🔍 **Interpreted Query:** {refined_query}")
    st.markdown(f"**Answer:**\n{answer}")
