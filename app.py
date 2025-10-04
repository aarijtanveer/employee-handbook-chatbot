import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import re

# ----------------------------
# CONFIG
# ----------------------------
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

# LLM client
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.1-70b-versatile"
)

# ----------------------------
# HELPERS
# ----------------------------

def clean_and_translate(query: str) -> str:
    """
    Detects Roman Urdu and rephrases query into clean English before retrieval.
    Uses regex + LLM to refine.
    """
    # Quick Roman Urdu detection (naive, but works for most HR queries)
    roman_urdu_keywords = ["chutti", "chuttian", "mujhe", "kitni", "karne", "tareeqa"]
    if any(word in query.lower() for word in roman_urdu_keywords):
        translation_prompt = f"Translate the following Roman Urdu into clear English for HR context:\n\n{query}"
        translated = llm.invoke(translation_prompt)
        return translated.content.strip()
    return query

def retrieve_answer(query: str, k: int = 6):
    """
    Fetch relevant handbook content using ChromaDB.
    """
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    docs = db.similarity_search(query, k=k)
    if not docs:
        return None, None

    context = "\n\n".join([d.page_content for d in docs])
    return context, docs

def generate_answer(query: str, context: str) -> str:
    """
    Ask the LLM to generate a concise, helpful answer using handbook context.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an HR Assistant. Only answer using the Employee Handbook content provided."),
        ("human", "Question: {query}\n\nEmployee Handbook Content:\n{context}\n\nAnswer clearly and practically.")
    ])
    final_prompt = prompt_template.format(query=query, context=context)
    response = llm.invoke(final_prompt)
    return response.content.strip()

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("📘 Employee Handbook Assistant")
st.write("Ask me questions in **English or Roman Urdu** about the HR policies.")

query = st.text_input("Your question:")

if query:
    # Step 1: Translate/rephrase
    refined_query = clean_and_translate(query)

    # Step 2: Retrieve from handbook
    context, docs = retrieve_answer(refined_query)

    if not context:
        st.error("I couldn’t find anything relevant in the handbook.")
    else:
        # Step 3: Generate Answer
        answer = generate_answer(refined_query, context)
        st.success(answer)

        # Optional: Debug info
        with st.expander("See retrieved context"):
            for i, d in enumerate(docs, 1):
                st.write(f"**Snippet {i}:** {d.page_content[:500]}...")
