# app.py
import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langdetect import detect
import requests
from typing import List

load_dotenv()

st.set_page_config(page_title="Employee Handbook Assistant", page_icon="🧾")
st.title("Employee Handbook Assistant")
st.write("Ask a question and I will answer using only the content inside the provided Employee Handbook.")

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-13b")
GROQ_COMPLETION_URL = "https://api.groq.com/v1/completions"

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not found. Set it in your environment or Streamlit secrets.")

@st.cache_resource(show_spinner=False)
def load_vectordb():
    if not os.path.exists(CHROMA_DIR):
        st.error("Chroma DB not found. Run index_builder.py first.")
        return None
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vectordb

vectordb = load_vectordb()
if vectordb is None:
    st.stop()

retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

PROMPT_PREFIX = (
    "You are an assistant. You MUST answer using ONLY the provided CONTEXT below. "
    "Do NOT use any outside knowledge. If the exact answer is not present in CONTEXT, reply exactly: \"I don't know.\" "
    "Keep the answer short and in the same language as the question. If multiple sections apply, synthesize them briefly.\n\n"
)

def build_prompt(context_chunks: List[str], question: str) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = PROMPT_PREFIX + "CONTEXT:\n" + context + "\n\nQUESTION:\n" + question + "\n\nANSWER:"
    return prompt

def call_groq(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    if not GROQ_API_KEY:
        return "Missing GROQ API key."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(GROQ_COMPLETION_URL, headers=headers, json=payload, timeout=60)
    except Exception as e:
        return f"Failed to call Groq API: {e}"

    if resp.status_code != 200:
        return f"Groq API error {resp.status_code}: {resp.text}"

    data = resp.json()
    text = None
    try:
        if "choices" in data and len(data["choices"]) > 0:
            ch0 = data["choices"][0]
            if isinstance(ch0, dict):
                text = ch0.get("text") or (ch0.get("message") and ch0["message"].get("content"))
        if not text:
            text = data.get("text") or data.get("completion") or data.get("result") or None
    except Exception:
        text = None

    if not text:
        return "Could not parse Groq response: " + json.dumps(data)[:500]

    return text.strip()

if prompt_text := st.chat_input("Ask about the Employee Handbook..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.chat_message("user").write(prompt_text)

    with st.spinner("Searching the handbook..."):
        docs = retriever.get_relevant_documents(prompt_text)
        snippets = [d.page_content.strip().replace("\n", " ")[:1000] for d in docs]

    if not snippets:
        answer = "I don't know."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
    else:
        prompt = build_prompt(snippets, prompt_text)

        try:
            user_lang = detect(prompt_text)
        except Exception:
            user_lang = "en"

        with st.spinner("Asking the model..."):
            llm_answer = call_groq(prompt, max_tokens=512, temperature=0.0)

        if llm_answer.strip() == "":
            llm_answer = "I don't know."

        st.session_state.messages.append({"role": "assistant", "content": llm_answer})
        st.chat_message("assistant").write(llm_answer)

        with st.expander("Sources and snippets"):
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "unknown")
                snippet = d.page_content.replace("\n", " ")[:500]
                st.write(f"{i}. **{src}** — {snippet}...")
