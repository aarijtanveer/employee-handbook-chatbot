# app.py
import os
import json
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langdetect import detect
import requests
from typing import List

st.set_page_config(page_title="Employee Handbook Assistant", page_icon="🧾")
st.title("Employee Handbook Assistant")
st.write("Ask me about HR policies from the Employee Handbook. You can use English, Roman Urdu, or Urdu.")

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 4

# Groq API settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_COMPLETION_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY not found. Add it in your environment or Streamlit secrets.")

@st.cache_resource(show_spinner=False)
def load_vectordb():
    if not os.path.exists(CHROMA_DIR):
        st.error("Chroma DB not found. Run index_builder.py first.")
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
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
    "You are an HR assistant. You MUST answer only from the provided CONTEXT below. "
    "Do NOT use outside knowledge. If the exact answer is not in CONTEXT, reply: 'I don't know.' "
    "Keep answers concise and in the same language as the question. "
    "If the user asks a follow-up, use chat history to understand context.\n\n"
)

def build_prompt(context_chunks: List[str], question: str, history: List[dict]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history if msg["role"] != "system"])
    prompt = (
        PROMPT_PREFIX
        + "CONTEXT:\n" + context
        + "\n\nCHAT HISTORY:\n" + history_text
        + "\n\nQUESTION:\n" + question
        + "\n\nANSWER:"
    )
    return prompt

def call_groq(messages: List[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
    if not GROQ_API_KEY:
        return "Missing GROQ API key."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
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
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Could not parse Groq response: " + json.dumps(data)[:500]

def rephrase_question(question: str, history: List[dict]) -> str:
    """Ask Groq to rewrite the follow-up question into a standalone one."""
    system_msg = {"role": "system", "content": "You are a rephraser. Rewrite the latest question into a standalone, complete question using the chat history for context."}
    history_msgs = [{"role": msg["role"], "content": msg["content"]} for msg in history]
    user_msg = {"role": "user", "content": question}
    messages = [system_msg] + history_msgs + [user_msg]
    return call_groq(messages, max_tokens=100, temperature=0.0)

# Chat UI
if prompt_text := st.chat_input("Ask about the Employee Handbook..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.chat_message("user").write(prompt_text)

    with st.spinner("Thinking..."):
        # Rephrase if context exists
        if len(st.session_state.messages) > 1:
            standalone_question = rephrase_question(prompt_text, st.session_state.messages[:-1])
        else:
            standalone_question = prompt_text

        docs = retriever.get_relevant_documents(standalone_question)
        snippets = [d.page_content.strip().replace("\n", " ")[:1000] for d in docs]

    if not snippets:
        answer = "I don't know."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
    else:
        prompt = build_prompt(snippets, standalone_question, st.session_state.messages[:-1])

        with st.spinner("Answering..."):
            llm_answer = call_groq(
                [{"role": "system", "content": "You are an HR assistant."},
                 {"role": "user", "content": prompt}]
            )

        if not llm_answer.strip():
            llm_answer = "I don't know."

        st.session_state.messages.append({"role": "assistant", "content": llm_answer})
        st.chat_message("assistant").write(llm_answer)

        with st.expander("Sources and snippets"):
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "unknown")
                snippet = d.page_content.replace("\n", " ")[:500]
                st.write(f"{i}. **{src}** — {snippet}...")
