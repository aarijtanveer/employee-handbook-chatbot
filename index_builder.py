# index_builder.py
import os
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Paths
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

def load_documents():
    docs = []
    print(f"Looking for files in: {os.path.abspath(DATA_DIR)}")
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.lower().endswith(".pdf"):
            print(f"Loaded: {file}")
            loader = PyPDFium2Loader(path)
            docs.extend(loader.load())
    return docs

def build_index():
    docs = load_documents()
    if not docs:
        print("No documents loaded. Put your handbook in the data/ folder.")
        return
    
    print(f"Loaded {len(docs)} documents. Chunking...")

    # Split into chunks of ~1000 characters
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # Use sentence-transformer embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Build Chroma vector database
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()
    print(f"✅ Index built and saved to {CHROMA_DIR}")

if __name__ == "__main__":
    build_index()
