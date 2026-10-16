"""RAG Document Intelligence Platform
Upload PDFs/TXT files, index them with vector embeddings, and ask questions.

Stack: Streamlit + LangChain + HuggingFace embeddings + FAISS + Groq / Ollama.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")


# ---------- Helpers ----------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_documents(files):
    """Convert uploaded PDF/TXT files into LangChain Documents."""
    docs = []
    for f in files:
        name = f.name
        if name.lower().endswith(".pdf"):
            reader = PdfReader(f)
            for page_no, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        Document(page_content=text, metadata={"source": name, "page": page_no})
                    )
        else:
            text = f.read().decode("utf-8", errors="ignore")
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": name, "page": 1}))
    return docs


def build_vector_store(files, chunk_size, chunk_overlap):
    docs = load_documents(files)
    if not docs:
        return None, 0
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, get_embeddings())
    return store, len(chunks)


def get_llm(provider, model, groq_key):
    if provider == "Groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, api_key=groq_key, temperature=0)
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=0)


def answer_question(question, store, llm, top_k):
    retrieved = store.similarity_search(question, k=top_k)
    context = "\n\n".join(
        f"[{d.metadata['source']} - page {d.metadata['page']}]\n{d.page_content}"
        for d in retrieved
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content, retrieved


# ---------- Session state ----------
st.session_state.setdefault("store", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("indexed_info", "")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")

    provider = st.radio("LLM provider", ["Groq", "Ollama (local)"])
    provider_key = "Groq" if provider == "Groq" else "Ollama"

    if provider_key == "Groq":
        model = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
        groq_key = st.text_input(
            "Groq API key", type="password", value=os.getenv("GROQ_API_KEY", "")
        )
    else:
        model = st.text_input("Ollama model", value="llama3.2")
        groq_key = ""
        st.caption("Make sure Ollama is running and the model is pulled: `ollama pull llama3.2`")

    st.divider()
    chunk_size = st.slider("Chunk size", 200, 2000, 800, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 100, step=50)
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 4)

    st.divider()
    uploaded = st.file_uploader(
        "Upload documents", type=["pdf", "txt"], accept_multiple_files=True
    )
    if st.button("Build index", type="primary", use_container_width=True):
        if not uploaded:
            st.warning("Upload at least one file first.")
        else:
            with st.spinner("Chunking and embedding..."):
                store, n_chunks = build_vector_store(uploaded, chunk_size, chunk_overlap)
            if store is None:
                st.error("No readable text found (scanned PDFs need OCR).")
            else:
                st.session_state.store = store
                st.session_state.messages = []
                st.session_state.indexed_info = f"{len(uploaded)} file(s), {n_chunks} chunks"
                st.success("Index ready.")

    if st.session_state.indexed_info:
        st.caption(f"Indexed: {st.session_state.indexed_info}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

# ---------- Main ----------
st.title("📄 RAG Document Q&A")
st.caption("Upload documents, build the index, then ask questions grounded in your files.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']} (page {s['page']})**")
                    st.caption(s["text"])

question = st.chat_input("Ask a question about your documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if st.session_state.store is None:
            reply = "Please upload documents and click **Build index** first."
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        elif provider_key == "Groq" and not groq_key:
            reply = "Please enter your Groq API key in the sidebar."
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            try:
                with st.spinner("Thinking..."):
                    llm = get_llm(provider_key, model, groq_key)
                    answer, retrieved = answer_question(
                        question, st.session_state.store, llm, top_k
                    )
                st.markdown(answer)
                sources = [
                    {
                        "source": d.metadata["source"],
                        "page": d.metadata["page"],
                        "text": d.page_content[:400] + "...",
                    }
                    for d in retrieved
                ]
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"**{s['source']} (page {s['page']})**")
                        st.caption(s["text"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                st.error(f"Error: {e}")