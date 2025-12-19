import streamlit as st
import fitz
import re
import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PDF Analyzer", layout="wide")
st.title("📄 PDF Analyzer")

# ---------------- MODELS ----------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return embedder, tokenizer, model

embedder, tokenizer, model = load_models()

def run_llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------- SESSION STATE ----------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

# ---------------- LOAD PDF ----------------
uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf and st.button("Load PDF"):
    chunks = []
    raw_text = ""

    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")

    for page in doc:
        text = page.get_text("text")
        text = re.sub(r"\s+", " ", text)
        raw_text += text + " "

        # aggressive chunking
        for chunk in re.split(r"(?<=\.) ", text):
            if len(chunk) > 30:
                chunks.append(chunk.strip())

    vectors = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.raw_text = raw_text

    st.success("PDF loaded and indexed successfully.")

# ---------------- ANSWER QUERY ----------------
question = st.text_input("Ask a question")

if st.button("Ask"):
    if st.session_state.index is None:
        st.warning("Load a PDF first.")
        st.stop()

    q_lower = question.lower()

    # ✅ 1. KEYWORD SEARCH FIRST (CRITICAL FIX)
    if "course outcome" in q_lower or "course outcomes" in q_lower:
        matches = [
            c for c in st.session_state.chunks
            if "outcome" in c.lower()
        ]

        if matches:
            st.success("Answer from PDF (keyword match):")
            st.write(" ".join(matches[:5]))
            st.stop()

    # ✅ 2. SEMANTIC SEARCH
    q_vec = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, idx = st.session_state.index.search(q_vec, k=6)
    context = " ".join(st.session_state.chunks[i] for i in idx[0])

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
    answer = run_llm(prompt)
    st.success("Answer:")
    st.write(answer)
