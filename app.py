%%writefile app.py
import streamlit as st
import fitz
import re
import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PDF Analyzer",
    layout="wide"
)

st.title("📄 PDF Analyzer (Streamlit)")
st.write("Ask questions directly from your PDF")

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
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------- SESSION STATE ----------------
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "full_text" not in st.session_state:
    st.session_state.full_text = ""

# ---------------- PDF LOADING ----------------
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf and st.button("Load PDF"):
    pdf_chunks = []
    full_text = ""

    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")

    for page in doc:
        text = page.get_text()
        if text:
            cleaned = re.sub(r"\s+", " ", text).strip()
            full_text += cleaned + " "

            paragraphs = re.split(r"\n{2,}", cleaned)
            for p in paragraphs:
                if len(p.strip()) > 200:
                    pdf_chunks.append(p.strip())

    if not pdf_chunks:
        st.error("No readable text found in PDF")
    else:
        vectors = embedder.encode(
            pdf_chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        st.session_state.pdf_chunks = pdf_chunks
        st.session_state.faiss_index = index
        st.session_state.full_text = full_text

        st.success(f"PDF loaded successfully. {len(pdf_chunks)} chunks indexed.")

# ---------------- QUESTION ANSWERING ----------------
st.divider()
question = st.text_input("Ask a question from the PDF")

if st.button("Ask Question"):
    if st.session_state.faiss_index is None:
        st.warning("Please load a PDF first.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        q_vec = embedder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, idx = st.session_state.faiss_index.search(q_vec, k=8)
        confidence = float(scores[0][0])

        context = " ".join(
            st.session_state.pdf_chunks[i] for i in idx[0]
        )

        if confidence > 0.12:
            prompt = f"""
You MUST answer strictly using the PDF content below.
If the answer is not found, say "Not found in the document".

PDF Content:
{context}

Question:
{question}

Answer:
"""
            answer = run_llm(prompt)
            st.success("Answer from PDF:")
            st.write(answer)
        else:
            st.warning("Low confidence from PDF. General answer:")
            answer = run_llm(f"Answer concisely: {question}")
            st.write(answer)
