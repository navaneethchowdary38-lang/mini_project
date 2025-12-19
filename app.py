import streamlit as st
import fitz
import re
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Universal PDF Analyzer", layout="wide")
st.title("📄 Universal PDF Analyzer")

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
    outputs = model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------- SESSION STATE ----------------
if "paragraphs" not in st.session_state:
    st.session_state.paragraphs = []

if "index" not in st.session_state:
    st.session_state.index = None

# ---------------- TEXT RECONSTRUCTION ----------------
def reconstruct_paragraphs(lines):
    paragraphs = []
    buffer = ""

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # merge numbered lists
        if re.match(r"^\d+[\.\)]?$", line):
            buffer += " " + line
            continue

        # short broken lines → merge
        if len(line) < 40:
            buffer += " " + line
        else:
            if buffer:
                paragraphs.append(buffer.strip())
            buffer = line

    if buffer:
        paragraphs.append(buffer.strip())

    return paragraphs

# ---------------- LOAD PDF ----------------
uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf and st.button("Load PDF"):
    all_lines = []

    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")

    for page in doc:
        text = page.get_text("text")
        page_lines = text.split("\n")
        all_lines.extend(page_lines)

    paragraphs = reconstruct_paragraphs(all_lines)

    embeddings = embedder.encode(
        paragraphs,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    st.session_state.paragraphs = paragraphs
    st.session_state.index = index

    st.success(f"PDF loaded successfully. Indexed {len(paragraphs)} paragraphs.")

# ---------------- ANSWER QUERY ----------------
question = st.text_input("Ask a question from the PDF")

if st.button("Ask"):
    if st.session_state.index is None:
        st.warning("Please load a PDF first.")
        st.stop()

    # semantic search
    q_vec = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = st.session_state.index.search(q_vec, k=8)

    # expand context (important!)
    context_blocks = []
    for i in ids[0]:
        context_blocks.append(st.session_state.paragraphs[i])

    context = "\n".join(context_blocks)

    prompt = f"""
Answer the question strictly using the information from the context below.
If the answer is not present, say "Not found in the document".

Context:
{context}

Question:
{question}

Answer:
"""
    answer = run_llm(prompt)

    st.success("Answer from PDF:")
    st.write(answer)
