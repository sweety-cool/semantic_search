import streamlit as st
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_PATH = "my_document.pdf"  # match your actual filename
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ---------- Same functions from main.py ----------
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------- Streamlit caching: this is the key piece ----------
# @st.cache_resource tells Streamlit: "run this function ONCE, then
# reuse the result for every user interaction, instead of re-running
# it on every button click." Without this, the PDF would be re-read
# and re-embedded every single time someone searches.
@st.cache_resource
def load_everything():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    text = extract_text(PDF_PATH)
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = model.encode(chunks, show_progress_bar=False)
    return model, chunks, embeddings


def search(query, model, chunks, embeddings, top_k=3):
    query_embedding = model.encode([query])[0]
    scores = []
    for i, chunk_embedding in enumerate(embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)
        scores.append((score, i))
    scores.sort(reverse=True)
    return scores[:top_k]


# ---------- The actual UI ----------
st.set_page_config(page_title="Semantic Search", page_icon="🔍")
st.title("🔍 Semantic Search over your PDF")
st.caption(f"Searching: {PDF_PATH}")

# This line runs load_everything() — but thanks to @st.cache_resource,
# the heavy lifting (PDF read + embedding) only actually happens once,
# the first time the app starts.
with st.spinner("Loading model and embedding document (first load only)..."):
    model, chunks, embeddings = load_everything()

st.success(f"Ready — {len(chunks)} chunks indexed.")

query = st.text_input("Ask a question about the document:")
top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)

if query:
    results = search(query, model, chunks, embeddings, top_k=top_k)
    st.subheader(f"Top {top_k} results")
    for score, idx in results:
        with st.container(border=True):
            st.markdown(f"**Similarity: `{score:.3f}`**")
            st.write(chunks[idx])