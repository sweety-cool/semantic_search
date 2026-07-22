import re
import streamlit as st
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_PATH = "sonnets_only.pdf"  # match your actual filename

# ---------- Task 1: Text Extraction from PDF ----------
def extract_text(pdf_path):
    """Extracts all text from the given PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ---------- Task 2: Robust Line-by-Line Chunking (Sonnets) ----------
def chunk_sonnets(text):
    """Splits the raw text into individual sonnet chunks based on numbering."""
    lines = text.split('\n')
    chunks = []
    current_sonnet_lines = []

    for line in lines:
        clean_line = line.strip()

        # Skip empty lines or page numbers like "p. 1", "p. 25"
        if not clean_line or re.match(r'^p\.\s*\d+$', clean_line):
            continue

        # Trigger: Detect if the line is strictly a number from 1 to 154
        if clean_line.isdigit() and 1 <= int(clean_line) <= 154:
            # Save the previous sonnet if we have one
            if current_sonnet_lines:
                sonnet_text = "\n".join(current_sonnet_lines).strip()
                if len(sonnet_text) > 150:  # Ensures we don't save fragmented garbage
                    chunks.append(sonnet_text)

            # Start a new sonnet
            current_sonnet_lines = [f"Sonnet {clean_line}:"]

        else:
            # Add text to the current sonnet (this automatically ignores the intro
            # text before Sonnet 1 because current_sonnet_lines is empty)
            if current_sonnet_lines:
                current_sonnet_lines.append(clean_line)

    # Add the final sonnet (Sonnet 154), but chop off the Gutenberg license footer
    if current_sonnet_lines:
        final_text = "\n".join(current_sonnet_lines)
        final_text = final_text.split("THE END")[0].strip()
        if len(final_text) > 150:
            chunks.append(final_text)

    return chunks

# ---------- Task 3: Vector Similarity Calculation ----------
def cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- Task 4: Streamlit Caching & Model Loading ----------
@st.cache_resource
def load_everything():
    """Loads the model and prepares the embeddings."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    text = extract_text(PDF_PATH)
    chunks = chunk_sonnets(text)

    # Encode all sonnet chunks
    embeddings = model.encode(chunks, show_progress_bar=False)
    return model, chunks, embeddings

# ---------- Task 5: Semantic Search Logic ----------
def search(query, model, chunks, embeddings, top_k=3):
    """Performs semantic search to find the top_k most similar chunks to the query."""
    query_embedding = model.encode([query])[0]

    scores = []
    for i, chunk_embedding in enumerate(embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)
        scores.append((score, i))

    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:top_k]

# ---------- Task 6: The Actual UI ----------
st.set_page_config(page_title="Semantic Search", page_icon="🔍")
st.title("🔍 Semantic Search over Shakespeare Sonnets")
st.caption(f"Searching: {PDF_PATH}")

with st.spinner("Loading local ML model and indexing sonnets..."):
    model, chunks, embeddings = load_everything()

st.success(f"Ready — {len(chunks)} distinct sonnets indexed.")

query = st.text_input("Ask a question about the document:")
top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)

if query:
    # ---------- Task 7: Execute Search and Display Results ----------
    results = search(query, model, chunks, embeddings, top_k=top_k)
    st.subheader(f"Top {top_k} results")
    for score, idx in results:
        with st.container(border=True):
            st.markdown(f"**Similarity: `{score:.3f}`**")
            st.text(chunks[idx])