"""
Semantic Search Engine — Project 1
Learn: text extraction -> chunking -> embeddings -> similarity search





Think of it like a map, not numbers

Imagine you have a giant map of a country. Cities that are culturally or economically similar tend to get placed near each other, cities that are totally different are far apart. That's basically what's happening — except instead of a 2D map (latitude, longitude), the "map" for text has 384 dimensions.

You obviously can't picture 384 dimensions in your head — nobody can, it's not a spatial thing you visualize. Just think of it as: "way more directions than a normal map, so it can capture way more shades of meaning than just north/south/east/west."

What actually happens to your text

Each chunk of your PDF (like that "transparency and corporate communication" paragraph) gets converted into 384 numbers — like GPS coordinates, but for meaning instead of location.

"The blog is strategically aimed at communication..." → [0.12, -0.45, 0.88, ... ] (384 numbers)

That list of numbers is the chunk's "location" on the meaning-map.

Why this is useful

Two chunks that talk about similar ideas — even if they use completely different words — will land in nearby "locations" on this map.

Example: if one chunk of your paper says "employees are given freedom in how they blog" and another says "bloggers report autonomy over their content" — those use zero overlapping keywords, but they mean almost the same thing. A regular keyword search (like Ctrl+F) would completely miss the connection. But on the meaning-map, those two chunks will land close together, because the model understood they're about the same concept.

That's the entire value of semantic search over plain keyword search.
 
Run:
    python3 main.py
"""

# ---------- STEP 1: Extract text from PDF ----------




from pypdf import PdfReader

reader = PdfReader("my_document.pdf")   # or my_docment.pdf, matching your real filename
text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

print(f"Extracted {len(text)} characters")
print(text[:500])


##########  step2: chunking the text


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end= start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

chunks = chunk_text(text)
print(f"\nCreated {len(chunks)} chunks")
print("\n--- Sample chunk (index 10) ---")
print(chunks[10])
print("\n--- Sample chunk (index 50) ---")
print(chunks[50])

# ---------- STEP 3: Create embeddings chunk ----------


from sentence_transformers import SentenceTransformer

print("\nLoading model (first run download ~80MB)")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("\n Embedding all chunks....")
embeddings = model.encode(chunks, show_progress_bar=True)
print(f"\nCreated {len(embeddings)} embeddings")
print("\n--- Sample embedding (index 10) ---")
print(embeddings[10])
print("\n--- Sample embedding (index 50) ---")
print(embeddings[50])

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"That means: {embeddings.shape[0]} chunks, each turned into a {embeddings.shape[1]}-dimensional vector")



# ---------- STEP 4a: Manual cosine similarity (DO THIS FIRST — understand the math) ----------
import numpy as np

def cosine_similarity(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, top_k=3):
    #STEPA: Turn the question into the same 384-number fingerprint, 384 is columns in the embeddings matrix and 148 is rows in the embeddings matrix
    query_embedding = model.encode([query]) [0] 

    # this takes your question and runs it through the exact same model that made the chunk fingerprints. 
    # That's critical: your question and your chunks have to be measured with the same ruler, or the comparison is meaningless.

    #StepB: Compare it against each chunks embeddings one by one, cosine similarity is the dot product of the two vectors divided by the product of the two vectors' magnitudes

    scores= []
    for i,chunk_embedding in enumerate(embeddings): # this is the enumerate function. It's a built-in Python function that gives you both the index and the value of each item in a list at the same time.
        score= cosine_similarity(query_embedding, chunk_embedding)
        scores.append((score, i))

        #this is the "how close are two fingerprints" math. It outputs a number between -1 and 1:

        #1.0 = basically identical meaning
        #0.0 = unrelated
        #negative = opposite meaning (rare in practice for this kind of text)

    #StepC: Sort the chunks by similarity score, highest to lowest
    scores.sort(reverse=True)

    #StepD: Show the top results
    print("\n--- Top search results ---")

    for score,idx in scores[:top_k]: # this is the top k results . top_k just means "how many results do you want back. 
        #The k is a placeholder variable name (common in programming/math), standing in for any number you choose. It's not a fixed thing — it's a setting.
        print(f"[similarity: {score: .4f}] {chunks[idx]}") # this is the similarity score and the chunk text
        print("-"*60) # this is just a separator to make the output easier to read
        print(f"\nTop search results for: '{query}'")
    

search("what is the main idea of the document?", top_k=2) # this is the question we are searching for
search("why do employees feel restricted when blogging for their company")
search("what are the four discursive positions of bloggers identified in this study")
search("Who's the CUURRENT CEO of Meta?")
search("how was the data collected for this case study")