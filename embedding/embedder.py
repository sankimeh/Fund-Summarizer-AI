from sentence_transformers import SentenceTransformer

print("🔄 Loading BGE embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    return [
        {
            "text": text,
            "embedding": emb.tolist(),
            "metadata": chunk["metadata"]
        }
        for text, emb, chunk in zip(texts, embeddings, chunks)
    ]
