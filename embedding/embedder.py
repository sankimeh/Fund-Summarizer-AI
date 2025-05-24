from sentence_transformers import SentenceTransformer

print("🔄 Loading BGE embedding model...")
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")  # 1024-dim embeddings

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
