import hashlib
import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- Embedding Model (1024-dim BGE) ----------
print("🔄 Loading BGE embedding model (1024-dim)...")
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")  # 1024-dim embeddings

# ---------- ChromaDB Setup ----------
print("📦 Setting up ChromaDB...")
client = chromadb.Client(Settings(persist_directory="./chroma_store"))

# Delete and recreate collection to match new 1024-dim embeddings (only run once)
try:
    client.delete_collection("fund_chunks")
except:
    pass
collection = client.create_collection(name="fund_chunks")

def make_uid(text, meta):
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{meta['fund_name'].replace(' ', '_')}_{meta['page_number']}_{meta['source_file']}_{h}"

def store_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    for chunk, emb in zip(chunks, embeddings):
        meta = chunk["metadata"]
        uid = make_uid(chunk["text"], meta)
        collection.add(
            documents=[chunk["text"]],
            embeddings=[emb.tolist()],
            metadatas=[meta],
            ids=[uid]
        )

def hybrid_search(query, fund_filter=None, top_k=15):
    where_clause = {"fund_name": fund_filter} if fund_filter else {}

    # 🔹 Encode query with same model (1024-dim)
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_clause
    )
    return [
        {
            "text": doc,
            "score": score,
            "metadata": metadata
        }
        for doc, score, metadata in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        )
    ]

# ---------- Reranker ----------
print("🔄 Loading reranker model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large").to(device)
model.eval()

def rerank(query, candidates, top_n=5):
    pairs = [(query, c["text"]) for c in candidates]
    inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

    with torch.no_grad():
        raw_scores = model(**inputs).logits.squeeze(-1)

    scores = raw_scores.tolist()
    min_score, max_score = min(scores), max(scores)
    norm_scores = [(s - min_score) / (max_score - min_score) if max_score != min_score else 1.0 for s in scores]

    ranked = sorted(zip(norm_scores, candidates), key=lambda x: x[0], reverse=True)
    return [
        {
            **c,
            "metadata": {
                **c.get("metadata", {}),
                "rerank_score": float(s)
            }
        }
        for s, c in ranked[:top_n]
    ]

# ---------- Combined Search Pipeline ----------
def search_with_rerank(query, fund_filter=None, top_k=15, rerank_top_n=5):
    raw_results = hybrid_search(query, fund_filter=fund_filter, top_k=top_k)
    return rerank(query, raw_results, top_n=rerank_top_n)
