import hashlib
import re
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
except Exception:
    pass
collection = client.create_collection(name="fund_chunks")

# ---------- Helper Functions ----------
def sanitize_filename(name: str) -> str:
    # Replace non-alphanumeric characters with underscore
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)

def make_uid(text: str, meta: dict, idx: int) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    fund_name_s = sanitize_filename(meta.get("fund_name", "fund"))
    source_file_s = sanitize_filename(meta.get("source_file", "source"))
    page_number = meta.get("page_number", 0)
    return f"{fund_name_s}_{page_number}_{source_file_s}_{idx}_{h}"

# ---------- Financial Metadata Tagging ----------
def add_financial_metadata(chunk: dict) -> dict:
    text = chunk["text"].lower()
    financial_tags_patterns = {
        "fees": [r"management fee", r"expense", r"ter[\s:]", r"incentive", r"admin"],
        "liquidity": [r"redemption", r"liquidity", r"lock-?up", r"withdrawal"],
        "risk": [r"risk", r"volatility", r"default", r"loss", r"cyber"],
        "performance": [r"return", r"yield", r"irr", r"benchmark", r"performance"],
        "strategy": [r"investment strategy", r"asset allocation", r"portfolio"],
        "structure": [r"legal structure", r"\blp\b", r"\bllc\b", r"trust"],
        "redemption_terms": [r"redemption", r"lock-?in", r"withdrawal fee", r"early withdrawal"]
    }

    matched_tags = []
    for tag, patterns in financial_tags_patterns.items():
        if any(re.search(p, text) for p in patterns):
            matched_tags.append(tag)

    # Serialize tags as comma-separated string for ChromaDB storage
    chunk["metadata"]["financial_tags"] = ", ".join(matched_tags) if matched_tags else "general"
    return chunk

# ---------- Store Embeddings to ChromaDB ----------
def store_embeddings(chunks: list):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    metadatas = []
    ids = []
    docs = []
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk = add_financial_metadata(chunk)
        meta = chunk["metadata"]

        if isinstance(meta.get("financial_tags"), list):
            meta["financial_tags"] = ", ".join(meta["financial_tags"])

        uid = make_uid(chunk["text"], meta, idx)

        metadatas.append(meta)
        ids.append(uid)
        docs.append(chunk["text"])

    collection.add(
        documents=docs,
        embeddings=[emb.tolist() for emb in embeddings],
        metadatas=metadatas,
        ids=ids,
    )

# ---------- Hybrid Search with Tag Filters ----------
def hybrid_search(query: str, fund_filter: str = None, financial_tag_filter: str = None, top_k: int = 15):
    where_clause = {}
    if fund_filter:
        where_clause["fund_name"] = fund_filter
    if financial_tag_filter:
        # Since tags stored as string, use substring contains filter
        where_clause["financial_tags"] = {"$contains": financial_tag_filter}

    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_clause if where_clause else None
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

# ---------- Reranker with Metadata-Aware Boost ----------
print("🔄 Loading reranker model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large").to(device)
model.eval()

def rerank(query: str, candidates: list, top_n: int = 5) -> list:
    pairs = [(query, c["text"]) for c in candidates]
    inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

    with torch.no_grad():
        raw_scores = model(**inputs).logits.squeeze(-1)

    scores = raw_scores.tolist()
    query_lower = query.lower()
    boosted_scores = []

    for score, c in zip(scores, candidates):
        tags_str = c["metadata"].get("financial_tags", "")
        tags = [tag.strip() for tag in tags_str.split(",")] if tags_str else []
        # Boost if any financial tag is present in query text
        boost = 0.1 if any(tag in query_lower for tag in tags) else 0
        boosted_scores.append(score + boost)

    min_score, max_score = min(boosted_scores), max(boosted_scores)
    norm_scores = [(s - min_score) / (max_score - min_score) if max_score != min_score else 1.0 for s in boosted_scores]

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

# ---------- Combined Search ----------
def search_with_rerank(
    query: str,
    fund_filter: str = None,
    financial_tag_filter: str = None,
    top_k: int = 15,
    rerank_top_n: int = 5
):
    raw_results = hybrid_search(query, fund_filter=fund_filter, financial_tag_filter=financial_tag_filter, top_k=top_k)
    return rerank(query, raw_results, top_n=rerank_top_n)
