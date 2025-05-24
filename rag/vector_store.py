import chromadb
from chromadb.config import Settings

client = chromadb.Client(
    Settings(
        persist_directory="./chroma_store"
    )
)

collection = client.get_or_create_collection(name="fund_chunks")

def store_embeddings(embedded_chunks):
    for chunk in embedded_chunks:
        meta = chunk["metadata"]
        uid = f"{meta['fund_name'].replace(' ', '_')}_{meta['page_number']}_{meta['source_file']}"
        collection.add(
            documents=[chunk["text"]],
            embeddings=[chunk["embedding"]],
            metadatas=[meta],
            ids=[uid]
        )

def hybrid_search(query, fund_filter=None, top_k=15):
    where_clause = {"fund_name": fund_filter} if fund_filter else {}
    results = collection.query(
        query_texts=[query],
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
