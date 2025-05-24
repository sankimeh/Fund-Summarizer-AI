import os
from embedding.embedder import embed_chunks
from fund_summarizer import process_files
from llm.fund_comparator import compare_funds_with_reranked_results
from rag.vector_store import store_embeddings, hybrid_search
from reranker.reranker import rerank

# File paths relative to the current script
base_path = "documents/BlackRock Private Credit Fund"
files = [
    os.path.join(base_path, "blk-credit-strategies-fund-investor-guide.pdf"),
    os.path.join(base_path, "offshore1_4-1.pdf"),
    os.path.join(base_path, "pro-privatecreditfundalt-bdebt.pdf"),
]

fund_name = "BlackRock Private Credit Fund"

print("📁 Processing...")
chunks = process_files(fund_name, files)
print(f" - Number of chunks processed: {len(chunks)}")
if not chunks:
    raise RuntimeError("No chunks were processed from the documents!")

print("🔢 Embedding...")
embedded = embed_chunks(chunks)
print(f" - Number of embeddings generated: {len(embedded)}")
if not embedded:
    raise RuntimeError("Embedding generation failed or returned empty!")

print("💾 Storing embeddings...")
store_embeddings(embedded)
print(" - Embeddings stored successfully.")

aspect = "Investment Objective"
query = aspect  # Keeping it aligned

print(f"\n🔍 Searching for: {query}")
results = hybrid_search(query, fund_filter=fund_name)
print(f" - Number of search results: {len(results)}")

print("🔁 Reranking search results...")
top = rerank(query, results)
print(f" - Number of top reranked chunks: {len(top)}")

funds = [fund_name]

print(f"\n📊 LLM Comparison on aspect: {aspect}")
comparison = compare_funds_with_reranked_results(fund_names=funds, aspect=aspect, top_chunks=top)

print("\n✅ Answer:")
print(comparison.get("answer", "No answer returned."))

print("\n📚 Sources:")
for s in comparison.get("sources", []):
    print(f"- {s['source_file']} (Page {s['page_number']}) | Score: {s['score']:.4f}")
    print(f"  {s['text_excerpt']}...\n")

