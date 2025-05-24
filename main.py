import os
import json
from embedding.embedder import embed_chunks
from fund_summarizer import process_files
from llm.fund_comparator import compare_funds_with_reranked_results
from rag.vector_store import store_embeddings, hybrid_search
from reranker.reranker import rerank

# Define your fund and files
base_path = "documents/BlackRock Private Credit Fund"
files = [
    os.path.join(base_path, "blk-credit-strategies-fund-investor-guide.pdf"),
    os.path.join(base_path, "offshore1_4-1.pdf"),
    os.path.join(base_path, "pro-privatecreditfundalt-bdebt.pdf"),
]

fund_name = "BlackRock Private Credit Fund"

# Full pipeline
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

# Questions to be answered
fund_questions = [
    "What is the fund's investment objective?",
    "What is the underlying investment strategy?",
    "How does the fund differentiate itself from its competitors?",
    "What is the benchmark used for performance comparison?",
    "What is the legal structure of the fund?",
    "What are the key risk factors associated with this fund?",
    "What is the fund's liquidity profile?",
    "What is the minimum investment required?",
    "What is the total expense ratio (TER) or management fee structure?",
    "Are there any performance based fees?",
    "Are there any hidden costs, such as trading or administrative fees?",
    "What are the expected returns based on historical data or backtracking?",
    "How does the fund perform in different market conditions?",
    "What is the expected volatility and drawdown potential?",
    "What are the redemption terms, including lock in period and withdrawal fees?",
    "Is there an option for early withdrawal, and what are the underlying conditions?",
    "How are proceeds distributed in case of fund liquidation?"
]

results = []

for question in fund_questions:
    print(f"\n🔍 Searching for: {question}")
    search_results = hybrid_search(question, fund_filter=fund_name)
    print(f" - Found {len(search_results)} results.")

    if not search_results:
        print("⚠️  No chunks found for this question.")
        results.append({
            "question": question,
            "answer": "No relevant information found in the provided documents.",
            "sources": []
        })
        continue

    print("🔁 Reranking...")
    top_chunks = rerank(question, search_results)
    print(f" - Top {len(top_chunks)} chunks selected.")

    print(f"📊 LLM answering: {question}")
    response = compare_funds_with_reranked_results(
        fund_names=[fund_name],
        aspect=question,
        top_chunks=top_chunks
    )

    print("\n✅ Answer:")
    print(response.get("answer", "No answer returned."))

    print("\n📚 Sources:")
    for s in response.get("sources", []):
        print(f"- {s['source_file']} (Page {s['page_number']}) | Score: {s['score']:.4f}")
        print(f"  {s['text_excerpt']}...\n")

    results.append({
        "question": question,
        "answer": response["answer"],
        "sources": response["sources"]
    })

# Save all results to JSON
output_file = f"{fund_name.replace(' ', '_').lower()}_qa_summary.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n📁 All results saved to: {output_file}")
