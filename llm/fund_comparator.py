# fund_comparator.py

import json
import requests
from typing import List, Dict


def build_prompt(fund_names: List[str], aspect: str) -> str:
    if len(fund_names) == 1:
        return build_single_fund_prompt(fund_names[0], aspect)
    return build_multi_fund_prompt(fund_names, aspect)


def build_single_fund_prompt(fund_name: str, aspect: str) -> str:
    return f"""
You are a financial assistant AI. Extract the "{aspect}" for the fund "{fund_name}" based on the context provided.

If the exact investment objective is not directly stated, infer it from related sections such as strategies, investment approach, or fund summary.

Provide a short, precise summary of the objective and cite the document name and page number where applicable.
"""


def build_multi_fund_prompt(fund_names: List[str], aspect: str) -> str:
    fund_list = "\n".join([f"- {name}" for name in fund_names])
    return f"""
Compare the following funds on the aspect: "{aspect}":
{fund_list}

Use only the information provided in the context.
Present the comparison in a table format with the following rows:
- Entry Load
- Exit Load
- Annual Expense Ratio
- Performance Fee

Include citations for each value (document name and page/slide number).
If data is missing for any fund, leave the value blank or indicate "Not available".
"""


def format_context(chunks: List[Dict]) -> str:
    return "\n\n".join(
        f"[{chunk['metadata'].get('source_file', 'unknown')}, Page {chunk['metadata'].get('page_number', '?')}]\n{chunk['text']}"
        for chunk in chunks
    )


def query_llm_with_context(prompt: str, context: str, model: str = "llama3.2:latest") -> str:
    full_prompt = f"""
You are a financial assistant AI. Use only the context below to answer the question.

### CONTEXT
{context}

### QUESTION
{prompt}
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": True},
            stream=True,
            timeout=60
        )

        full_output = ""
        for chunk in response.iter_lines():
            if chunk:
                try:
                    data = json.loads(chunk.decode("utf-8"))
                    full_output += data.get("response", "")
                except json.JSONDecodeError:
                    continue
        return full_output.strip()

    except requests.RequestException as e:
        return f"❌ LLM request failed: {e}"


def compare_funds_with_reranked_results(
    fund_names: List[str],
    aspect: str,
    top_chunks: List[Dict]
) -> Dict:
    prompt = build_prompt(fund_names, aspect)
    context = format_context(top_chunks)
    answer = query_llm_with_context(prompt, context)

    sources = [
        {
            "source_file": c["metadata"].get("source_file", "unknown"),
            "page_number": c["metadata"].get("page_number", "N/A"),
            "score": round(c.get("score", 0), 4),
            "text_excerpt": c["text"][:200].strip()
        }
        for c in top_chunks
    ]

    return {
        "answer": answer,
        "sources": sources
    }
