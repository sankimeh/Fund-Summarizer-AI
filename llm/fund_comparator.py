import json
import re
import requests
from typing import List, Dict

# ---------------------- PROMPT BUILDERS ----------------------
def build_prompt(fund_names: List[str], aspect: str) -> str:
    return build_single_fund_prompt(fund_names[0], aspect) if len(fund_names) == 1 else build_multi_fund_prompt(fund_names, aspect)

def build_single_fund_prompt(fund_name: str, aspect: str) -> str:
    return f"""
You are a financial due diligence assistant AI helping an investment advisor review private fund materials.

Your task is to extract **precise, well-supported information** related to the following question about the fund "{fund_name}":

\u27a1\ufe0f **Question:** {aspect}

\ud83d\udccc Instructions:
- Use **only the provided context** (below).
- Be specific and concise (2–4 lines).
- If the answer is not directly mentioned, try to infer it from related sections such as investment strategy, risk factors, fees, or legal terms.
- Clearly state **if information is not available**.
- For every fact, **cite the source document name and page number**, like: _[Investor Guide, p. 5]_.

\ud83c\udfaf Output format:
1. **Answer**: <your concise answer>
2. **Citations**: [Document Name, Page #]

Stay factual. Do not fabricate details.
"""

def build_multi_fund_prompt(fund_names: List[str], aspect: str) -> str:
    fund_list = "\n".join([f"- {name}" for name in fund_names])
    return f"""
You are a financial assistant AI helping an advisor compare multiple funds.

Compare the following funds based on the aspect: **"{aspect}"**:

{fund_list}

\ud83d\udccc Instructions:
- Use **only the provided context**.
- If the data is missing for a fund, return "Not available".
- Return a comparison table. Each row should be one fund, and each column should be:
  - Value (answer)
  - Citation (document and page)

Example:

| Fund Name | Value | Citation |
|-----------|-------|----------|
| Fund A    | 2% annual fee | [FundA_PPT, Slide 12] |
| Fund B    | Not available | — |

Do **not** fabricate any values. Stay factual and grounded in the source.
"""

# ---------------------- CONTEXT FORMATTER ----------------------
def format_context(chunks: List[Dict]) -> str:
    return "\n\n".join(
        f"[{chunk['metadata'].get('source_file', 'unknown')}, Page {chunk['metadata'].get('page_number', '?')}]\n{chunk['text']}"
        for chunk in chunks
    )

# ---------------------- CHUNK FILTER ----------------------
def filter_context_chunks(chunks: List[Dict], keywords: List[str]) -> List[Dict]:
    return [
        chunk for chunk in chunks
        if any(kw.lower() in chunk['text'].lower() for kw in keywords)
    ]

# ---------------------- LLM QUERY ----------------------
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
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": True,
                "temperature": 0.0,
                "top_p": 1.0,
                "repeat_penalty": 1.0
            },
            stream=True,
            timeout=60,

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
        return f"\u274c LLM request failed: {e}"

# ---------------------- CITATION VALIDATOR ----------------------
def extract_citations(text: str) -> List[str]:
    return re.findall(r'\[.*?,\s*p\.\s*\d+\]', text)

def validate_llm_output(answer: str, sources: List[Dict]) -> Dict:
    issues = []
    if "not available" in answer.lower() and any("text_excerpt" in s and len(s["text_excerpt"]) > 100 for s in sources):
        issues.append("Potential mismatch: answer says 'not available' but sources are substantive.")
    if not extract_citations(answer):
        issues.append("Answer missing citation structure.")

    return {"validated": len(issues) == 0, "issues": issues}

# ---------------------- MASTER FUNCTION ----------------------
def compare_funds_with_reranked_results(
    fund_names: List[str],
    aspect: str,
    top_chunks: List[Dict]
) -> Dict:
    prompt = build_prompt(fund_names, aspect)
    filtered_chunks = filter_context_chunks(top_chunks, [aspect] + ["fees", "risk", "strategy", "structure"])
    context = format_context(filtered_chunks or top_chunks)
    answer = query_llm_with_context(prompt, context)

    sources = [
        {
            "source_file": c["metadata"].get("source_file", "unknown"),
            "page_number": c["metadata"].get("page_number", "N/A"),
            "score": round(c.get("score", 0), 4),
            "text_excerpt": c["text"][:200].strip()
        }
        for c in filtered_chunks or top_chunks
    ]

    validation = validate_llm_output(answer, sources)

    return {
        "answer": answer,
        "sources": sources,
        "validation": validation
    }