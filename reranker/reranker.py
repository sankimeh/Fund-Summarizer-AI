from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")
model.eval()


def rerank(query, candidates, top_n=5):
    pairs = [(query, c["text"]) for c in candidates]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        raw_scores = model(**inputs).logits.squeeze(-1)

    # Normalize scores to [0, 1]
    scores = raw_scores.tolist()
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        norm_scores = [1.0 for _ in scores]
    else:
        norm_scores = [(s - min_score) / (max_score - min_score) for s in scores]

    # Attach normalized scores to candidates
    ranked = sorted(zip(norm_scores, candidates), key=lambda x: x[0], reverse=True)
    return [dict(c, score=float(s)) for s, c in ranked[:top_n]]
