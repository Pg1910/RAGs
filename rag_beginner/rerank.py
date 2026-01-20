from sentence_transformers import CrossEncoder

# load once (this will download on first run)
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker
def rerank(query: str , candidates : list[dict], top_k: int = 5) -> list[dict]:
    """
    candidates :list of dicts containing at least text 
    returns : some dicts woth added 'rerank_score' , sorted desc.
    """

    reranker = get_reranker()
    pairs = [(query , c["text"])for c in candidates]
    scores = reranker.predict(pairs)

    for c , s in zip (candidates , scores):
        c["rerank_score"] = float(s)

    ranked = sorted (candidates , key = lambda x: x["rerank_score"], reverse = True )
    return ranked[:top_k]

if __name__ == "__main__":
    # simple test
    query = "What is FAISS used for in RAG?"
    candidates = [
        {"text": "FAISS is a library for efficient similarity search and clustering of dense vectors.", "source": "doc1", "global_id": 1},
        {"text": "FAISS is used in RAG to perform fast nearest neighbor search.", "source": "doc2", "global_id": 2},
        {"text": "FAISS stands for Facebook AI Similarity Search.", "source": "doc3", "global_id": 3},
    ]
    ranked = rerank(query, candidates, top_k=2)
    for r in ranked:
        print(f"Score: {r['rerank_score']:.4f}, Text: {r['text']}")