import os 
import pickle
import re
from tracemalloc import start
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from rerank import rerank  # type: ignore

OUT_DIR = "artifacts"
INDEX_PATH = os.path.join(OUT_DIR, "index.faiss")
STORE_PATH = os.path.join(OUT_DIR, "store.pkl")

STOPWORDS = {"what", "does", "do", "is", "are", "the", "a", "an", "for", "in", "of", "to", "and"}

def simple_tokenize(text: str):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    norm = []
    for t in toks:
        if t in STOPWORDS:
            continue
        # very light stemming: stands -> stand
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        norm.append(t)
    return norm


def load_artifacts():
    index = faiss.read_index(INDEX_PATH)
    with open (STORE_PATH, "rb") as f:
        store = pickle.load(f)
    texts = store["texts"]
    metas = store["metas"]
    return index , texts , metas

def faiss_candidates(query : str, k: int , model: SentenceTransformer , index ):
    q = model.encode([query], convert_to_numpy = True).astype(np.float32)
    distances , indices = index.search(q,k)
    # convert distances --> similarity -like score (higher better) via negation 
    sims = 1.0/(1.0 + distances[0])
    ids = indices[0]
    return { int(i): float(s) for i , s in zip(ids , sims)}

def bm25_candidates(query: str, k: int, bm25: BM25Okapi):
    q_tokens = simple_tokenize(query)
    scores = bm25.get_scores(q_tokens)  # numpy array, len = corpus size

    # debug: show max score
    max_score = float(np.max(scores)) if len(scores) else 0.0

    # get top-k indices by score
    top_idx = np.argsort(scores)[::-1][:k]

    out = {}
    for i in top_idx:
        out[int(i)] = float(scores[int(i)])
    return out


def max_norm(score_dict: dict[int, float]) -> dict[int, float]: # type: ignore
    if not score_dict:
        return{}
    vmax = max(score_dict.values())
    if vmax == 0:
        return { k: 0.0 for k in score_dict}
    return { k:v / vmax for k , v in score_dict.items()}


def hybrid_retrieve(query : str , top_k : int =5 , faiss_k : int = 10 , bm25_k : int = 10 , alpha :float = 0.5):
    
    """
    Performs hybrid retrieval combining FAISS semantic search and BM25 keyword search.
    
    Retrieves relevant documents using both dense vector similarity (FAISS) and 
    sparse keyword matching (BM25), then ranks results by a weighted combination of both scores.
    
    Args:
        query (str): The search query string.
        top_k (int, optional): Number of top results to return. Defaults to 5.
        faiss_k (int, optional): Number of candidates to retrieve from FAISS. Defaults to 10.
        bm25_k (int, optional): Number of candidates to retrieve from BM25. Defaults to 10.
        alpha (float, optional): Weight for FAISS score in hybrid ranking (0.0 to 1.0).
                                 BM25 weight is (1 - alpha). Defaults to 0.5.
    
    Returns:
        list: A list of dictionaries containing ranked results, each with:
            - rank (int): Result position (1-indexed)
            - hybrid_score (float): Combined normalized score
            - text (str): Document chunk text
            - source (str): Document source
            - global_id: Document global identifier
            - doc_chunk_id: Document chunk identifier
            - bm25_norm (float): Normalized BM25 score
            - faiss_norm (float): Normalized FAISS score
    """
    index , texts , metas = load_artifacts()
    tokenized = [simple_tokenize(t) for t in texts]
    
    bm25 = BM25Okapi(tokenized)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    faiss_raw = faiss_candidates(query , faiss_k , model , index)
    bm25_raw = bm25_candidates(query , bm25_k , bm25)

    faiss_n = max_norm(faiss_raw)
    bm25_n = max_norm(bm25_raw)
    # union of candidate ids
    all_ids = set(faiss_n.keys()) | set(bm25_n.keys())
    combined = {}
    for i in all_ids :
        combined[i] = alpha * faiss_n.get(i, 0.0) + (1- alpha)* bm25_n.get(i, 0.0)

    # sort by combined score descending
    ranked = sorted (combined.items(), key = lambda x: x[1], reverse =True)[:top_k] # type: ignore
    results = []
    for rank , (i , score) in enumerate(ranked , start = 1):
        results.append({
            "rank": rank,
            "hybrid_score": float(score),
            "text": texts[i],
            "source": metas[i]["source"],
            "global_id": metas[i].get("global_id", i),
            "doc_chunk_id": metas[i].get("chunk_id", None),
            "bm25_norm": float(bm25_n.get(i,0.0)),
            "faiss_norm": float(faiss_n.get(i,0.0))
        })
    q = query.lower()
    if ("stand for" not in q) and ("stands for" not in q):
        results = [r for r in results if "stands for" not in r["text"].lower()]
    reranked = rerank(q , results , top_k = top_k)
    for j , c in enumerate (reranked, start =1):
        c["rank"] = j

    return reranked

if __name__ == "__main__":
    q = "what is the role of vector index in RAG?"
    res = hybrid_retrieve(q , top_k =3 , faiss_k =10 , bm25_k =10 , alpha =0.2)
    print("query :", q)
    for r in res:
        print("\n---")
        print("rank:", r["rank"])
        print("hybrid score:", r["hybrid_score"])
        print("bm25 norm:", r["bm25_norm"])
        print("faiss norm:", r["faiss_norm"])
        print("text:", r["text"])
        print("source:", r["source"], "global_id:", r["global_id"], "doc_chunk_id:", r["doc_chunk_id"])
        print("rerank score:", r.get("rerank_score", None))

