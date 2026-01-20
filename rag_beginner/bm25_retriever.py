import os 
import pickle
import re
from rank_bm25 import BM25Okapi

OUT_DIR = "artifacts"
STORE_PATH = os.path.join(OUT_DIR, "store.pkl" )

def simple_tokenize(text: str) :
    # lowecase , keep only alphanumercs as tokens 
    return re.findall(r"[a-z0-9]+", text.lower())

def load_store():
    with open(STORE_PATH , "rb") as f:
        store = pickle.load(f)
    return store["texts"], store["metas"]

def bm25_retrieve(query : str,k :int =5):
    texts , metas = load_store()
    tokenized_corpus = [simple_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    q_tokens = simple_tokenize(query)
    scores = bm25.get_scores(q_tokens) # higher is better

    # get top-k indices by score
    top_idx = sorted(range(len(scores)), key = lambda i : scores[i], reverse=True)[:k]
    results = []
    for rank , i in enumerate(top_idx , start = 1):
        results.append({
            "rank": rank,
            "bm_25_score": float(scores[i]),
            "text": texts[i],
            "source": metas[i]["source"],
            "chunk_id": metas[i].get("global_id",i)

        })
        return results

if __name__ == "__main__":
    query = "RAG stands for ?"
    results = bm25_retrieve(query , k =5)
    print("query :", query)
    for r in results: # type: ignore
        print("\n---")
        print(f"Rank: {r['rank']}, Score: {r['bm_25_score']}")
        print(f"Text: {r['text']}")
        print(f"Source: {r['source']}, Chunk ID: {r['chunk_id']}")
        