import numpy as np
from index import build_index

def retrieve(query: str , k: int = 3):
    index , texts ,metas , model = build_index() # type: ignore

    q_emb = model.encode([query], convert_to_numpy = True)
    # faiss expects float32
    q_emb = q_emb.astype(np.float32)

    distances , indices = index.search(q_emb , k) # type: ignore
    results = []
    for rank ,(idx , dist) in enumerate(zip(indices[0],distances[0]),start =1):
        results.append({
            "rank": rank,
            "distance": float(dist),
            "text": texts[idx],
            "source": metas[idx]["source"],
        })
        return results
    
if __name__ == "__main__":
    query = "what is FAISS used for in RAG ?"
    results = retrieve(query , k = 5)

    print("query:", query)
    for r in results: # type: ignore
        print("\n ---")
        print("rank :" , r["rank"])
        print("distance :" , r["distance"])
        print("source :" , r["source"])
        print("text :" , r["text"])
        
