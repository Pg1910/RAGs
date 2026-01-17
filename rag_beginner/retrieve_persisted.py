import os 
import pickle
from tracemalloc import start 
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

OUT_DIR = "artifacts"
INDEX_PATH = os.path.join(OUT_DIR , "index.faiss")
STORE_PATH = os.path.join(OUT_DIR , "store.pkl")

def load_artifacts():
    index = faiss.read_index(INDEX_PATH)
    with open(STORE_PATH , "rb") as f:
        store  = pickle.load(f)
    texts = store["texts"]
    metas = store["metas"]
    return index , texts , metas

def retrieve(query: str , k: int = 5):
    index , texts , metas = load_artifacts()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    q_emb = model.encode([query], convert_to_numpy= True).astype(np.float32)
    distances , indices = index.search(q_emb , k) # type: ignore
    results = []

    for rank ,(idx , dist) in enumerate(zip(indices[0],distances[0]),start =1):
        text = texts[idx].strip()
    # If chunk starts mid-word, drop the first partial token
        if text and text[0].islower() and " " in text[:30]:
            first_space = text.find(" ")
            text = text[first_space+1:].strip()
        results.append({
            "rank": rank,
            "distance": float(dist),
            "text": text,
            "source": metas[idx]["source"],
        })
        return results
    
if __name__ == "__main__":
    query = "what does RAG stand for?"
    results = retrieve(query , k = 5)
    print("query:", query)
    for r in results: # type: ignore
        print("\n ---")
        print("rank :" , r["rank"])
        print("distance :" , r["distance"])
        print("source :" , r["source"])
        print("text :" , r["text"])
    