import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from chunking import load_txt_files , chunk_text

def build_index(data_dir ="data"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = []
    metadatas = []

    docs = load_txt_files(data_dir)
    for d in docs:
        chunks = chunk_text(d["text"], chunk_size = 120 , overlap =30)
        for j, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({"source": d["source"], "chunk_id": j,
                              })


    embeddings = model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings) # type: ignore
    return index, texts, metadatas, model

if __name__ == "__main__":
    index , texts, metas, model = build_index() # type: ignore
    print("total chunks indexed :",len(texts))
    print("embedding dimension :", index.d)