import os 
import pickle 
import faiss 
from index import build_index

OUT_DIR = "artifacts"
INDEX_PATH = os.path.join(OUT_DIR , "index.faiss")
STORE_PATH = os.path.join(OUT_DIR , "store.pkl")


def save():
    os.makedirs(OUT_DIR , exist_ok =True)
    index , texts , metas , _model = build_index() # type: ignore
    faiss.write_index(index , INDEX_PATH)
    for i , m in enumerate(metas):
        m["global_id"] = i
    with open(STORE_PATH , "wb") as f:
        pickle.dump({"texts": texts , "metas": metas}, f)

    print("saved:", INDEX_PATH, STORE_PATH)
    print(" total chunks indexed :", len(texts))


if __name__ == "__main__":
    save()

    