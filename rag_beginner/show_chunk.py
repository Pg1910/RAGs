import os ,pickle

STORE_PATH = os.path.join("artifacts", "store.pkl")

def show(i: int):
    with open (STORE_PATH , "rb") as f:
        store = pickle.load(f)
    texts = store["texts"]
    metas = store["metas"]
    print(f"--- Chunk {i} ---")
    print("Source:", metas[i]["source"])
    print("Global ID:", metas[i].get("global_id", i))
    print("Text:\n", texts[i])

if __name__ == "__main__":
    show(5)

