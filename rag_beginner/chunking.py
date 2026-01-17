from pathlib import Path 
from typing import List ,Dict

def load_txt_files(folder: str) -> List[Dict]:
    docs = []
    for p in sorted(Path(folder).glob("*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        docs.append({"source": str(p), "text": text})
    return docs

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Whitespace-aware character chunking:
    - tries to end chunks at a whitespace boundary to avoid cutting words
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        raw_end = min(start + chunk_size, n)
        end = raw_end

        if raw_end < n:
            # move end left to nearest whitespace (up to 40 chars back)
            backtrack = 0
            while end > start and backtrack < 40 and text[end - 1] not in (" ", "\n", "\t"):
                end -= 1
                backtrack += 1
            # if we failed to find whitespace, fall back to raw_end
            if end == start:
                end = raw_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break
        # Ensure we always move forward by at least 1 character
        next_start = end - overlap
        start = max(next_start, start + 1)

    return chunks # type: ignore


if __name__ == "__main__":
    docs = load_txt_files("data")
    for d in docs:
        chunks = chunk_text(d["text"], chunk_size=120,overlap = 30)
        print(" \n source :", d["source"])
        for i , c in enumerate(chunks):
            print(f"---- chunk {i} (len = {len(c)})")
            print(c)