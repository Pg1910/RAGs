import subprocess
from retrieve_persisted import retrieve

SYSTEM_RULES = """You are a question-answering assistant.

Rules (must follow):
- Use ONLY the provided CONTEXT.
- If the answer is not explicitly in the CONTEXT, reply exactly:
I don't know based on the provided context.
- When the question asks for a definition/expansion, copy the exact wording from CONTEXT (do not rephrase).
- Cite sources using ONLY this exact format: [data/doc2.txt] (or other sources exactly as shown in CONTEXT).
- Do not output any other citation format (do not write 'source:' or 'provided context')."""


def call_ollama(model:str, prompt: str) -> str:
    result = subprocess.run(
        ["ollama","run",model ,prompt],
        capture_output = True,
        text = True,
        check = True,
    )
    return result.stdout.strip()

def build_prompt(query: str, contexts : list[dict]) -> str:
    ctx_lines = []
    for i, r in enumerate(contexts):
        chunk_id = r.get('chunk_id', i)  # fallback to index if no chunk_id
        ctx_lines.append(f"[{r['source']}#{chunk_id}] {r['text']}")
    context_block = "\n".join(ctx_lines)
    allowed_sources = " ".join(sorted({f"[{r['source']}#{r.get('chunk_id', i)}]" for i, r in enumerate(contexts)}))
    return f"""{SYSTEM_RULES}

Context:
{context_block}

ALLOWED_CITATIONS:
{allowed_sources}

Question:
{query}

INSTRUCTION:
If the QUESTION asks what something "stands for" or asks for a definition, output ONLY the exact phrase from CONTEXT that answers it, followed by one citation.

ANSWER:
"""


import re
def deterministic_stands_for(query: str , contexts: list[dict]):
    q = query.lower()
    if ("stand for" not in q) and ("stands for" not in q):
        return None
    if not contexts:
        return "I don't know based on the provided context."
    top = contexts[0]
    text = top["text"]
    print("DEBUG top_text_repr:", repr(text))


    m = re.search(r"stands for\s+(.+)$", text, flags = re.IGNORECASE)
    if not m :
        return "I don't know based on the provided context." 
    extracted = m.group(1).strip()
    return f"{extracted} [{top['source']}#{top.get('chunk_id', 0)}]"




if __name__ == "__main__":
    query = "Copy exactly from context: what does RAG stand for?"
    contexts = retrieve(query , k = 5) # type: ignore
    det = deterministic_stands_for(query , contexts)
    if det is not None:
        print("question ", query)
        print("Answer:", det)
        print("sources:")
        for r in contexts: #type: ignore
            print("- ", f"{r['source']}#{r.get('chunk_id', 0)}")
        raise SystemExit(0)
    print("\n retrieved contexts :")
    for r in contexts: # type: ignore
        print(f"\n[{r['source']}](distance = {r['distance']})")
        print(r["text"])
    prompt= build_prompt (query , contexts) # type: ignore
    answer = call_ollama("llama3.2:3b", prompt)
    print("Question:", query)
    print("\nsources:")
    for r in contexts: #type: ignore
        print(f"- {r['source']}")
    print("\nAnswer:")
    print(answer)