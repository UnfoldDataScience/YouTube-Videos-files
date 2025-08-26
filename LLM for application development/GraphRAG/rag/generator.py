
from typing import List

def generate_answer(query: str, contexts: List[str]) -> str:
    snippets = []
    for c in contexts[:2]:
        s = ' '.join(c.split('.')[:2]).strip()
        if s:
            snippets.append(s)
    preview = ' '.join(snippets) if snippets else "No sufficient context retrieved."
    cites = ''.join(f"\n[Source {i+1}] {contexts[i][:120]}..." for i in range(min(4, len(contexts))))
    return f"""
Question: {query}

Answer (grounded by retrieved contexts):
{preview}

Citations:
{cites}
""".strip()
