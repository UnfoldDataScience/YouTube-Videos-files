
import os, pickle, re
from typing import List

def load_graph(store_dir: str):
    with open(os.path.join(store_dir, 'graph.pkl'), 'rb') as f:
        return pickle.load(f)

def pick_entity_from_query(q: str) -> str:
    cands = re.findall(r'\b([A-Z][a-zA-Z0-9_-]{2,})\b', q)
    if not cands: return ""
    return max(cands, key=len)

def graph_context(entity: str, G, docs, max_chunks: int = 4) -> List[str]:
    if not entity or entity not in G: return []
    idxs = list(G[entity])[:max_chunks]
    return [docs[i] for i in idxs]
