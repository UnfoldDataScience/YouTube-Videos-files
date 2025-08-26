
from typing import List, Dict, Set
import re, pickle

def extract_entities(text: str) -> List[str]:
    ents = re.findall(r'\b([A-Z][a-zA-Z0-9_-]{2,30})\b', text)
    seen=set(); out=[]
    for e in ents:
        if e not in seen:
            seen.add(e); out.append(e)
    return out

def build_graph(chunks: List[str]) -> Dict[str, Set[int]]:
    g = {}
    for i, c in enumerate(chunks):
        for e in extract_entities(c):
            g.setdefault(e, set()).add(i)
    return g

def save_graph(graph, path: str):
    with open(path, 'wb') as f: pickle.dump(graph, f)

def load_graph(path: str):
    with open(path, 'rb') as f: return pickle.load(f)
