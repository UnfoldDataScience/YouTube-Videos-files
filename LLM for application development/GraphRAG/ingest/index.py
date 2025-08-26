
import os, pickle
from typing import List
from .embed import TfidfIndex
from .bm25 import BM25Lite
from .graph import build_graph, save_graph

def build_all(chunks: List[str], store_dir: str):
    os.makedirs(store_dir, exist_ok=True)
    tfidf = TfidfIndex().fit(chunks)
    tfidf.save(os.path.join(store_dir, 'tfidf.pkl'))
    bm25 = BM25Lite(chunks)
    with open(os.path.join(store_dir, 'bm25.pkl'), 'wb') as f:
        pickle.dump(bm25, f)
    G = build_graph(chunks)
    save_graph(G, os.path.join(store_dir, 'graph.pkl'))
