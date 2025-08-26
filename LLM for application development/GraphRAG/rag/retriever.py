
from typing import Tuple, List
import os, pickle
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def load_indices(store_dir: str):
    with open(os.path.join(store_dir, 'tfidf.pkl'), 'rb') as f:
        tfidf = pickle.load(f)  # (vectorizer, doc_vectors, docs)
    with open(os.path.join(store_dir, 'bm25.pkl'), 'rb') as f:
        bm25 = pickle.load(f)  # BM25Lite
    return tfidf, bm25

def hybrid_search(query: str, k: int, tfidf, bm25):
    vec, mat, docs = tfidf
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat).ravel()
    tfidf_candidates = [(i, float(sims[i])) for i in sims.argsort()[-k:][::-1]]
    bm25_candidates = bm25.topk(query, k=k)

    # normalize bm25 0..1
    if bm25_candidates:
        bmax = max(s for _,s in bm25_candidates) or 1.0
        bmin = min(s for _,s in bm25_candidates)
    else:
        bmax, bmin = 1.0, 0.0

    cand = defaultdict(lambda: [0.0, 0.0])
    for i,s in tfidf_candidates: cand[i][0] = max(cand[i][0], s)
    for i,s in bm25_candidates:
        ns = (s - bmin) / (bmax - bmin + 1e-9)
        cand[i][1] = max(cand[i][1], ns)

    alpha = 0.6
    scored = [(i, alpha*v[0] + (1-alpha)*v[1]) for i,v in cand.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k], docs
