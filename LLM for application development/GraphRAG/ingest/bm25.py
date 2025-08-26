
from typing import List, Tuple
import math
from collections import Counter

class BM25Lite:
    """Pure-Python BM25 (Okapi-style)."""
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1; self.b = b
        self.docs = [doc.split() for doc in corpus]
        self.N = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        self.df = {}
        for d in self.docs:
            for term in set(d):
                self.df[term] = self.df.get(term, 0) + 1
        self.tfs = [Counter(d) for d in self.docs]

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query: str, idx: int) -> float:
        q_terms = query.split()
        dl = self.doc_len[idx]
        score = 0.0
        for t in q_terms:
            tf = self.tfs[idx].get(t, 0)
            if tf == 0: continue
            idf = self.idf(t)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def topk(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
