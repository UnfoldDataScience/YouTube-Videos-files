
from typing import List
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfIndex:
    def __init__(self):
        self.vectorizer = None
        self.doc_vectors = None
        self.docs = None

    def fit(self, docs: List[str]):
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(docs)
        self.docs = docs
        return self

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump((self.vectorizer, self.doc_vectors, self.docs), f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.vectorizer, self.doc_vectors, self.docs = pickle.load(f)
        return self

    def search(self, query: str, k: int = 20):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.doc_vectors).ravel()
        idxs = sims.argsort()[-k:][::-1]
        return [(int(i), float(sims[i])) for i in idxs]
