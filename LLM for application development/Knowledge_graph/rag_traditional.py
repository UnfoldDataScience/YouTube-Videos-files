# -*- coding: utf-8 -*-
"""
Traditional RAG — minimal, no database.
- Tiny toy corpus
- Simple chunk → embed → cosine → prompt → generate
Requires: openai, numpy
"""

import os
import numpy as np
from openai import OpenAI

# ---------- Config ----------
EMBED_MODEL = "text-embedding-3-small"   # cheap, solid for demos
GEN_MODEL   = "gpt-4o"                   # versatile model for answers
TOP_K       = 3
CHUNK_SIZE  = 400                         # characters
SEPARATOR   = "\n---\n"

# ---------- Data (tiny toy set) ----------
DOCS = [
    {
        "id": "d1",
        "text": (
            "GraphRAG is a retrieval technique that builds a knowledge graph from documents. "
            "By connecting entities and relations, it enables multi-hop reasoning and reduces missed links."
        )
    },
    {
        "id": "d2",
        "text": (
            "Hybrid RAG mixes vector search with keyword and metadata filters. "
            "It is useful when strict filters (like source or date) are important."
        )
    },
    {
        "id": "d3",
        "text": (
            "Alice proposed GraphRAG in a 2023 workshop paper and later extended it. "
            "Her work focuses on entity linking and relation extraction."
        )
    },
    {
        "id": "d4",
        "text": (
            "Alice works at OpenAI, where she studies retrieval-augmented generation and evaluation."
        )
    },
    {
        "id": "d5",
        "text": (
            "Bob developed Hybrid RAG in 2024. Bob works at Google on search relevance and ranking."
        )
    },
]

# ---------- Helpers ----------
def chunk_text(s: str, chunk_size: int = CHUNK_SIZE):
    return [s[i:i+chunk_size] for i in range(0, len(s), chunk_size)]

def build_corpus(docs):
    rows = []
    for d in docs:
        chunks = chunk_text(d["text"])
        for i, c in enumerate(chunks):
            rows.append({"doc_id": d["id"], "chunk_id": f"{d['id']}_c{i}", "text": c})
    return rows

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    den = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else 0.0

# ---------- Build corpus ----------
CORPUS = build_corpus(DOCS)

# ---------- OpenAI client ----------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(e.embedding, dtype=np.float32) for e in resp.data]

# Pre-embed corpus
CORPUS_EMBS = embed_texts([r["text"] for r in CORPUS])

def retrieve(query: str, top_k: int = TOP_K):
    q_emb = embed_texts([query])[0]
    scored = [(cosine_sim(q_emb, emb), row) for row, emb in zip(CORPUS, CORPUS_EMBS)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def answer_with_context(query: str, ctx_chunks):
    context = SEPARATOR.join([r["text"] for r in ctx_chunks])
    prompt = f"""Use ONLY the provided context to answer.

Context:
{context}

Question: {query}
"""
    resp = client.responses.create(model=GEN_MODEL, input=prompt)
    return resp.output_text

if __name__ == "__main__":
    query = "Which company employs the person who proposed the method that builds a knowledge graph for RAG?"
    top = retrieve(query)
    top_rows = [row for _, row in top]

    print("\n[traditional RAG] top chunks:")
    for i, (score, row) in enumerate(top, 1):
        print(f"{i}. {row['chunk_id']}  score={score:.3f}  text={row['text'][:90]}...")

    print("\n[traditional RAG] answer:")
    print(answer_with_context(query, top_rows))
