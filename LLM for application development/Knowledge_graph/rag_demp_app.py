# -*- coding: utf-8 -*-
# Streamlit demo: Compare Traditional RAG vs Graph-RAG
# ----------------------------------------------------
# - Traditional RAG: vector search over toy corpus
# - Graph-RAG: in-memory NetworkX knowledge graph
#
# Run:
#   streamlit run rag_demo_app.py

import os
import numpy as np
import streamlit as st
import networkx as nx
from openai import OpenAI

# -------------------- Config --------------------
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL   = "gpt-4o"
SEPARATOR   = "\n---\n"

# -------------------- Tiny corpus ----------------
DOCS = [
    {"id": "d1", "text": "GraphRAG is a retrieval technique that builds a knowledge graph from documents."},
    {"id": "d2", "text": "Hybrid RAG mixes vector search with keyword and metadata filters."},
    {"id": "d3", "text": "Alice proposed GraphRAG in 2023 and works on entity linking."},
    {"id": "d4", "text": "Alice works at OpenAI on retrieval-augmented generation."},
    {"id": "d5", "text": "Bob developed Hybrid RAG in 2024. Bob works at Google."},
]

NODES = {
    "GraphRAG": {"type": "Method", "desc": "A method that builds a knowledge graph from documents."},
    "Hybrid RAG": {"type": "Method", "desc": "A method mixing vector search with keyword filters."},
    "Alice": {"type": "Person", "desc": "Proposed GraphRAG in 2023."},
    "Bob": {"type": "Person", "desc": "Developed Hybrid RAG in 2024."},
    "OpenAI": {"type": "Org", "desc": "Company where Alice works."},
    "Google": {"type": "Org", "desc": "Company where Bob works."},
}
TRIPLES = [
    ("Alice", "PROPOSED", "GraphRAG"),
    ("Bob", "PROPOSED", "Hybrid RAG"),
    ("Alice", "WORKS_AT", "OpenAI"),
    ("Bob", "WORKS_AT", "Google"),
]

# -------------------- Helpers --------------------
def cosine_sim(a, b):
    den = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else 0.0

def embed_many(client, texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

def answer(client, model, context, query):
    prompt = f"""Use ONLY the provided context to answer.

Context:
{context}

Question: {query}
"""
    resp = client.responses.create(model=model, input=prompt, temperature=0)
    return resp.output_text

def build_graph(nodes, triples):
    G = nx.MultiDiGraph()
    for n, attrs in nodes.items():
        G.add_node(n, **attrs)
    for h, r, t in triples:
        G.add_edge(h, t, relation=r)
    return G

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="RAG vs Graph-RAG Demo", layout="wide")
st.title("🔎 Traditional RAG vs 🕸️ Graph-RAG")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    model = st.text_input("Generation model", value=GEN_MODEL)
    st.caption("Defaults: embeddings=text-embedding-3-small, gen=gpt-4o")

if not api_key:
    st.warning("Enter your OpenAI key in the sidebar.")
    st.stop()

client = OpenAI(api_key=api_key)

query = st.text_input("Ask a question:", value="Which company employs the person who proposed the method that builds a knowledge graph for RAG?")

tab1, tab2 = st.tabs(["Traditional RAG", "Graph-RAG"])

# -------------------- Traditional RAG --------------------
with tab1:
    st.subheader("Traditional RAG")
    corpus = [{"chunk_id": d["id"], "text": d["text"]} for d in DOCS]
    corpus_embs = embed_many(client, [c["text"] for c in corpus])
    q_emb = embed_many(client, [query])[0]

    scored = [(cosine_sim(q_emb, e), c) for c, e in zip(corpus, corpus_embs)]
    scored.sort(reverse=True)
    top = scored[:3]

    st.markdown("**Top Chunks:**")
    for i, (score, row) in enumerate(top, 1):
        st.write(f"{i}. {row['chunk_id']} | score={score:.3f}")
        st.code(row["text"])

    ctx = SEPARATOR.join([r["text"] for _, r in top])
    ans = answer(client, model, ctx, query)
    st.success(ans)

# -------------------- Graph-RAG --------------------
with tab2:
    st.subheader("Graph-RAG")
    G = build_graph(NODES, TRIPLES)
    node_texts = [f"{k}: {NODES[k]['desc']}" for k in NODES]
    node_embs = embed_many(client, node_texts)
    q_emb = embed_many(client, [query])[0]

    scored = [(cosine_sim(q_emb, e), k) for k, e in zip(NODES.keys(), node_embs)]
    scored.sort(reverse=True)
    seeds = [k for _, k in scored[:2]]
    st.write("**Seed nodes:**", seeds)

    # collect neighborhood
    seen, lines = set(), []
    for n in seeds:
        for node in [n] + list(G.predecessors(n)) + list(G.successors(n)):
            if node in seen:
                continue
            seen.add(node)
            attr = G.nodes[node]
            lines.append(f"Node: {node} ({attr.get('type')}) {attr.get('desc')}")
            for _, t, data in G.out_edges(node, data=True):
                lines.append(f"  {node} -[{data['relation']}]-> {t}")
            for h, _, data in G.in_edges(node, data=True):
                lines.append(f"  {h} -[{data['relation']}]-> {node}")
            lines.append("")
    graph_ctx = "\n".join(lines)

    st.markdown("**Neighborhood context:**")
    st.code(graph_ctx)

    ans = answer(client, model, graph_ctx, query)
    st.success(ans)
