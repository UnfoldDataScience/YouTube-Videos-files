
import os
import numpy as np
import networkx as nx
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL   = "gpt-4o"
TOP_NODE_K  = 2
SEPARATOR   = "\n---\n"

# ----- Entities & triples -----
NODES = {
    "GraphRAG": {"type": "Method", "desc": "A method that builds a knowledge graph from documents."},
    "Hybrid RAG": {"type": "Method", "desc": "A method mixing vector search with keyword filters."},
    "Alice": {"type": "Person", "desc": "Proposed GraphRAG in 2023; works on entity linking."},
    "Bob": {"type": "Person", "desc": "Developed Hybrid RAG in 2024; works on search ranking."},
    "OpenAI": {"type": "Org", "desc": "Company where Alice works."},
    "Google": {"type": "Org", "desc": "Company where Bob works."},
}
TRIPLES = [
    ("Alice", "PROPOSED", "GraphRAG"),
    ("Bob", "PROPOSED", "Hybrid RAG"),
    ("Alice", "WORKS_AT", "OpenAI"),
    ("Bob", "WORKS_AT", "Google"),
]

# ----- Build graph -----
G = nx.MultiDiGraph()
for n, attrs in NODES.items():
    G.add_node(n, **attrs)
for h, r, t in TRIPLES:
    G.add_edge(h, t, relation=r)

# ----- OpenAI client -----
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(e.embedding, dtype=np.float32) for e in resp.data]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    den = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else 0.0

# Pre-embed node descs
NODE_KEYS = list(NODES.keys())
NODE_TEXTS = [f"{k}: {NODES[k]['desc']}" for k in NODE_KEYS]
NODE_EMBS = embed_texts(NODE_TEXTS)

def rank_nodes_for_query(query: str, top_k: int = TOP_NODE_K):
    q_emb = embed_texts([query])[0]
    scores = [(cosine_sim(q_emb, emb), key) for key, emb in zip(NODE_KEYS, NODE_EMBS)]
    scores.sort(reverse=True)
    return [k for _, k in scores[:top_k]]

def neighborhood_context(seed_nodes):
    lines, seen = [], set()
    for n in seed_nodes:
        for node in [n] + list(G.predecessors(n)) + list(G.successors(n)):
            if node in seen:
                continue
            seen.add(node)
            attr = G.nodes[node]
            lines.append(f"Node: {node} (type={attr.get('type')}) desc={attr.get('desc')}")
            for _, t, data in G.out_edges(node, data=True):
                lines.append(f"  {node} -[{data.get('relation')}]-> {t}")
            for h, _, data in G.in_edges(node, data=True):
                lines.append(f"  {h} -[{data.get('relation')}]-> {node}")
            lines.append("")
    return "\n".join(lines)

def answer_with_graph(query: str, graph_text: str):
    prompt = f"""You get a small knowledge graph neighborhood.
Use ONLY this info to answer.

Graph:
{graph_text}

Question: {query}
"""
    resp = client.responses.create(model=GEN_MODEL, input=prompt)
    return resp.output_text

if __name__ == "__main__":
    query = "Which company employs the person who proposed the method that builds a knowledge graph for RAG?"
    seeds = rank_nodes_for_query(query)
    graph_text = neighborhood_context(seeds)

    print("[graph RAG] seed nodes:", seeds)
    print("\n[graph RAG] neighborhood:\n")
    print(graph_text)

    print("\n[graph RAG] answer:")
    print(answer_with_graph(query, graph_text))
