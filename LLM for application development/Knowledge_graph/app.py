from __future__ import annotations
import openai
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from dotenv import load_dotenv
env_path = r'E:\YTReusable\.env'
load_dotenv(env_path)

openai.api_key = os.getenv("OPENAI_API_KEY")

# --------- Optional deps (fail gracefully) ----------
try:
    import networkx as nx  # type: ignore
    HAS_NETWORKX = True
except Exception:  # pragma: no cover
    nx = None
    HAS_NETWORKX = False

try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
    HAS_PLOTLY = True
except Exception:  # pragma: no cover
    px = go = None
    HAS_PLOTLY = False

# --------- Optional OpenAI (fallback if missing) ----------
_OPENAI_AVAILABLE = False
try:
    import openai  # type: ignore
    _OPENAI_AVAILABLE = True
except Exception:
    openai = None

# ---------------------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------------------

@dataclass
class KGExtraction:
    nodes: List[str]
    edges: List[Tuple[str, str, str]]  # (source, relation, target)


# ---------------------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert information extraction system.
Given raw text, extract a knowledge graph as:
- Unique entities (proper nouns, key concepts).
- Directed relations between entities with concise relation labels.
Return a JSON object with keys:
{
  "nodes": ["EntityA", "EntityB", ...],
  "edges": [{"source":"EntityA", "relation":"REL_LABEL", "target":"EntityB"}, ...]
}
Only include entities actually present or unambiguously implied in the text.
Keep relation labels short (e.g., founded, acquired, works_at, located_in, authored).
"""

def _call_openai_for_graph(text: str, model: str = "gpt-4o-mini") -> Optional[KGExtraction]:
    """
    Calls OpenAI to extract a graph. Returns None if OpenAI/keys not available.
    """
    if not _OPENAI_AVAILABLE:
        return None
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        return None

    try:
        # OpenAI v1-style client
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text[:8000]},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "{}"
    except Exception:
        # Experimental older SDK fallback
        try:
            openai.api_key = api_key  # type: ignore
            resp = openai.ChatCompletion.create(  # type: ignore
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text[:8000]},
                ],
                temperature=0.2,
            )
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            return None

    # Try to locate a JSON block in the content
    try:
        import json as _json
        start = content.find("{")
        end = content.rfind("}")
        payload = _json.loads(content[start:end+1])
        nodes = list(dict.fromkeys([str(n).strip() for n in payload.get("nodes", []) if str(n).strip()]))
        edges = []
        for e in payload.get("edges", []):
            s = str(e.get("source", "")).strip()
            r = str(e.get("relation", "")).strip() or "related_to"
            t = str(e.get("target", "")).strip()
            if s and t:
                edges.append((s, r, t))
        return KGExtraction(nodes=nodes, edges=edges)
    except Exception:
        return None


def _heuristic_extract(text: str) -> KGExtraction:
    """
    Very lightweight heuristic extractor if OpenAI is unavailable.
    - Splits sentences, finds Title-Case tokens as candidate entities.
    - Builds naive relations using verbs around "founded", "acquired", "works at", etc.
    This is only to keep the app demo-functional offline.
    """
    import re
    sentences = re.split(r"[.!?]\s+", text.strip())
    candidate_nodes: Dict[str, int] = {}
    edges: List[Tuple[str, str, str]] = []

    # collect entities as Title Case sequences (very rough)
    for s in sentences:
        for m in re.finditer(r"(?:[A-Z][a-zA-Z0-9&\-]+(?:\s+[A-Z][a-zA-Z0-9&\-]+){0,3})", s):
            entity = m.group(0).strip()
            if len(entity) < 2:
                continue
            # filter obvious non-entities
            if entity.lower() in {"the", "a", "an"}:
                continue
            candidate_nodes[entity] = candidate_nodes.get(entity, 0) + 1

    nodes = sorted(candidate_nodes, key=candidate_nodes.get, reverse=True)[:200]

    # naive patterns
    patterns = [
        (r"([A-Z][A-Za-z0-9&\-\s]{1,60}) founded ([A-Z][A-Za-z0-9&\-\s]{1,60})", "founded"),
        (r"([A-Z][A-Za-z0-9&\-\s]{1,60}) acquired ([A-Z][A-Za-z0-9&\-\s]{1,60})", "acquired"),
        (r"([A-Z][A-Za-z0-9&\-\s]{1,60}) works at ([A-Z][A-Za-z0-9&\-\s]{1,60})", "works_at"),
        (r"([A-Z][A-Za-z0-9&\-\s]{1,60}) leads ([A-Z][A-Za-z0-9&\-\s]{1,60})", "leads"),
        (r"([A-Z][A-Za-z0-9&\-\s]{1,60}) partnered with ([A-Z][A-Za-z0-9&\-\s]{1,60})", "partnered_with"),
        (r"([A-Z][A-Za-z0-9&\-\s]{1,60}) located in ([A-Z][A-Za-z0-9&\-\s]{1,60})", "located_in"),
        (r"([A-Z][A-Za-z0-9&\-\s]{1,60}) invested in ([A-Z][A-Za-z0-9&\-\s]{1,60})", "invested_in"),
    ]

    for s in sentences:
        for pat, rel in patterns:
            for m in re.finditer(pat, s):
                a = m.group(1).strip()
                b = m.group(2).strip()
                if a in nodes and b in nodes and a != b:
                    edges.append((a, rel, b))

    # If edges empty, create weak co-occur edges (same sentence)
    if not edges:
        for s in sentences:
            ents = [e for e in nodes if e in s]
            for i in range(len(ents)):
                for j in range(i+1, len(ents)):
                    edges.append((ents[i], "related_to", ents[j]))

    # unique
    seen = set()
    uniq_edges = []
    for a, r, b in edges:
        key = (a, r, b)
        if key not in seen and a != b:
            seen.add(key)
            uniq_edges.append(key)

    return KGExtraction(nodes=list(nodes), edges=uniq_edges)


def extract_with_openai(text: str) -> KGExtraction:
    """
    Top-level extraction function used by Streamlit and CLI.
    Tries OpenAI; falls back to heuristic extraction.
    """
    text = (text or "").strip()
    if not text:
        return KGExtraction(nodes=[], edges=[])

    kg = _call_openai_for_graph(text)
    if kg is not None and kg.nodes:
        return kg
    return _heuristic_extract(text)


# ---------------------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------------------

def build_graph(nodes: Iterable[str], edges: Iterable[Tuple[str, str, str]]):
    """
    Returns a networkx.Graph if available; otherwise returns an adjacency dict {node: set(neighbors)}.
    """
    nodes = [n for n in nodes if n]
    edges = [(a, r or "related_to", b) for a, r, b in edges if a and b]

    if HAS_NETWORKX:
        G = nx.Graph()
        for n in nodes:
            G.add_node(n)
        for a, r, b in edges:
            G.add_edge(a, b, relation=r)
        return G
    else:
        adj: Dict[str, set] = {n: set() for n in nodes}
        for a, _, b in edges:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        return adj


def graph_statistics(graph) -> Dict[str, float]:
    """
    Basic graph stats for UI.
    """
    if HAS_NETWORKX and isinstance(graph, nx.Graph):
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        density = nx.density(graph) if n > 1 else 0.0
        return {"nodes": n, "edges": m, "density": density}
    else:
        n = len(graph)
        m = sum(len(v) for v in graph.values()) // 2
        density = 0.0
        if n > 1:
            density = m / (n * (n - 1) / 2.0)
        return {"nodes": n, "edges": m, "density": density}


def create_interactive_graph(nodes: List[str], edges: List[Tuple[str, str, str]]):
    """
    Build a Plotly graph figure (or return None if Plotly isn't installed).
    """
    if not HAS_PLOTLY:
        return None

    # Build networkx for layout even if not installed; if not, simple layout fallback
    G = None
    if HAS_NETWORKX:
        G = nx.Graph()
        for n in nodes:
            G.add_node(n)
        for a, r, b in edges:
            G.add_edge(a, b, relation=r)

        pos = nx.spring_layout(G, k=0.8, seed=42) if G.number_of_nodes() > 0 else {}
    else:
        # fake layout: place nodes in a circle
        import math
        pos = {}
        N = len(nodes)
        for i, n in enumerate(nodes):
            angle = 2 * math.pi * i / max(1, N)
            pos[n] = (math.cos(angle), math.sin(angle))

    # Build scatter traces
    x_nodes, y_nodes, labels = [], [], []
    for n in nodes:
        xy = pos.get(n, (0, 0))
        x_nodes.append(xy[0])
        y_nodes.append(xy[1])
        labels.append(n)

    x_edges, y_edges = [], []
    edge_labels = []
    for a, r, b in edges:
        xa, ya = pos.get(a, (0, 0))
        xb, yb = pos.get(b, (0, 0))
        x_edges += [xa, xb, None]
        y_edges += [ya, yb, None]
        edge_labels.append(( (xa+xb)/2.0, (ya+yb)/2.0, r ))

    fig = go.Figure()

    # edges
    fig.add_trace(go.Scatter(
        x=x_edges, y=y_edges,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none",
        showlegend=False
    ))

    # nodes
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode="markers+text",
        text=labels,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=12),
        showlegend=False
    ))

    # Edge labels (as annotations)
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Knowledge Graph", x=0.5),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )
    for (x, y, r) in edge_labels:
        fig.add_annotation(x=x, y=y, text=r, showarrow=False, font=dict(size=10))

    return fig


# ---------------------------------------------------------------------------------------
# Context insights & summary
# ---------------------------------------------------------------------------------------

def extract_context_insights(nodes: List[str], edges: List[Tuple[str, str, str]]) -> Dict[str, List[str]]:
    """
    Simple heuristic "insights" about context richness or gaps.
    Returns:
      {
        "well_connected": [entity...],
        "isolated": [entity...],
        "missing_context": [entity...]
      }
    """
    from collections import defaultdict
    deg = defaultdict(int)
    neighbors = defaultdict(set)
    for a, _, b in edges:
        deg[a] += 1
        deg[b] += 1
        neighbors[a].add(b)
        neighbors[b].add(a)

    well = [n for n in nodes if deg[n] >= 3]
    iso = [n for n in nodes if deg[n] == 0]
    # "missing context" = degree 1 (dangling) — likely needs more info
    missing = [n for n in nodes if deg[n] == 1]

    return {"well_connected": well, "isolated": iso, "missing_context": missing}


def generate_context_summary(nodes: List[str], edges: List[Tuple[str, str, str]]) -> List[str]:
    """
    Turns the insights into short bullet points (for UI display).
    """
    ins = extract_context_insights(nodes, edges)
    bullets: List[str] = []
    bullets.append(f"Entities: {len(nodes)}  •  Relations: {len(edges)}")
    if ins["well_connected"]:
        bullets.append(f"Rich clusters around: {', '.join(ins['well_connected'][:5])}")
    if ins["isolated"]:
        bullets.append(f"Isolated entities (need context): {', '.join(ins['isolated'][:5])}")
    if ins["missing_context"]:
        bullets.append(f"Dangling entities (degree=1): {', '.join(ins['missing_context'][:8])}")
    return bullets


# ---------------------------------------------------------------------------------------
# Path finding & rankings & link suggestions
# ---------------------------------------------------------------------------------------

def shortest_path(graph, edges: List[Tuple[str, str, str]], src: str, tgt: str) -> Tuple[List[str], List[str]]:
    """
    Returns (node_path, relation_labels_between_consecutive_nodes).
    """
    if not src or not tgt or src == tgt:
        return [], []

    if HAS_NETWORKX and isinstance(graph, nx.Graph):
        try:
            nodes_path = nx.shortest_path(graph, source=src, target=tgt)
        except Exception:
            return [], []
    else:
        # BFS in adjacency dict
        from collections import deque
        adj = graph
        q = deque([src])
        parent = {src: None}
        while q:
            u = q.popleft()
            if u == tgt: break
            for v in adj.get(u, []):
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        if tgt not in parent:
            return [], []
        # reconstruct
        nodes_path = []
        cur = tgt
        while cur is not None:
            nodes_path.append(cur)
            cur = parent[cur]
        nodes_path.reverse()

    # relation labels along path
    rels = []
    e_map = {}
    for a, r, b in edges:
        e_map[(a, b)] = r
        e_map[(b, a)] = r
    for i in range(len(nodes_path)-1):
        a, b = nodes_path[i], nodes_path[i+1]
        rels.append(e_map.get((a, b), "related_to"))
    return nodes_path, rels


def centrality(graph):
    """
    Returns (degree_dict, pagerank_dict_or_None)
    """
    if HAS_NETWORKX and isinstance(graph, nx.Graph):
        deg = dict(graph.degree())
        pr = nx.pagerank(graph) if graph.number_of_edges() > 0 else {n: 0.0 for n in graph.nodes()}
        return deg, pr
    else:
        deg = {n: len(neigh) for n, neigh in graph.items()}
        return deg, None


def suggest_links(graph, top_k: int = 10) -> List[Tuple[str, str, int]]:
    """
    Simple link prediction by common neighbors.
    Returns list of (u, v, score) sorted by score desc.
    """
    pairs: Dict[Tuple[str, str], int] = {}
    if HAS_NETWORKX and isinstance(graph, nx.Graph):
        for u in graph.nodes():
            for v in graph.nodes():
                if u >= v or graph.has_edge(u, v):
                    continue
                cn = len(list(nx.common_neighbors(graph, u, v)))
                if cn > 0:
                    pairs[(u, v)] = cn
    else:
        for u, Nu in graph.items():
            for v, Nv in graph.items():
                if u >= v or v in Nu:
                    continue
                cn = len(Nu.intersection(Nv))
                if cn > 0:
                    pairs[(u, v)] = cn

    ranked = sorted(((a, b, s) for (a, b), s in pairs.items()), key=lambda x: x[2], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------------------
# CLI (for quick tests)
# ---------------------------------------------------------------------------------------

DEMO_TEXTS = {
    "startup": """Sarah founded TechFlow, a machine learning startup, and hired Maria as Head of Sales. 
    TechFlow partnered with Venture Capital Partners. Before that, Maria worked at Microsoft. 
    John joined as CTO and led the platform team."""
}

def cli_extract(text: str) -> Dict:
    kg = extract_with_openai(text)
    G = build_graph(kg.nodes, kg.edges)
    stats = graph_statistics(G)
    return {
        "nodes": kg.nodes,
        "edges": [{"source": a, "relation": r, "target": b} for a, r, b in kg.edges],
        "stats": stats,
    }

def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph extractor demo")
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--file", type=str, help="Path to a text file")
    g.add_argument("--demo", choices=list(DEMO_TEXTS.keys()), help="Run a built-in demo")
    args = parser.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = DEMO_TEXTS.get(args.demo or "startup")

    result = cli_extract(text)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
