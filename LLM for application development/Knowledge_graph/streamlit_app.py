from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import streamlit as st

from app import (
    KGExtraction,
    build_graph,
    centrality,
    create_interactive_graph,
    extract_context_insights,
    extract_with_openai,
    generate_context_summary,
    graph_statistics,
    shortest_path,
    suggest_links,
)

# ---------------------------------- Page setup ----------------------------------
st.set_page_config(
    page_title="Knowledge Graph Builder + Demos",
    page_icon="🧠",
    layout="wide",
)

st.title("Knowledge Graph from Text — with Practical")
st.caption("Paste text → Build a graph → Prove it's useful (relationships, influencers, missing links).")

# ---------------------------------- Sidebar ----------------------------------
with st.sidebar:
    st.header("⚙️ Options")
    use_sample = st.toggle("Use sample text", value=True, help="Quick demo text to try the app")
    model_hint = st.text_input("LLM model (hint)", value="gpt-4o-mini")
    st.caption("If your OpenAI key is in env, we'll try the LLM; otherwise a fallback extractor keeps the demo working.")

# ---------------------------------- Input text ----------------------------------
DEFAULT_TEXT = """Sarah founded TechFlow, a machine learning startup, and hired Maria as Head of Sales.
TechFlow partnered with Venture Capital Partners. Before that, Maria worked at Microsoft.
John joined as CTO and led the platform team."""

text = DEFAULT_TEXT if use_sample else st.text_area("Paste your text here", height=220, placeholder="Paste any article, notes, or transcript...")

col0a, col0b = st.columns([1,1])
with col0a:
    extract_button = st.button("🚀 Extract Knowledge Graph", type="primary", use_container_width=True)
with col0b:
    clear_button = st.button("🧹 Clear", use_container_width=True)

if clear_button:
    for k in ["nodes", "edges", "graph", "stats", "original_text"]:
        st.session_state.pop(k, None)
    st.rerun()

# ---------------------------------- Extraction ----------------------------------
if extract_button:
    kg = extract_with_openai(text)
    G = build_graph(kg.nodes, kg.edges)
    stats = graph_statistics(G)
    st.session_state["nodes"] = kg.nodes
    st.session_state["edges"] = kg.edges
    st.session_state["graph"] = G
    st.session_state["stats"] = stats
    st.session_state["original_text"] = text

# If we already have a graph, show tabs
if all(k in st.session_state for k in ["nodes", "edges", "graph", "stats"]):
    nodes: List[str] = st.session_state["nodes"]
    edges: List[Tuple[str, str, str]] = st.session_state["edges"]
    graph = st.session_state["graph"]
    stats = st.session_state["stats"]

    tab1, tab2, tab_use, tab3, tab4 = st.tabs([
        "📊 Graph Visualization",
        "🔍 Node Explorer",
        "🧪 Use Cases (Prove Value)",
        "📋 Raw Data",
        "📈 Statistics",
    ])

    # ---------------------------- Tab 1: Visualization ----------------------------
    with tab1:
        st.subheader("Graph Visualization")
        fig = create_interactive_graph(nodes, edges)
        if fig is None:
            st.warning("Install Plotly to see the interactive graph: `pip install plotly`")
        else:
            st.plotly_chart(fig, use_container_width=True, height=650)

    # ----------------------------- Tab 2: Explorer -------------------------------
    with tab2:
        st.subheader("Node Explorer")
        colA, colB = st.columns([1, 2])
        with colA:
            picked = st.selectbox("Pick an entity", [""] + nodes)
        if picked:
            # neighbors + relations
            neigh = []
            rels = []
            for a, r, b in edges:
                if a == picked:
                    neigh.append(b)
                    rels.append((a, r, b))
                elif b == picked:
                    neigh.append(a)
                    rels.append((b, r, a))
            st.markdown(f"**Neighbors of `{picked}` ({len(set(neigh))})**")
            st.write(sorted(set(neigh)))

            # subgraph
            sub_nodes = sorted(set([picked] + neigh))
            sub_edges = [e for e in edges if e[0] in sub_nodes and e[2] in sub_nodes]
            mini = create_interactive_graph(sub_nodes, sub_edges)
            if mini:
                mini.update_layout(title=dict(text=f"Subgraph around {picked}"))
                st.plotly_chart(mini, use_container_width=True)

    # -------------------------- Tab Use: Proving Value ---------------------------
    with tab_use:
        st.subheader("🧪 Demonstrating Graph Utility")

        # (A) Relationship Finder
        st.markdown("### A) 🔎 How are two entities related?")
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            src = st.selectbox("Entity A", [""] + nodes, key="src_pick")
        with c2:
            tgt = st.selectbox("Entity B", [""] + nodes, key="tgt_pick")
        with c3:
            go = st.button("Find Path", type="secondary", use_container_width=True)

        if go:
            if not src or not tgt or src == tgt:
                st.warning("Pick two different existing entities.")
            else:
                path_nodes, path_rels = shortest_path(graph, edges, src, tgt)
                if not path_nodes:
                    st.error(f"No path found between **{src}** and **{tgt}**.")
                else:
                    st.success(f"Found a path with {len(path_nodes)-1} hops")
                    steps = []
                    for i in range(len(path_rels)):
                        steps.append(f"**{path_nodes[i]}** — *{path_rels[i]}* → **{path_nodes[i+1]}**")
                    st.markdown("<br>".join(steps), unsafe_allow_html=True)

                    # subgraph figure
                    sub_nodes = list(dict.fromkeys(path_nodes))
                    sub_edges = []
                    for i in range(len(path_nodes)-1):
                        a, b = path_nodes[i], path_nodes[i+1]
                        # find label
                        rel = "related_to"
                        for (x, r, y) in edges:
                            if (x == a and y == b) or (x == b and y == a):
                                rel = r; break
                        sub_edges.append((a, rel, b))
                    mini = create_interactive_graph(sub_nodes, sub_edges)
                    if mini:
                        mini.update_layout(title=dict(text="Shortest Path Subgraph"))
                        st.plotly_chart(mini, use_container_width=True)

        st.divider()

        # (B) Influence ranking
        st.markdown("### B) 🏆 Top Influential Entities")
        deg, pr = centrality(graph)
        import pandas as pd
        deg_df = pd.DataFrame(sorted(deg.items(), key=lambda x: x[1], reverse=True), columns=["Entity", "Degree"])
        colb1, colb2 = st.columns(2)
        with colb1:
            st.markdown("**By Degree (connections)**")
            st.dataframe(deg_df.head(10), use_container_width=True)
        with colb2:
            if pr is not None:
                pr_df = pd.DataFrame(sorted(pr.items(), key=lambda x: x[1], reverse=True), columns=["Entity", "PageRank"])
                st.markdown("**By PageRank (influence flow)**")
                st.dataframe(pr_df.head(10), use_container_width=True)
            else:
                st.info("Install `networkx` to compute PageRank.")

        st.divider()

        # (C) Link prediction
        st.markdown("### C) 🤝 Suggested Missing Links")
        links = suggest_links(graph, top_k=10)
        if not links:
            st.info("No suggestions found — try a richer text.")
        else:
            import pandas as pd
            df = pd.DataFrame([{"Source": a, "Target": b, "Score": s} for a, b, s in links])
            st.dataframe(df, use_container_width=True)
            st.caption("Higher score = more common neighbors ⇒ likely missing connection.")

    # ------------------------------ Tab 3: Raw Data -------------------------------
    with tab3:
        st.subheader("Raw Data")
        st.markdown("**Nodes**")
        st.write(nodes)
        st.markdown("**Edges**")
        st.json([{"source": a, "relation": r, "target": b} for a, r, b in edges])

    # ------------------------------ Tab 4: Stats ---------------------------------
    with tab4:
        st.subheader("Statistics & Context")
        st.json(stats)
        bullets = generate_context_summary(nodes, edges)
        st.markdown("### Context Summary")
        for b in bullets:
            st.markdown(f"- {b}")
else:
    st.info("Paste text or use the sample, then click **Extract Knowledge Graph**.")
