# visualize_graph.py
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

with open(Path("store") / "graph.pkl", "rb") as f:
    Gdict = pickle.load(f)

G = nx.Graph()
for ent, chunks in Gdict.items():
    for c in chunks:
        G.add_edge(ent, f"chunk_{c}")

plt.figure(figsize=(9,7))
nx.draw(G, with_labels=True, node_size=600, font_size=8)
plt.tight_layout()
plt.show()
