
# Streamlit UI for Agentic Graph-RAG (OpenAI LLM)
import os, sys, pickle
from pathlib import Path

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import streamlit as st

from ingest.chunk import chunk_corpus
from ingest.index import build_all
from rag.retriever import load_indices, hybrid_search
from rag.graphrag import load_graph, pick_entity_from_query, graph_context
from rag.crag import retrieval_confidence, corrective_actions
from rag.generator import generate_answer as simple_answer
from rag.generator_openai import openai_generate

from dotenv import load_dotenv
load_dotenv("E:/YTReusable/.env")
STORE = (BASE / "store").resolve()

st.set_page_config(page_title=" Graph‑RAG (OpenAI)", layout="wide")
st.title("Graph‑RAG with OpenAI LLM")

st.sidebar.header("Corpus")
default_docs = [
    "# RAG Overview\nRetrieval-Augmented Generation (RAG) allows language models to access external, trusted data sources before generating responses. This reduces hallucinations by grounding answers in domain-specific context and up-to-date information.",
    "# Agentic AI\nAgentic AI systems act with goals, decompose tasks, and can operate over tools and memory in multi-step workflows. They differ from chatbots that only respond turn-by-turn.",
    "# On-device AI and Compact Models\nNPUs in modern laptops enable local inference for small language models, reducing latency and improving privacy. Compact models can match larger models on narrow tasks when fine-tuned."
]

use_defaults = st.sidebar.checkbox("Use sample corpus", value=True)
uploaded = st.sidebar.text_area("Or paste your own corpus (separate docs with \n\n---\n\n):", height=200)

model_name = st.sidebar.text_input("OpenAI model", value="gpt-4o-mini")
st.sidebar.caption("Set your OPENAI_API_KEY before running")

if st.sidebar.button("Build Index"):
    if not use_defaults and uploaded.strip():
        docs = [d.strip() for d in uploaded.split("\n\n---\n\n") if d.strip()]
    else:
        docs = default_docs
    chunks, meta = chunk_corpus(docs, max_words=80)
    build_all(chunks, str(STORE))
    st.sidebar.success(f"Built indices for {len(chunks)} chunks.")

q = st.text_input("Ask a question about your corpus", value="Why does RAG improve accuracy?")

colA, colB = st.columns([2,1])
with colA:
    mode = st.selectbox("Answering mode", ["Simple Template", "OpenAI LLM"])
with colB:
    topk = st.number_input("Top-K", min_value=3, max_value=20, value=8, step=1)

if st.button("Search & Answer"):
    try:
        tfidf, bm25 = load_indices(str(STORE))
        G = load_graph(str(STORE))
    except Exception:
        st.error("Please click 'Build Index' in the sidebar first.")
        st.stop()

    scored, all_docs = hybrid_search(q, k=int(topk), tfidf=tfidf, bm25=bm25)
    conf = retrieval_confidence(scored)
    entity = pick_entity_from_query(q)
    gctx = graph_context(entity, G, all_docs)
    contexts = [all_docs[i] for i,_ in scored[:4]] + gctx[:2]

    st.metric("Retrieval confidence", f"{conf:.2f}")

    if conf < 0.25:
        st.warning("Low confidence detected. Corrective plan:")
        for act in corrective_actions(q):
            st.write(f"- **{act['type']}** → {act.get('subs') or act.get('hint')}")

    st.subheader("Answer")
    if mode == "Simple Template":
        answer = simple_answer(q, contexts)
        st.code(answer, language="markdown")
    else:
        try:
            answer = openai_generate(q, contexts, model=model_name)
            st.write(answer)
        except Exception as e:
            st.error(f"OpenAI error: {e}. Did you set OPENAI_API_KEY?")

    st.subheader("Sources")
    for i,(idx,score) in enumerate(scored[:6], 1):
        with st.expander(f"Source {i} · score={score:.2f}"):
            st.write(all_docs[idx])
