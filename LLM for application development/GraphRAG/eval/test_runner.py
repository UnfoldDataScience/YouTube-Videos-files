
import os, json
from ingest.chunk import chunk_corpus
from ingest.index import build_all
from rag.retriever import load_indices, hybrid_search
from rag.graphrag import load_graph, pick_entity_from_query, graph_context
from rag.crag import retrieval_confidence, corrective_actions
from rag.generator import generate_answer

STORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'store'))

def sample_docs():
    return [
        "# RAG Overview\nRetrieval-Augmented Generation (RAG) allows language models to access external, trusted data sources before generating responses. This reduces hallucinations by grounding answers in domain-specific context and up-to-date information.",
        "# Agentic AI\nAgentic AI systems act with goals, decompose tasks, and can operate over tools and memory in multi-step workflows. They differ from chatbots that only respond turn-by-turn.",
        "# On-device AI and Compact Models\nNPUs in modern laptops enable local inference for small language models, reducing latency and improving privacy. Compact models can match larger models on narrow tasks when fine-tuned."
    ]

def main():
    docs = sample_docs()
    chunks, meta = chunk_corpus(docs, max_words=80)
    build_all(chunks, STORE)
    tfidf, bm25 = load_indices(STORE)
    G = load_graph(STORE)

    queries = ["Why does RAG improve accuracy?", "How are agentic AI systems different from chatbots?"]
    results = []
    for q in queries:
        scored, all_docs = hybrid_search(q, k=8, tfidf=tfidf, bm25=bm25)
        conf = retrieval_confidence(scored)
        entity = pick_entity_from_query(q)
        gctx = graph_context(entity, G, all_docs)
        contexts = [all_docs[i] for i,_ in scored[:4]] + gctx[:2]
        answer = generate_answer(q, contexts)
        results.append({
            "query": q,
            "confidence": conf,
            "top_indices": [int(i) for i,_ in scored[:4]],
            "answer_excerpt": answer[:200] + "...",
            "needs_corrective_action": conf < 0.25
        })
    print(json.dumps({"n_chunks": len(chunks), "queries": results}, indent=2))

if __name__ == "__main__":
    main()
