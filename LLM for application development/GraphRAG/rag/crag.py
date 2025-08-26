
from typing import List

def retrieval_confidence(scored: List[tuple]) -> float:
    if not scored: return 0.0
    vals = [s for _,s in scored[:3]]
    return sum(vals)/len(vals)

def corrective_actions(query: str):
    subs = [q.strip() for q in query.replace(' and ', ';').split(';') if q.strip()]
    return [
        {"type":"decompose", "subs": subs[:3]},
        {"type":"hyde", "prompt": f"Draft a 3-sentence passage that would answer: {query}"},
        {"type":"expand_scope", "hint": "include glossary; longer chunks; relaxed filters"}
    ]
