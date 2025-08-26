
from typing import List
import os
from openai import OpenAI

SYSTEM_PROMPT = """You are a careful assistant. Answer using ONLY the provided context.
Cite sources inline like [S1], [S2] matching the order of context snippets.
If the answer is not in the context, say you don't know."""

TEMPLATE = """{question}

# Context
{context}

# Instructions
- Ground your answer strictly in the context
- Add short inline citations like [S1], [S2]
- Be concise but complete
"""

def build_context(snippets: List[str]) -> str:
    lines = []
    for i, s in enumerate(snippets, 1):
        lines.append(f"[S{i}] {s}")
    return "\n".join(lines)

def openai_generate(question: str, contexts: List[str], model: str = "gpt-4o-mini") -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in your environment.")
    client = OpenAI(api_key=api_key)
    ctx = build_context(contexts)
    user = TEMPLATE.format(question=question, context=ctx)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system", "content": SYSTEM_PROMPT},
            {"role":"user", "content": user}
        ],
        temperature=0.2,
        max_tokens=700,
    )
    return resp.choices[0].message.content.strip()
