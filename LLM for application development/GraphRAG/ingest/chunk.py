
import re
from typing import List, Tuple

def split_md(text: str, max_words: int = 300) -> List[str]:
    """Heading-aware splitter with word budget."""
    parts = re.split(r'(?m)^#{1,6}\s.*$', text)
    chunks, buf, count = [], [], 0
    for part in parts:
        words = part.strip().split()
        for w in words:
            buf.append(w)
            count += 1
            if count >= max_words:
                chunks.append(" ".join(buf).strip())
                buf, count = [], 0
    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c]

def chunk_corpus(texts: List[str], max_words: int = 300) -> Tuple[List[str], List[dict]]:
    chunks, meta = [], []
    for i, t in enumerate(texts):
        cs = split_md(t, max_words=max_words)
        for j, c in enumerate(cs):
            chunks.append(c)
            meta.append({"doc_id": i, "chunk_id": j})
    return chunks, meta
