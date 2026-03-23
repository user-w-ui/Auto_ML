from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from src.rag.embeddings_ollama import ollama_embed


@dataclass(frozen=True)
class KBChunk:
    doc_id: str
    chunk_id: int
    text: str


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    """
    Simple & robust chunking for mini KB.
    """
    text = text.replace("\r\n", "\n")
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        chunks.append(text[i:j].strip())
        if j == n:
            break
        i = max(0, j - overlap)
    return [c for c in chunks if c]


def cosine(u: List[float], v: List[float]) -> float:
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


class MiniVectorIndex:
    """
    Local mini vector index with JSON cache.
    Good enough for:
    - kb/*.md (a small set of docs)
    - no external DB
    - fast iteration
    """

    def __init__(self, chunks: List[KBChunk], vectors: List[List[float]]):
        self.chunks = chunks
        self.vectors = vectors

    @staticmethod
    def _load_kb_chunks(kb_dir: Path) -> List[KBChunk]:
        kb_dir = Path(kb_dir)
        md_files = sorted([p for p in kb_dir.glob("*.md") if p.is_file()])
        chunks: List[KBChunk] = []
        for p in md_files:
            doc_id = p.name
            parts = chunk_text(p.read_text(encoding="utf-8"))
            for idx, part in enumerate(parts):
                chunks.append(KBChunk(doc_id=doc_id, chunk_id=idx, text=part))
        return chunks

    @staticmethod
    def build_or_load(
        kb_dir: Path,
        cache_path: Path,
        ollama_base_url: str,
        ollama_model: str,
    ) -> "MiniVectorIndex":
        chunks = MiniVectorIndex._load_kb_chunks(kb_dir)

        cache_path = Path(cache_path)
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            # very simple cache validity check:
            if (
                data.get("ollama_model") == ollama_model
                and data.get("kb_dir") == str(Path(kb_dir).resolve())
                and data.get("num_chunks") == len(chunks)
            ):
                return MiniVectorIndex(chunks=chunks, vectors=data["vectors"])

        texts = [c.text for c in chunks]
        vectors = ollama_embed(texts, model=ollama_model, base_url=ollama_base_url)

        payload = {
            "kb_dir": str(Path(kb_dir).resolve()),
            "ollama_model": ollama_model,
            "num_chunks": len(chunks),
            "vectors": vectors,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")

        return MiniVectorIndex(chunks=chunks, vectors=vectors)

    def search(
        self,
        query: str,
        top_k: int,
        ollama_base_url: str,
        ollama_model: str,
    ) -> List[Tuple[float, KBChunk]]:
        qvec = ollama_embed([query], model=ollama_model, base_url=ollama_base_url)[0]
        scored: List[Tuple[float, KBChunk]] = []
        for vec, chunk in zip(self.vectors, self.chunks):
            scored.append((cosine(qvec, vec), chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]