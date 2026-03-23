from __future__ import annotations

from typing import List
import requests


def ollama_embed(
    texts: List[str],
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
    timeout: int = 60,
) -> List[List[float]]:
    """
    Embed a list of texts via Ollama (local, free).

    Uses POST /api/embeddings:
      {"model": "...", "prompt": "..."}
    """
    vectors: List[List[float]] = []
    url = f"{base_url.rstrip('/')}/api/embeddings"

    for t in texts:
        resp = requests.post(url, json={"model": model, "prompt": t}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        vectors.append(data["embedding"])
    return vectors