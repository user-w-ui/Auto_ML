from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

from src.rag.kb_index import MiniVectorIndex


def build_rag_injector(
    config: Dict[str, Any],
    run_dir: Path,
    progress: Callable[[str], None],
) -> Callable[[str], str]:
    """Build RAG context injector; return empty injector when unavailable."""
    kb_dir = Path(config.get("rag", {}).get("kb_dir", "kb"))
    if not kb_dir.exists():

        def _empty(_q: str) -> str:
            return ""

        return _empty

    rag_cfg = config.get("rag", {})
    ollama_base_url = str(rag_cfg.get("ollama_base_url", "http://localhost:11434"))
    ollama_model = str(rag_cfg.get("ollama_model", "nomic-embed-text"))
    top_k = int(rag_cfg.get("top_k", 4))

    cache_path = run_dir / "kb_index.json"
    try:
        index = MiniVectorIndex.build_or_load(
            kb_dir=kb_dir,
            cache_path=cache_path,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
        )
    except Exception as e:
        progress(f"RAG disabled: build/load index failed ({type(e).__name__}: {e})")

        def _empty(_q: str) -> str:
            return ""

        return _empty

    def rag_context(query: str) -> str:
        try:
            results = index.search(query=query, top_k=top_k, ollama_base_url=ollama_base_url, ollama_model=ollama_model)
        except Exception as e:
            progress(f"RAG retrieval skipped: {type(e).__name__}: {e}")
            return ""

        blocks = []
        for score, chunk in results:
            blocks.append(f"[{chunk.doc_id}#{chunk.chunk_id} score={score:.3f}]\n{chunk.text}")
        return "Relevant Knowledge (RAG):\n" + "\n\n".join(blocks) + "\n"

    return rag_context
