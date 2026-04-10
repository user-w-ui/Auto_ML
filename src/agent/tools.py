from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.rag.embeddings_ollama import ollama_embed


ToolHandler = Callable[..., object]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: ToolHandler


ToolRegistry = Dict[str, ToolSpec]


def _split_model_cards(markdown_text: str) -> List[str]:
    """Split model-card markdown into cards by headings like `## 1) Model Name`."""
    text = markdown_text.replace("\r\n", "\n")
    heading_re = re.compile(r"^##\s+\d+\)\s+.+$", re.MULTILINE)
    matches = list(heading_re.finditer(text))
    if not matches:
        return [text.strip()] if text.strip() else []

    cards: List[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if block:
            cards.append(block)
    return cards


def _extract_labels_text(card_text: str) -> str:
    """Extract label tokens from a card's '**labels**' line and join as one text."""
    label_line_re = re.compile(r"^\s*-\s*\*\*labels\*\*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    token_re = re.compile(r"`([^`]+)`")
    m = label_line_re.search(card_text)
    if not m:
        return ""
    tokens = [t.strip() for t in token_re.findall(m.group(1)) if t.strip()]
    return " ".join(tokens)


def _cosine_similarity(u: List[float], v: List[float]) -> float:
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


def build_model_card_rag_tools(kb_dir: Path | str, card_file_name: str = "ml_model_cards_for_rag.md") -> List[ToolSpec]:
    """
    Build RAG reader tools for model cards.

    The target markdown is parsed by second-level headings, where each heading block is one model card.
    """
    card_path = Path(kb_dir) / card_file_name

    def read_model_cards_for_rag(model_keyword: str = "", top_k: int = 5) -> str:
        """
        Read model cards from kb/ml_model_cards_for_rag.md.

        Args:
            model_keyword: Query text used for label-embedding similarity retrieval.
            top_k: Max number of cards to return (default 5).
        """
        if not card_path.exists():
            return f"RAG model card file not found: {card_path}"

        text = card_path.read_text(encoding="utf-8")
        cards = _split_model_cards(text)
        if not cards:
            return "No model cards found in RAG file."

        keyword = (model_keyword or "").strip().lower()
        if not keyword:
            return "model_keyword is required for label-vector retrieval."

        label_texts = [_extract_labels_text(card) for card in cards]
        if not any(label_texts):
            return "No labels found in model cards; cannot run label-vector retrieval."

        embed_base_url = os.environ.get("RAG_OLLAMA_BASE_URL", "http://localhost:11434")
        embed_model = os.environ.get("RAG_OLLAMA_EMBED_MODEL", "nomic-embed-text")

        try:
            card_vectors = ollama_embed(label_texts, model=embed_model, base_url=embed_base_url)
            query_vector = ollama_embed([keyword], model=embed_model, base_url=embed_base_url)[0]
            scored: List[tuple[float, str]] = []
            for vec, card in zip(card_vectors, cards):
                scored.append((_cosine_similarity(query_vector, vec), card))
            scored.sort(key=lambda x: x[0], reverse=True)
            k = max(1, int(top_k))
            selected = scored[:k]
            rendered = []
            for score, card in selected:
                rendered.append(f"[label_similarity={score:.3f}]\n{card}")
            return "\n\n---\n\n".join(rendered)
        except Exception:
            # Fallback keeps workflow usable when embedding service is unavailable.
            matches = [card for card in cards if keyword in card.lower()]
            if not matches:
                return f"No model cards matched keyword: {model_keyword}"
            return "\n\n---\n\n".join(matches[: max(1, int(top_k))])

    return [
        ToolSpec(
            name="read_model_cards_for_rag",
            description=(
                "Read ML model cards from kb/ml_model_cards_for_rag.md. "
                "Cards are split by '## n) ...' headings. "
                "Retrieve top model cards by vector similarity between model_keyword and card labels."
            ),
            handler=read_model_cards_for_rag,
        )
    ]


def create_tool_registry(tools: Optional[Iterable[ToolSpec]] = None) -> ToolRegistry:
    registry: ToolRegistry = {}
    for tool in tools or []:
        registry[tool.name] = tool
    return registry


def attach_tools(agent: object, tool_names: Optional[List[str]], registry: ToolRegistry) -> None:
    """
    Keep tool wiring in one place.

    Today this only validates names and stores metadata on the agent object.
    Later we can switch this to real AutoGen tool registration in one function.
    """
    if not tool_names:
        return

    missing = [name for name in tool_names if name not in registry]
    if missing:
        raise ValueError(f"Unknown tools requested: {missing}")

    selected = [registry[name] for name in tool_names]

    # Best-effort native registration for AutoGen function-calling capable agents.
    register_for_llm: Any = getattr(agent, "register_for_llm", None)
    if callable(register_for_llm):
        for spec in selected:
            try:
                decorator: Any = register_for_llm(name=spec.name, description=spec.description)
                if callable(decorator):
                    decorator(spec.handler)
            except Exception:
                # Keep metadata fallback to avoid breaking existing workflow.
                pass

    setattr(agent, "_registered_tool_specs", selected)


def create_rag_tool_registry(kb_dir: Path | str, card_file_name: str = "ml_model_cards_for_rag.md") -> ToolRegistry:
    """Create tool registry for RAG-related tool handlers."""
    return create_tool_registry(build_model_card_rag_tools(kb_dir=kb_dir, card_file_name=card_file_name))
