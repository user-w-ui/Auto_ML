from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional


ToolHandler = Callable[..., object]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: ToolHandler


ToolRegistry = Dict[str, ToolSpec]


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
    setattr(agent, "_registered_tool_specs", selected)
