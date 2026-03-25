from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import autogen

from src.agent.prompts import WORKFLOW_PROMPT_SPECS
from src.agent.tools import ToolRegistry, attach_tools


@dataclass(frozen=True)
class AgentProfile:
    name: str
    prompt: str
    rag_query: Optional[str] = None


def _compose_system_message(profile: AgentProfile, mem_ctx: str, rag_context: Callable[[str], str]) -> str:
    rag_block = rag_context(profile.rag_query) if profile.rag_query else ""
    return (mem_ctx + rag_block + profile.prompt).strip()


def create_initializer() -> autogen.UserProxyAgent:
    return autogen.UserProxyAgent(name="Init", code_execution_config=False)


def create_assistant_agent(
    profile: AgentProfile,
    llm_config: Dict[str, Any],
    mem_ctx: str,
    rag_context: Callable[[str], str],
    tool_registry: Optional[ToolRegistry] = None,
    tool_names: Optional[list[str]] = None,
) -> autogen.AssistantAgent:
    agent = autogen.AssistantAgent(
        name=profile.name,
        llm_config=llm_config,
        system_message=_compose_system_message(profile, mem_ctx=mem_ctx, rag_context=rag_context),
    )
    attach_tools(agent, tool_names=tool_names, registry=tool_registry or {})
    return agent


def create_workflow_agents(
    llm_config: Dict[str, Any],
    mem_ctx: str,
    rag_context: Callable[[str], str],
    tool_registry: Optional[ToolRegistry] = None,
) -> Dict[str, autogen.AssistantAgent]:
    key_map = {
        "Data_Explorer": "data_explorer",
        "Data_Processer": "data_processer",
        "Model_Trainer": "model_trainer",
        "Code_Summarizer": "summarizer",
    }
    agents: Dict[str, autogen.AssistantAgent] = {}

    for spec in WORKFLOW_PROMPT_SPECS:
        key = key_map.get(spec.name)
        if not key:
            continue

        profile = AgentProfile(name=spec.name, prompt=spec.prompt, rag_query=spec.rag_query)
        agents[key] = create_assistant_agent(
            profile=profile,
            llm_config=llm_config,
            mem_ctx=mem_ctx,
            rag_context=rag_context,
            tool_registry=tool_registry,
            tool_names=spec.tool_names,
        )

    missing = [v for v in key_map.values() if v not in agents]
    if missing:
        raise ValueError(f"Missing workflow agent specs for keys: {missing}")

    return agents
