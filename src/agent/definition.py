from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import autogen

from src.agent.prompts import WORKFLOW_PROMPT_SPECS
from src.agent.tools import ToolRegistry, attach_tools


@dataclass(frozen=True)
class AgentProfile:
    name: str
    prompt: str
    tool_names: Optional[list[str]] = None


def _compose_system_message(profile: AgentProfile, mem_ctx: str, tool_registry: Optional[ToolRegistry]) -> str:
    tool_block = ""
    if profile.tool_names:
        lines = []
        for name in profile.tool_names:
            spec = (tool_registry or {}).get(name)
            if spec:
                lines.append(f"- {spec.name}: {spec.description}")
        if lines:
            tool_block = "\n\nAvailable tools:\n" + "\n".join(lines) + "\n"
    return (mem_ctx + profile.prompt + tool_block).strip()


def create_initializer() -> autogen.UserProxyAgent:
    return autogen.UserProxyAgent(name="Init", code_execution_config=False)


def create_assistant_agent(
    profile: AgentProfile,
    llm_config: Dict[str, Any],
    mem_ctx: str,
    tool_registry: Optional[ToolRegistry] = None,
) -> autogen.AssistantAgent:
    agent = autogen.AssistantAgent(
        name=profile.name,
        llm_config=llm_config,
        system_message=_compose_system_message(profile, mem_ctx=mem_ctx, tool_registry=tool_registry),
    )
    attach_tools(agent, tool_names=profile.tool_names, registry=tool_registry or {})
    return agent


def create_workflow_agents(
    llm_config: Dict[str, Any],
    mem_ctx: str,
    tool_registry: Optional[ToolRegistry] = None,
) -> Dict[str, autogen.AssistantAgent]:
    key_map = {
        "Data_Explorer": "data_explorer",
        "Data_Processer": "data_processer",
        "Model_Trainer": "model_trainer",
        "Code_Summarizer": "summarizer",
        "Evaluator": "evaluator",
    }
    agents: Dict[str, autogen.AssistantAgent] = {}

    for spec in WORKFLOW_PROMPT_SPECS:
        key = key_map.get(spec.name)
        if not key:
            continue

        profile = AgentProfile(name=spec.name, prompt=spec.prompt, tool_names=spec.tool_names)
        agents[key] = create_assistant_agent(
            profile=profile,
            llm_config=llm_config,
            mem_ctx=mem_ctx,
            tool_registry=tool_registry,
        )

    missing = [v for v in key_map.values() if v not in agents]
    if missing:
        raise ValueError(f"Missing workflow agent specs for keys: {missing}")

    return agents
