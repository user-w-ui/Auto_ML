from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import autogen
from autogen import OpenAIWrapper
from autogen.coding.jupyter import DockerJupyterServer, JupyterCodeExecutor, LocalJupyterServer

from src.utils import analyze_code_execution_output, count_train_trials, did_code_execution_fail, is_ready_for_train
from src.agent.definition import create_initializer, create_workflow_agents
from src.rag.kb_index import MiniVectorIndex
from src.rag.run_memory import RunMemory
from src.workflow.prompts import CODE_EXECUTOR_SYSTEM_MESSAGE, build_task_prompt


def _ensure_windows_scripts_on_path() -> None:
    """在 Windows 环境下，将当前 Python 的 Scripts 目录加入 PATH。"""
    if os.name == "nt":
        scripts_dir = Path(sys.executable).parent / "Scripts"
        if scripts_dir.exists():
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{scripts_dir}{os.pathsep}{current_path}"


def build_llm_config_from_env() -> Dict[str, Any]:
    """从环境变量构建 LLM 配置，缺少关键密钥时直接报错。"""
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    deepseek_base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    if not deepseek_api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY. Please set it in .env or environment variables.")
    return {
        "api_type": "openai",
        "model": deepseek_model,
        "api_key": deepseek_api_key,
        "base_url": deepseek_base_url,
    }


def _build_rag_injector(config: Dict[str, Any], run_dir: Path):
    """构建 RAG 上下文注入函数；若知识库目录不存在则返回空注入器。"""
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
    # 构建或加载向量索引，避免每次都全量重建。
    try:
        index = MiniVectorIndex.build_or_load(
            kb_dir=kb_dir,
            cache_path=cache_path,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
        )
    except Exception as e:
        # Fail-open: RAG unavailable should not block the workflow.
        _progress(f"RAG disabled: build/load index failed ({type(e).__name__}: {e})")

        def _empty(_q: str) -> str:
            return ""

        return _empty

    def rag_context(query: str) -> str:
        # 检索知识块并格式化成统一上下文，供后续 Agent 提示词拼接。
        try:
            results = index.search(query=query, top_k=top_k, ollama_base_url=ollama_base_url, ollama_model=ollama_model)
        except Exception as e:
            # Fail-open: one retrieval failure should not fail the run.
            _progress(f"RAG retrieval skipped: {type(e).__name__}: {e}")
            return ""

        blocks = []
        for score, chunk in results:
            blocks.append(f"[{chunk.doc_id}#{chunk.chunk_id} score={score:.3f}]\n{chunk.text}")
        return "Relevant Knowledge (RAG):\n" + "\n\n".join(blocks) + "\n"

    return rag_context


def _memory_context(mem: RunMemory, n: int = 12) -> str:
    """提取最近 n 条运行记忆并拼接为提示上下文。"""
    tail = mem.tail(n=n).strip()
    if not tail:
        return ""
    return (
        "Recent Run Memory (JSONL, newest last):\n"
        + tail
        + "\n\n"
        + "Use this memory to avoid repeating failed steps and to reuse what already worked.\n"
    )


def _progress(message: str) -> None:
    """统一工作流日志输出格式，便于终端追踪。"""
    print(f"[workflow] {message}", flush=True)


def _extract_python_code_blocks(content: str) -> str:
    """从 Markdown 文本中提取 ```python``` 代码块并合并返回。"""
    text = content or ""
    blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if not blocks:
        return ""
    return "\n\n".join(b.strip() for b in blocks if b.strip())


def _sanitize_python_code(code: str) -> str:
    """清理在 Notebook 沙盒中容易失败或危险的代码模式。"""
    lines = []
    for ln in (code or "").splitlines():
        s = ln.strip()
        # Avoid recursive/self execution patterns that break in notebook sandbox.
        if re.search(r"^exec\s*\(\s*open\s*\(.*\)\s*\.read\s*\(\s*\)\s*\)\s*$", s):
            continue
        # Skip %run commands in notebook
        if re.search(r"^\s*%run\s+", s):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def _state_script_name(step_name: str) -> str:
    """将工作流状态名映射为固定脚本文件名。"""
    mapping = {
        "Explore": "exploration.py",
        "Preprocess": "preprocess.py",
        "Train": "train.py",
        "Summarize": "summary.py",
    }
    return mapping.get(step_name, f"{step_name.lower()}.py")


def _organize_generated_files(run_dir: Path, plot_dir: Path, data_dir: Path) -> None:
    """将运行目录下散落的输出文件按类型归档到 plot/data 子目录。"""
    image_exts = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".pdf"}
    data_exts = {".csv", ".parquet", ".json", ".xlsx", ".pkl", ".joblib"}
    reserved_dirs = {"plot", "data", "coding", "logs"}
    reserved_files = {"status.json", "run_config.json", "run_memory.jsonl"}

    for p in run_dir.iterdir():
        if p.is_dir() or p.name in reserved_files:
            continue
        if p.parent == run_dir and p.name in reserved_dirs:
            continue

        ext = p.suffix.lower()
        dest = None
        if ext in image_exts:
            dest = plot_dir / p.name
        elif ext in data_exts:
            dest = data_dir / p.name

        if dest is None:
            continue

        if dest.exists():
            # 避免重名覆盖：自动追加递增后缀。
            stem = p.stem
            suffix = p.suffix
            idx = 1
            while True:
                candidate = dest.with_name(f"{stem}_{idx}{suffix}")
                if not candidate.exists():
                    dest = candidate
                    break
                idx += 1

        p.replace(dest)


def run_workflow(config: Dict[str, Any], run_dir: Optional[Path] = None) -> Dict[str, Any]:
    """执行 AutoML 多 Agent 主流程：Explore -> Preprocess -> Train -> Summarize。"""
    _ensure_windows_scripts_on_path()
    os.environ.setdefault("KG_WS_PING_INTERVAL_SECS", "0")

    run_dir = Path(run_dir or ".").resolve()
    _progress(f"run started | run_dir={run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 准备本次运行所需目录结构。
    plot_dir = run_dir / "plot"
    data_dir = run_dir / "data"
    coding_dir = run_dir / "coding"
    logs_dir = run_dir / "logs"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    coding_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    mem = RunMemory(run_dir / "run_memory.jsonl")

    # 读取并规范化配置项，保证缺省值可用。
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    workflow_cfg = config.get("workflow", {}) if isinstance(config.get("workflow", {}), dict) else {}
    exec_cfg = config.get("execution", {}) if isinstance(config.get("execution", {}), dict) else {}

    data_path = str(data_cfg.get("path", "./house_prices_train.csv"))
    target = str(data_cfg.get("target", "SalePrice"))
    task_type = str(data_cfg.get("task_type", "regression"))

    train_trials = int(workflow_cfg.get("train_trials", 2))
    max_rounds = int(workflow_cfg.get("max_rounds", 20))
    preprocess_max_attempts = int(workflow_cfg.get("preprocess_max_attempts", 3))
    train_max_attempts = int(workflow_cfg.get("train_max_attempts", max(train_trials * 3, train_trials)))

    executor_backend = str(exec_cfg.get("code_executor_backend", os.environ.get("CODE_EXECUTOR_BACKEND", "local-jupyter"))).strip().lower()
    docker_image = exec_cfg.get("docker_jupyter_image") or os.environ.get("DOCKER_JUPYTER_IMAGE") or None

    llm_config = build_llm_config_from_env()
    client = OpenAIWrapper(config_list=[llm_config])

    # 组装 RAG 与历史记忆上下文，减少重复试错。
    rag_context = _build_rag_injector(config, run_dir)
    mem_ctx = _memory_context(mem, n=12)

    # 创建初始化器与各阶段 Agent。
    initializer = create_initializer()
    workflow_agents = create_workflow_agents(
        llm_config=llm_config,
        mem_ctx=mem_ctx,
        rag_context=rag_context,
        tool_registry=None,
    )
    data_explorer = workflow_agents["data_explorer"]
    data_processer = workflow_agents["data_processer"]
    model_trainer = workflow_agents["model_trainer"]
    summarizer = workflow_agents["summarizer"]

    # 根据配置选择代码执行后端（本地 Jupyter 或 Docker Jupyter）。
    if executor_backend == "docker-jupyter":
        server = DockerJupyterServer(custom_image_name=docker_image)
    else:
        server = LocalJupyterServer()

    code_executor = autogen.UserProxyAgent(
        name="Code_Executor",
        system_message=CODE_EXECUTOR_SYSTEM_MESSAGE,
        human_input_mode="NEVER",
        code_execution_config={"executor": JupyterCodeExecutor(server, output_dir=coding_dir)},
    )

    task_prompt = build_task_prompt(
        target=target,
        task_type=task_type,
        data_path=data_path,
        plot_dir=str(plot_dir),
        data_dir=str(data_dir),
    )

    exec_seq = 0  # Counter for generated code files
    preprocess_attempts_done = 0
    train_attempts_done = 0

    def save_step_code(step_name: str, content: str) -> None:
        """从 Agent 消息中提取代码并保存（编号文件 + 状态固定文件）。"""
        nonlocal exec_seq
        code_text = _extract_python_code_blocks(content)
        if code_text:
            code_text = _sanitize_python_code(code_text)
            if code_text:
                exec_seq += 1
                # 保存编号脚本，便于按时序回放。
                code_path = coding_dir / f"{exec_seq:03d}_{step_name}.py"
                code_path.write_text(code_text + "\n", encoding="utf-8")
                # 保存状态固定脚本，便于快速定位最新版本。
                state_code_path = coding_dir / _state_script_name(step_name)
                state_code_path.write_text(code_text + "\n", encoding="utf-8")

    def state_transition(last_speaker, groupchat):
        """GroupChat 状态机：根据上一个发言者和执行结果决定下一位发言者。"""
        nonlocal preprocess_attempts_done, train_attempts_done
        messages = groupchat.messages

        if last_speaker is initializer:
            _progress("state=Explore")
            mem.append({"type": "state_enter", "state": "Explore"})
            return data_explorer

        if last_speaker in [data_explorer, data_processer, model_trainer]:
            # 在交给执行器前，先把 Agent 产出的代码落盘。
            agent_content = messages[-1].get("content", "") if messages else ""
            if agent_content:
                if last_speaker is data_explorer:
                    save_step_code("Explore", agent_content)
                elif last_speaker is data_processer:
                    save_step_code("Preprocess", agent_content)
                elif last_speaker is model_trainer:
                    save_step_code("Train", agent_content)
            return code_executor

        if last_speaker is code_executor:
            if len(messages) < 2:
                return data_explorer

            last_worker = messages[-2].get("name")
            executor_output = messages[-1].get("content", "") or ""

            if last_worker == "Data_Processer":
                preprocess_attempts_done += 1
                mem.append(
                    {
                        "type": "attempt",
                        "state": "Preprocess",
                        "attempt": preprocess_attempts_done,
                        "max_attempts": preprocess_max_attempts,
                    }
                )
            elif last_worker == "Model_Trainer":
                train_attempts_done += 1
                mem.append(
                    {
                        "type": "attempt",
                        "state": "Train",
                        "attempt": train_attempts_done,
                        "max_attempts": train_max_attempts,
                    }
                )

            # 执行失败时回到对应状态重试。
            if did_code_execution_fail(executor_output):
                failure_info = analyze_code_execution_output(executor_output)
                if last_worker == "Data_Processer" and preprocess_attempts_done >= preprocess_max_attempts:
                    mem.append(
                        {
                            "type": "state_exit",
                            "state": last_worker,
                            "ok": False,
                            "failure": {
                                "exit_code": failure_info.get("exit_code"),
                                "error_type": failure_info.get("error_type"),
                                "error_message": failure_info.get("error_message"),
                                "traceback": failure_info.get("traceback"),
                            },
                        }
                    )
                    mem.append({"type": "decision", "state": "Preprocess", "force_to_train": True, "reason": "max_attempts_reached"})
                    _progress("state=Preprocess failed | max attempts reached -> Train")
                    mem.append({"type": "state_enter", "state": "Train"})
                    return model_trainer

                if last_worker == "Model_Trainer" and train_attempts_done >= train_max_attempts:
                    mem.append(
                        {
                            "type": "state_exit",
                            "state": last_worker,
                            "ok": False,
                            "failure": {
                                "exit_code": failure_info.get("exit_code"),
                                "error_type": failure_info.get("error_type"),
                                "error_message": failure_info.get("error_message"),
                                "traceback": failure_info.get("traceback"),
                            },
                        }
                    )
                    mem.append({"type": "decision", "state": "Train", "force_to_summarize": True, "reason": "max_attempts_reached"})
                    _progress("state=Train failed | max attempts reached -> Summarize")
                    mem.append({"type": "state_enter", "state": "Summarize"})
                    return summarizer

                mem.append(
                    {
                        "type": "state_exit",
                        "state": last_worker,
                        "ok": False,
                        "failure": {
                            "exit_code": failure_info.get("exit_code"),
                            "error_type": failure_info.get("error_type"),
                            "error_message": failure_info.get("error_message"),
                            "traceback": failure_info.get("traceback"),
                        },
                    }
                )
                _progress(f"state={last_worker} failed -> retry")
                return groupchat.agent_by_name(last_worker)

            # Explore 成功后进入 Preprocess。
            if last_worker == "Data_Explorer":
                mem.append({"type": "state_exit", "state": "Explore", "ok": True})
                _progress("state=Explore done -> Preprocess")
                mem.append({"type": "state_enter", "state": "Preprocess"})
                return data_processer

            # Preprocess 后通过判定函数决定进入 Train 或回到 Explore。
            if last_worker == "Data_Processer":
                mem.append({"type": "state_exit", "state": "Preprocess", "ok": True})
                ready = is_ready_for_train(groupchat=groupchat, client=client)
                mem.append({"type": "decision", "state": "Preprocess", "ready_for_train": bool(ready)})
                if ready:
                    _progress("state=Preprocess done | ready_for_train=true -> Train")
                    mem.append({"type": "state_enter", "state": "Train"})
                    return model_trainer

                if preprocess_attempts_done >= preprocess_max_attempts:
                    mem.append({"type": "decision", "state": "Preprocess", "force_to_train": True, "reason": "max_attempts_reached"})
                    _progress("state=Preprocess done | ready_for_train=false | max attempts reached -> Train")
                    mem.append({"type": "state_enter", "state": "Train"})
                    return model_trainer

                _progress("state=Preprocess done | ready_for_train=false -> Explore")
                mem.append({"type": "state_enter", "state": "Explore"})
                return data_explorer

            # Train 阶段按试验次数循环，达到阈值后进入 Summarize。
            if last_worker == "Model_Trainer":
                mem.append({"type": "state_exit", "state": "Train", "ok": True})
                trials_done = count_train_trials(groupchat)
                _progress(f"state=Train progress {trials_done}/{train_trials} | attempts={train_attempts_done}/{train_max_attempts}")

                if train_attempts_done >= train_max_attempts and trials_done < train_trials:
                    mem.append({"type": "decision", "state": "Train", "force_to_summarize": True, "reason": "max_attempts_reached"})
                    _progress("state=Train max attempts reached -> Summarize")
                    mem.append({"type": "state_enter", "state": "Summarize"})
                    return summarizer

                if trials_done < train_trials:
                    mem.append({"type": "state_enter", "state": "Train"})
                    return model_trainer
                _progress("state=Train done -> Summarize")
                mem.append({"type": "state_enter", "state": "Summarize"})
                return summarizer

        if last_speaker is summarizer:
            mem.append({"type": "state_exit", "state": "Summarize", "ok": True})
            # 结束前保存总结阶段产出的最终代码。
            if messages:
                summarizer_content = messages[-1].get("content", "")
                if summarizer_content:
                    save_step_code("Summarize", summarizer_content)
            _progress("state=Summarize done | workflow completed")
            return None

        return None

    groupchat = autogen.GroupChat(
        agents=[initializer, data_explorer, data_processer, model_trainer, summarizer, code_executor],
        messages=[],
        max_round=max_rounds,
        speaker_selection_method=state_transition,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=None)

    chat_result = None
    try:
        # 启动多 Agent 对话主循环。
        chat_result = initializer.initiate_chat(manager, message=task_prompt)
        # 对输出产物做归档整理。
        _organize_generated_files(run_dir=run_dir, plot_dir=plot_dir, data_dir=data_dir)

        saved_train_file = None
        if chat_result and chat_result.chat_history:
            final_content = chat_result.chat_history[-1].get("content", "")
            code = _extract_python_code_blocks(final_content)
            if code:
                saved_train_file = coding_dir / "train_file_by_agent.py"
                saved_train_file.write_text(code + "\n", encoding="utf-8")

            (coding_dir / "chat_history.json").write_text(
                json.dumps(chat_result.chat_history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return {
            "data_path": data_path,
            "target": target,
            "task_type": task_type,
            "executor_backend": executor_backend,
            "run_memory": str(mem.path),
            "train_trials_done": count_train_trials(groupchat),
            "preprocess_attempts_done": preprocess_attempts_done,
            "preprocess_max_attempts": preprocess_max_attempts,
            "train_attempts_done": train_attempts_done,
            "train_max_attempts": train_max_attempts,
            "saved_train_file": str(saved_train_file) if saved_train_file else None,
        }
    finally:
        # 无论流程是否报错，都尝试优雅关闭执行服务器。
        try:
            server.stop()
        except Exception:
            pass
