from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import autogen
from autogen import OpenAIWrapper
from autogen.coding.jupyter import DockerJupyterServer, JupyterCodeExecutor, LocalJupyterServer

from utils import is_ready_for_train
from src.rag.kb_index import MiniVectorIndex
from src.rag.run_memory import RunMemory
from src.workflow.checkpoint import load_or_init_checkpoint, save_checkpoint


def _ensure_windows_scripts_on_path() -> None:
    if os.name == "nt":
        scripts_dir = Path(sys.executable).parent / "Scripts"
        if scripts_dir.exists():
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{scripts_dir}{os.pathsep}{current_path}"


def build_llm_config_from_env() -> Dict[str, Any]:
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
    index = MiniVectorIndex.build_or_load(
        kb_dir=kb_dir,
        cache_path=cache_path,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
    )

    def rag_context(query: str) -> str:
        results = index.search(
            query=query,
            top_k=top_k,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
        )
        blocks = []
        for score, chunk in results:
            blocks.append(f"[{chunk.doc_id}#{chunk.chunk_id} score={score:.3f}]\n{chunk.text}")
        return "Relevant Knowledge (RAG):\n" + "\n\n".join(blocks) + "\n"

    return rag_context


def _extract_last_executor_output(chat_result) -> str:
    """
    Find the last message from Code_Executor and return its content.
    """
    if not chat_result:
        return ""
    for msg in reversed(chat_result.chat_history):
        if msg.get("name") == "Code_Executor":
            return msg.get("content") or ""
    return ""


def _memory_context(mem: RunMemory, n: int = 12) -> str:
    """
    System-message injection: last N memory records.

    This is a lightweight "episodic memory" injection:
    - We don't do extra retrieval yet.
    - Just provide recent run events to reduce repetition / improve recovery.
    """
    tail = mem.tail(n=n).strip()
    if not tail:
        return ""
    return (
        "Recent Run Memory (JSONL, newest last):\n"
        + tail
        + "\n\n"
        + "Use this memory to avoid repeating failed steps and to reuse what already worked.\n"
    )


def run_workflow(config: Dict[str, Any], run_dir: Optional[Path] = None, resume: bool = True) -> Dict[str, Any]:
    """
    Minimal end-to-end workflow with REAL checkpoint/resume.

    Key skills learned here:
    - state machine (explicit states)
    - recoverable (checkpoint + resume)
    - RAG knowledge injection (embedding)
    - run memory (append-only JSONL)

    NEW in this version:
    - system_message injection of recent run memory (episodic memory)
    """
    _ensure_windows_scripts_on_path()
    os.environ.setdefault("KG_WS_PING_INTERVAL_SECS", "0")

    run_dir = Path(run_dir or ".").resolve()
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    mem = RunMemory(artifacts_dir / "run_memory.jsonl")

    # Load/Init checkpoint
    run_id = run_dir.name  # cheap run_id inference
    ckpt = load_or_init_checkpoint(run_id=run_id, run_dir=run_dir)

    # Config
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    workflow_cfg = config.get("workflow", {}) if isinstance(config.get("workflow", {}), dict) else {}
    exec_cfg = config.get("execution", {}) if isinstance(config.get("execution", {}), dict) else {}

    data_path = str(data_cfg.get("path", "./house_prices_train.csv"))
    target = str(data_cfg.get("target", "SalePrice"))
    task_type = str(data_cfg.get("task_type", "regression"))

    train_trials = int(workflow_cfg.get("train_trials", 2))
    max_rounds = int(workflow_cfg.get("max_rounds", 20))  # not heavily used now, kept for future

    executor_backend = str(exec_cfg.get("code_executor_backend", os.environ.get("CODE_EXECUTOR_BACKEND", "local-jupyter"))).strip().lower()
    docker_image = exec_cfg.get("docker_jupyter_image") or os.environ.get("DOCKER_JUPYTER_IMAGE") or None

    llm_config = build_llm_config_from_env()
    client = OpenAIWrapper(config_list=[llm_config])

    rag_context = _build_rag_injector(config, run_dir)

    # ====== memory injection (computed once per run init; light + stable) ======
    # Note: we compute it once here (not re-reading each step), to avoid excessive file reads.
    # If you want "live memory" injection, call _memory_context(mem) inside run_step instead.
    mem_ctx = _memory_context(mem, n=12)

    # Agents
    initializer = autogen.UserProxyAgent(name="Init", code_execution_config=False)

    data_explorer = autogen.AssistantAgent(
        name="Data_Explorer",
        llm_config=llm_config,
        system_message=(
            mem_ctx
            + rag_context("tabular dataset exploration checklist, missing values, target distribution, plots")
            + """
You are the data explorer. Write code to explore dataset characteristics (shape, head, info/describe, missing values, target distribution, basic plots).
Do NOT train models.
If you think the data is ready and no more exploration is needed, reply exactly: Ready for training
""".strip()
        ),
    )

    data_processer = autogen.AssistantAgent(
        name="Data_Processer",
        llm_config=llm_config,
        system_message=(
            mem_ctx
            + rag_context("tabular preprocessing checklist: missing values, categorical encoding, leakage prevention, pipeline")
            + """
You are the data processer. Clean/prepare the dataset for model training.
Handle missing values, encode categorical variables, scale when helpful.
Avoid inplace=True.
""".strip()
        ),
    )

    model_trainer = autogen.AssistantAgent(
        name="Model_Trainer",
        llm_config=llm_config,
        system_message=(
            mem_ctx
            + rag_context("tabular regression modeling, baselines, boosting models, metrics, residual plots")
            + """
You are the model trainer. Train ONE model per iteration.
Use 70/30 train/test split. Evaluate and save plots as images.
Try a different model or different hyperparameters each iteration. No grid search.
""".strip()
        ),
    )

    summarizer = autogen.AssistantAgent(
        name="Code_Summarizer",
        llm_config=llm_config,
        system_message=(
            mem_ctx
            + rag_context("write a concise ML report with metrics table and plot references")
            + """
You are the code summarizer. Integrate all error-free code into a single runnable snippet.
Summarize exploration, preprocessing, training, and conclude best model.
""".strip()
        ),
    )

    # Code executor
    coding_dir = artifacts_dir / "coding"
    coding_dir.mkdir(exist_ok=True)

    if executor_backend == "docker-jupyter":
        server = DockerJupyterServer(custom_image_name=docker_image)
    else:
        server = LocalJupyterServer()

    code_executor = autogen.UserProxyAgent(
        name="Code_Executor",
        system_message="Executor. Execute the code written by the Coder and report the result.",
        human_input_mode="NEVER",
        code_execution_config={"executor": JupyterCodeExecutor(server, output_dir=coding_dir)},
    )

    task_header = f"""Task:
- Predict target: `{target}`
- Task type: {task_type}
- Dataset path: `{data_path}`
Environment:
- Code executes in Jupyter; previous states are saved.
- Save plots as image files.
"""

    def run_step(agent, step_name: str, prompt: str):
        ckpt.current_state = step_name  # type: ignore
        ckpt.bump_attempt(step_name)  # type: ignore
        save_checkpoint(run_dir, ckpt)

        mem.append({"type": "state_enter", "state": step_name})

        chat = initializer.initiate_chat(
            agent,
            message=prompt,
            max_turns=1,  # single agent response; code is executed in next step explicitly
        )
        return chat

    def execute_code(prev_chat) -> str:
        # Execute code by sending previous agent content to Code_Executor
        content = prev_chat.chat_history[-1]["content"]
        exec_chat = initializer.initiate_chat(code_executor, message=content, max_turns=1)
        out = _extract_last_executor_output(exec_chat)
        return out

    try:
        # Resume logic: decide where to start
        start_state = ckpt.current_state if resume else "Init"

        # --- Explore ---
        if start_state in ["Init", "Explore"]:
            explore_prompt = task_header + "\nPlease explore the dataset first."
            chat1 = run_step(data_explorer, "Explore", explore_prompt)
            out = execute_code(chat1)
            ok = ("exitcode: 1" not in out)
            mem.append({"type": "state_exit", "state": "Explore", "ok": ok})
            if not ok:
                raise RuntimeError("Explore code execution failed (see executor output).")

            ckpt.current_state = "Preprocess"
            save_checkpoint(run_dir, ckpt)

        # --- Preprocess (loop until LLM says ready) ---
        if ckpt.current_state in ["Preprocess"]:
            while True:
                preprocess_prompt = task_header + "\nPlease preprocess/clean the dataset for training."
                chat2 = run_step(data_processer, "Preprocess", preprocess_prompt)
                out = execute_code(chat2)
                ok = ("exitcode: 1" not in out)
                mem.append({"type": "state_exit", "state": "Preprocess", "ok": ok})
                if not ok:
                    raise RuntimeError("Preprocess code execution failed (see executor output).")

                # Ask decision model (same as your previous is_ready_for_train)
                ready = is_ready_for_train(
                    groupchat=type("X", (), {"messages": chat2.chat_history})(),  # minimal adapter
                    client=client,
                )
                mem.append({"type": "decision", "state": "Preprocess", "ready_for_train": bool(ready)})

                if ready:
                    ckpt.current_state = "Train"
                    save_checkpoint(run_dir, ckpt)
                    break
                else:
                    # go back to Explore once, to avoid infinite preprocess loop
                    ckpt.current_state = "Explore"
                    save_checkpoint(run_dir, ckpt)

                    explore_prompt = task_header + "\nDo any additional exploration needed based on preprocessing results."
                    chatx = run_step(data_explorer, "Explore", explore_prompt)
                    outx = execute_code(chatx)
                    okx = ("exitcode: 1" not in outx)
                    mem.append({"type": "state_exit", "state": "Explore", "ok": okx})
                    if not okx:
                        raise RuntimeError("Explore (after preprocess) failed.")

                    ckpt.current_state = "Preprocess"
                    save_checkpoint(run_dir, ckpt)

        # --- Train (fixed number of trials) ---
        if ckpt.current_state in ["Train"]:
            while ckpt.train_trials_done < train_trials:
                trial_no = ckpt.train_trials_done + 1
                train_prompt = task_header + f"\nTrain iteration {trial_no}/{train_trials}: train a model and evaluate. Save plots."
                chat3 = run_step(model_trainer, "Train", train_prompt)
                out = execute_code(chat3)
                ok = ("exitcode: 1" not in out)
                mem.append({"type": "train_trial", "trial": trial_no, "ok": ok})
                if not ok:
                    # don't count failed trial
                    raise RuntimeError(f"Train trial {trial_no} failed (see executor output).")

                ckpt.train_trials_done += 1
                save_checkpoint(run_dir, ckpt)

            ckpt.current_state = "Summarize"
            save_checkpoint(run_dir, ckpt)

        # --- Summarize ---
        if ckpt.current_state in ["Summarize"]:
            summary_prompt = task_header + "\nSummarize the workflow and provide final integrated python code."
            chat4 = initializer.initiate_chat(summarizer, message=summary_prompt, max_turns=1)

            # Save integrated code if present
            saved_train_file = None
            content = chat4.chat_history[-1]["content"]
            if "```python" in content:
                code = content.split("```python")[1].split("```")[0].strip()
                saved_train_file = artifacts_dir / "train_file_by_agent.py"
                saved_train_file.write_text(code, encoding="utf-8")

            (artifacts_dir / "chat_history.json").write_text(
                __import__("json").dumps(chat4.chat_history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            mem.append(
                {
                    "type": "state_exit",
                    "state": "Summarize",
                    "ok": True,
                    "saved_train_file": str(saved_train_file) if saved_train_file else None,
                }
            )

            ckpt.current_state = "End"
            save_checkpoint(run_dir, ckpt)

        return {
            "data_path": data_path,
            "target": target,
            "task_type": task_type,
            "executor_backend": executor_backend,
            "run_memory": str(mem.path),
            "checkpoint": str((run_dir / "state.json")),
        }

    finally:
        try:
            server.stop()
        except Exception:
            pass