from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

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
        results = index.search(query=query, top_k=top_k, ollama_base_url=ollama_base_url, ollama_model=ollama_model)
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
    System-message injection: last N memory records (episodic memory).
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


def _summarize_executor_error(executor_output: str, max_chars: int = 900) -> str:
    """
    Make an error snippet short enough for prompt injection.
    We keep the tail part because it often includes stack trace bottom / error type.
    """
    s = (executor_output or "").strip()
    if not s:
        return "Unknown executor error (empty output)."
    # Keep last N chars (often contains the real exception line)
    if len(s) > max_chars:
        s = s[-max_chars:]
    return s


def _progress(message: str) -> None:
    print(f"[workflow] {message}", flush=True)


def _extract_python_code_blocks(content: str) -> str:
    text = content or ""
    blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if not blocks:
        return text.strip()
    return "\n\n".join(b.strip() for b in blocks if b.strip())


def _sanitize_python_code(code: str) -> str:
    lines = []
    for ln in (code or "").splitlines():
        s = ln.strip()
        # Avoid recursive/self execution patterns that break in notebook sandbox.
        if re.search(r"^exec\s*\(\s*open\s*\(.*\)\s*\.read\s*\(\s*\)\s*\)\s*$", s):
            continue
        if re.search(r"^%run\s+", s):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def _state_script_name(step_name: str) -> str:
    mapping = {
        "Explore": "exploration.py",
        "Preprocess": "preprocess.py",
        "Train": "train.py",
        "Summarize": "summary.py",
    }
    return mapping.get(step_name, f"{step_name.lower()}.py")


def _organize_generated_files(run_dir: Path, plot_dir: Path, data_dir: Path) -> None:
    image_exts = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".pdf"}
    data_exts = {".csv", ".parquet", ".json", ".xlsx", ".pkl", ".joblib"}
    reserved_files = {
        "state.json",
        "status.json",
        "run_config.json",
        "run_memory.jsonl",
    }
    reserved_dirs = {"plot", "data", "coding", "logs"}

    for p in run_dir.iterdir():
        if p.is_dir() and p.name in reserved_dirs:
            continue
        if p.is_dir():
            continue
        if p.name in reserved_files:
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


def run_workflow(config: Dict[str, Any], run_dir: Optional[Path] = None, resume: bool = True) -> Dict[str, Any]:
    """
    Minimal end-to-end workflow with checkpoint/resume + RAG + run memory.

    Updated behavior:
    - Train trial failures DO NOT terminate the run.
    - Failed trials are not counted.
    - We record failures and inject a temporary "do-not-repeat" memory into the next Train prompt.
    """
    _ensure_windows_scripts_on_path()
    os.environ.setdefault("KG_WS_PING_INTERVAL_SECS", "0")

    run_dir = Path(run_dir or ".").resolve()
    _progress(f"run started | run_dir={run_dir} | resume={resume}")
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = run_dir / "plot"
    data_dir = run_dir / "data"
    coding_dir = run_dir / "coding"
    logs_dir = run_dir / "logs"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    coding_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    mem = RunMemory(run_dir / "run_memory.jsonl")

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
    max_rounds = int(workflow_cfg.get("max_rounds", 20))  # kept for future, not heavily used now

    # prevent infinite failing in Train
    max_train_failures = int(workflow_cfg.get("max_train_failures", 6))

    executor_backend = str(exec_cfg.get("code_executor_backend", os.environ.get("CODE_EXECUTOR_BACKEND", "local-jupyter"))).strip().lower()
    docker_image = exec_cfg.get("docker_jupyter_image") or os.environ.get("DOCKER_JUPYTER_IMAGE") or None

    llm_config = build_llm_config_from_env()
    client = OpenAIWrapper(config_list=[llm_config])

    rag_context = _build_rag_injector(config, run_dir)

    # memory injection (episodic memory)
    mem_ctx = _memory_context(mem, n=12)

    # ===== temporary per-run memory (for "do not repeat") =====
    # This is separate from mem_ctx: it is guaranteed to reflect this run's latest failures.
    TEMP_MEMORY_LIMIT = 5
    temp_train_failures: List[str] = []
    train_failure_count = 0

    def temp_train_memory_context() -> str:
        if not temp_train_failures:
            return ""
        items = temp_train_failures[-TEMP_MEMORY_LIMIT:]
        joined = "\n\n---\n\n".join(items)
        return (
            "Recent Train Failures (do NOT repeat the same approach):\n"
            + joined
            + "\n\n"
            + "You must change the model choice or preprocessing approach to avoid repeating the same error.\n"
        )

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
- Save ALL plot/image files under: `{plot_dir}`
- Save ALL data/model files (.csv/.json/.pkl/.joblib/.xlsx/.parquet) under: `{data_dir}`
- Save generated scripts under: `{coding_dir}`
- Do NOT save files under project root like `/app`.
"""

    def run_step(agent, step_name: str, prompt: str):
        ckpt.current_state = step_name  # type: ignore
        ckpt.bump_attempt(step_name)    # type: ignore
        save_checkpoint(run_dir, ckpt)
        _progress(f"enter state={step_name}")

        mem.append({"type": "state_enter", "state": step_name})

        chat = initializer.initiate_chat(
            agent,
            message=prompt,
            max_turns=1,  # single agent response; code is executed in next step explicitly
        )
        return chat

    exec_seq = 0

    def execute_code(prev_chat, step_name: str) -> str:
        nonlocal exec_seq
        exec_seq += 1
        content = prev_chat.chat_history[-1]["content"]
        code_text = _sanitize_python_code(_extract_python_code_blocks(content))
        if code_text:
            code_path = coding_dir / f"{exec_seq:03d}_{step_name}.py"
            code_path.write_text(code_text + "\n", encoding="utf-8")
            state_code_path = coding_dir / _state_script_name(step_name)
            state_code_path.write_text(code_text + "\n", encoding="utf-8")

        exec_message = content
        if code_text:
            exec_message = f"```python\n{code_text}\n```"

        exec_chat = initializer.initiate_chat(code_executor, message=exec_message, max_turns=1)
        out = _extract_last_executor_output(exec_chat)
        _organize_generated_files(run_dir=run_dir, plot_dir=plot_dir, data_dir=data_dir)
        return out

    try:
        # Resume logic: decide where to start
        start_state = ckpt.current_state if resume else "Init"
        _progress(f"start_state={start_state} | backend={executor_backend}")

        # --- Explore ---
        if start_state in ["Init", "Explore"]:
            explore_prompt = task_header + "\nPlease explore the dataset first."
            chat1 = run_step(data_explorer, "Explore", explore_prompt)
            out = execute_code(chat1, "Explore")
            ok = ("exitcode: 1" not in out)
            error_snip = _summarize_executor_error(out) if not ok else None
            mem.append({"type": "state_exit", "state": "Explore", "ok": ok, "error_snippet": error_snip})
            if not ok:
                _progress("state=Explore failed")
                raise RuntimeError(f"Explore code execution failed. Executor error snippet:\n{error_snip}")

            _progress("state=Explore done")

            ckpt.current_state = "Preprocess"
            save_checkpoint(run_dir, ckpt)

        # --- Preprocess (loop until LLM says ready) ---
        if ckpt.current_state in ["Preprocess"]:
            while True:
                preprocess_prompt = task_header + "\nPlease preprocess/clean the dataset for training."
                chat2 = run_step(data_processer, "Preprocess", preprocess_prompt)
                out = execute_code(chat2, "Preprocess")
                ok = ("exitcode: 1" not in out)
                error_snip = _summarize_executor_error(out) if not ok else None
                mem.append({"type": "state_exit", "state": "Preprocess", "ok": ok, "error_snippet": error_snip})
                if not ok:
                    _progress("state=Preprocess failed")
                    raise RuntimeError(f"Preprocess code execution failed. Executor error snippet:\n{error_snip}")

                ready = is_ready_for_train(messages=chat2.chat_history, client=client)
                mem.append({"type": "decision", "state": "Preprocess", "ready_for_train": bool(ready)})

                if ready:
                    _progress("state=Preprocess done | ready_for_train=true")
                    ckpt.current_state = "Train"
                    save_checkpoint(run_dir, ckpt)
                    break
                else:
                    _progress("state=Preprocess done | ready_for_train=false -> back to Explore")
                    ckpt.current_state = "Explore"
                    save_checkpoint(run_dir, ckpt)

                    explore_prompt = task_header + "\nDo any additional exploration needed based on preprocessing results."
                    chatx = run_step(data_explorer, "Explore", explore_prompt)
                    outx = execute_code(chatx, "Explore")
                    okx = ("exitcode: 1" not in outx)
                    error_snip_x = _summarize_executor_error(outx) if not okx else None
                    mem.append({"type": "state_exit", "state": "Explore", "ok": okx, "error_snippet": error_snip_x})
                    if not okx:
                        _progress("state=Explore(after preprocess) failed")
                        raise RuntimeError(f"Explore (after preprocess) failed. Executor error snippet:\n{error_snip_x}")

                    _progress("state=Explore(after preprocess) done")

                    ckpt.current_state = "Preprocess"
                    save_checkpoint(run_dir, ckpt)

        # --- Train (fixed number of successful trials; failures do not terminate) ---
        if ckpt.current_state in ["Train"]:
            while ckpt.train_trials_done < train_trials:
                if train_failure_count >= max_train_failures:
                    # Give up to avoid infinite loop, but we still produce memory + checkpoint.
                    mem.append(
                        {
                            "type": "train_abort",
                            "reason": "max_train_failures reached",
                            "max_train_failures": max_train_failures,
                            "train_trials_done": ckpt.train_trials_done,
                        }
                    )
                    raise RuntimeError(f"Too many train failures ({train_failure_count}). Aborting run.")

                trial_no = ckpt.train_trials_done + 1
                _progress(f"state=Train trial={trial_no}/{train_trials}")

                train_prompt = (
                    task_header
                    + "\n"
                    + temp_train_memory_context()
                    + f"\nTrain iteration {trial_no}/{train_trials}: train a model and evaluate. Save plots."
                )

                chat3 = run_step(model_trainer, "Train", train_prompt)
                out = execute_code(chat3, "Train")
                ok = ("exitcode: 1" not in out)

                if ok:
                    _progress(f"state=Train trial={trial_no} succeeded")
                    mem.append({"type": "train_trial", "trial": trial_no, "ok": True})
                    ckpt.train_trials_done += 1
                    save_checkpoint(run_dir, ckpt)
                else:
                    _progress(f"state=Train trial={trial_no} failed")
                    # Failure: do NOT count this trial; stay in Train and try again
                    train_failure_count += 1
                    error_snip = _summarize_executor_error(out)

                    # 1) persistent memory
                    mem.append(
                        {
                            "type": "train_trial",
                            "trial": trial_no,
                            "ok": False,
                            "error_snippet": error_snip,
                        }
                    )

                    # 2) temporary per-run memory to prevent repeating the same approach
                    temp_train_failures.append(
                        f"Trial {trial_no} failed with executor error:\n{error_snip}"
                    )

                    # continue loop without incrementing train_trials_done
                    continue

            ckpt.current_state = "Summarize"
            save_checkpoint(run_dir, ckpt)

        # --- Summarize ---
        if ckpt.current_state in ["Summarize"]:
            _progress("state=Summarize running")
            summary_prompt = task_header + "\nSummarize the workflow and provide final integrated python code."
            chat4 = initializer.initiate_chat(summarizer, message=summary_prompt, max_turns=1)

            saved_train_file = None
            content = chat4.chat_history[-1]["content"]
            if "```python" in content:
                code = content.split("```python")[1].split("```")[0].strip()
                saved_train_file = coding_dir / "train_file_by_agent.py"
                saved_train_file.write_text(code, encoding="utf-8")

            (coding_dir / "chat_history.json").write_text(
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
            _progress("state=Summarize done | workflow completed")

        return {
            "data_path": data_path,
            "target": target,
            "task_type": task_type,
            "executor_backend": executor_backend,
            "run_memory": str(mem.path),
            "checkpoint": str((run_dir / "state.json")),
            "train_failures": train_failure_count,
        }

    finally:
        try:
            server.stop()
        except Exception:
            pass