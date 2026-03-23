"""
Minimal callable workflow.

目标：
- 只是把它包成函数，方便 CLI/runner 调用
- 后续我们往这里塞 RAG 注入、memory 注入、工具调用
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import autogen
from autogen import OpenAIWrapper
from autogen.coding.jupyter import DockerJupyterServer, JupyterCodeExecutor, LocalJupyterServer

from utils import is_ready_for_train, count_train_trials


def _ensure_windows_scripts_on_path() -> None:
    """你之前 main.py 的 Windows PATH 修复，保留。"""
    if os.name == "nt":
        scripts_dir = Path(sys.executable).parent / "Scripts"
        if scripts_dir.exists():
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{scripts_dir}{os.pathsep}{current_path}"


def build_llm_config_from_env() -> Dict[str, Any]:
    """保持你现在的 DeepSeek OpenAI-compatible 配置方式。"""
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


def run_workflow(config: Dict[str, Any], run_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run the multi-agent AutoML workflow.

    Returns a small dict summary for the runner to record.
    """
    _ensure_windows_scripts_on_path()

    # prevent kernel gateway websocket ping timeout during long-running code blocks.
    os.environ.setdefault("KG_WS_PING_INTERVAL_SECS", "0")

    run_dir = run_dir or Path(".")
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Read task/data from config.yaml
    data_path = config.get("data", {}).get("path", "./house_prices_train.csv")
    target = config.get("data", {}).get("target", "SalePrice")
    task_type = config.get("data", {}).get("task_type", "regression")

    train_trials = int(config.get("workflow", {}).get("train_trials", 2))
    max_rounds = int(config.get("workflow", {}).get("max_rounds", 20))

    llm_config = build_llm_config_from_env()
    config_list = [llm_config]

    initializer = autogen.UserProxyAgent(name="Init", code_execution_config=False)

    data_explorer = autogen.AssistantAgent(
        name="Data_Explorer",
        llm_config=llm_config,
        system_message=(
            "You are the data explorer. Given a dataset and a task, write code to explore the dataset.\n"
            "Focus on: shape, head, df.info/describe, missing values, target distribution, and necessary plots.\n"
            'If you think the data is ready and no more exploration is needed, reply with "Ready for training".'
        ),
    )

    data_processer = autogen.AssistantAgent(
        name="Data_Processer",
        llm_config=llm_config,
        system_message=(
            "You are the data processer. Clean/prepare data for model training.\n"
            "Handle missing values, encode categorical features, scale numerical features when needed.\n"
            "Avoid inplace=True; assign to new variables.\n"
        ),
    )

    model_trainer = autogen.AssistantAgent(
        name="Model_Trainer",
        llm_config=llm_config,
        system_message=(
            "You are the model trainer. Train one model per iteration, evaluate on a 70/30 train/test split.\n"
            "Try different models or hyperparameters across iterations (no grid search).\n"
            "Save performance visualizations as images.\n"
        ),
    )

    summarizer = autogen.AssistantAgent(
        name="Code_Summarizer",
        llm_config=llm_config,
        system_message=(
            "You are the code summarizer. Integrate all error-free code into a single runnable snippet.\n"
            "Provide a brief summary of exploration, preprocessing, and training steps.\n"
            "Conclude what model is best for the task.\n"
        ),
    )

    # Code execution backend (local or docker jupyter)
    executor_backend = str(config.get("execution", {}).get("code_executor_backend", "local-jupyter")).strip().lower()
    output_dir = artifacts_dir / "coding"
    output_dir.mkdir(exist_ok=True)

    if executor_backend == "docker-jupyter":
        docker_image = config.get("execution", {}).get("docker_jupyter_image") or None
        server = DockerJupyterServer(custom_image_name=docker_image)
    else:
        server = LocalJupyterServer()

    code_executor = autogen.UserProxyAgent(
        name="Code_Executor",
        system_message="Executor. Execute code and report result.",
        human_input_mode="NEVER",
        code_execution_config={"executor": JupyterCodeExecutor(server, output_dir=output_dir)},
    )

    client = OpenAIWrapper(config_list=config_list)

    def state_transition(last_speaker, groupchat):
        messages = groupchat.messages

        if last_speaker is initializer:
            return data_explorer

        elif last_speaker in [data_explorer, data_processer, model_trainer]:
            return code_executor

        elif last_speaker is code_executor:
            last_second_speaker_name = groupchat.messages[-2]["name"]

            if "exitcode: 1" in messages[-1]["content"]:
                return groupchat.agent_by_name(last_second_speaker_name)

            elif last_second_speaker_name == "Data_Explorer":
                return data_processer

            elif last_second_speaker_name == "Data_Processer":
                if is_ready_for_train(groupchat=groupchat, client=client):
                    return model_trainer
                return data_explorer

            elif last_second_speaker_name == "Model_Trainer":
                if count_train_trials(groupchat) < train_trials:
                    return model_trainer
                return summarizer

        elif last_speaker is summarizer:
            return None

    groupchat = autogen.GroupChat(
        agents=[initializer, data_explorer, data_processer, model_trainer, summarizer, code_executor],
        messages=[],
        max_round=max_rounds,
        speaker_selection_method=state_transition,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=None)

    task_prompt = f"""Please help me build a model to predict '{target}'.
- Task type: {task_type}
- The dataset is located at: `{data_path}`.
- All code will be executed in a Jupyter notebook, where previous states are saved.
- Save any plots to the current working directory (it will be persisted).
"""

    chat_result = None
    try:
        chat_result = initializer.initiate_chat(manager, message=task_prompt)
    finally:
        # Always stop server to avoid orphan kernel gateway processes/containers.
        try:
            server.stop()
        except Exception:
            jupyter_proc = getattr(server, "_subprocess", None)
            if jupyter_proc is not None and jupyter_proc.poll() is None:
                jupyter_proc.terminate()
                try:
                    jupyter_proc.wait(timeout=10)
                except Exception:
                    jupyter_proc.kill()

    # Save final integrated code if present
    saved_train_file = None
    if chat_result and "```python" in chat_result.chat_history[-1]["content"]:
        content = chat_result.chat_history[-1]["content"]
        content = content.split("```python")[1].split("```")[0].strip()
        saved_train_file = artifacts_dir / "train_file_by_agent.py"
        saved_train_file.write_text(content, encoding="utf-8")

    return {
        "saved_train_file": str(saved_train_file) if saved_train_file else None,
        "data_path": data_path,
        "target": target,
        "executor_backend": executor_backend,
    }