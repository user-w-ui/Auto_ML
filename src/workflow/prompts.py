from __future__ import annotations


CODE_EXECUTOR_SYSTEM_MESSAGE = "Executor. Execute the code written by the Coder and report the result."


def build_task_prompt(
    target: str,
    task_type: str,
    data_path: str,
    plot_dir: str,
    data_dir: str,
) -> str:
    return f"""Please help me to build a model to predict target `{target}`.
- Task type: {task_type}
- The dataset is available at: `{data_path}`.
- All code will be executed in a Jupyter notebook, where previous states are saved.
- Save ALL plot/image files under: `{plot_dir}`.
- Save ALL data/model files (.csv/.json/.pkl/.joblib/.xlsx/.parquet) under: `{data_dir}`.
"""
