# Automated Machine Learning Workflow

An AG2-based multi-agent pipeline that automates tabular ML end-to-end: exploration, preprocessing, training, and summarization.

## What This Project Does

The workflow uses a state-machine style orchestration:

1. `Explore`: profile data and generate exploratory code.
2. `Preprocess`: clean and transform features.
3. `Train`: train and compare models in multiple trials.
4. `Summarize`: produce final summary and artifacts.

At each step, an LLM agent proposes code and a code executor runs it. Failed execution keeps the workflow in the same state for retry.

## Core Features

- Multi-agent orchestration with AG2 `GroupChat` custom speaker transitions.
- Stateful workflow transitions (`Explore -> Preprocess -> Train -> Summarize`).
- Two execution backends:
  - `local-jupyter`
  - `docker-jupyter`
- RAG support from local knowledge base (`kb/`) using Ollama embeddings.
- Job-style run management with `run_id`, status tracking, and structured artifacts.

## Prerequisites

- Python 3.12+
- DeepSeek API key (or OpenAI-compatible endpoint)
- Docker (optional)
- Ollama (optional, only needed when using RAG embedding/indexing)

## Installation

1. Clone and enter project:

```bash
git clone https://github.com/user-w-ui/Auto_ML.git
cd Auto_ML
```

2. Install dependencies:

```bash
pip install -e .
```

3. Configure environment variables:

```bash
cp .env.example .env
```

Minimal required fields in `.env`:

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

PowerShell alternative:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
```

## Configuration Files

- `configs/example.yaml`: local development defaults.
- `configs/container.yaml`: container-friendly paths and settings.

Important fields:

- `data.path`: input CSV path
- `data.target`: target column name
- `execution.code_executor_backend`: `local-jupyter` or `docker-jupyter`
- `workflow.train_trials`: number of training trials
- `workflow.max_rounds`: max group chat rounds

## Run Methods

### Recommended: CLI Job Mode

Create one run with generated `run_id`:

```bash
python -m src.cli run --config configs/example.yaml
```

Foreground mode (stream workflow logs to terminal):

```bash
python -m src.cli run --foreground --config configs/example.yaml
```

Custom run id:

```bash
python -m src.cli run --config configs/example.yaml --run-id my_run_001
```

List runs:

```bash
python -m src.cli list
```

Query a run status:

```bash
python -m src.cli status --run-id <run_id>
```

Status command auto-reconciles stale `running` runs by default (process dead or heartbeat timeout):

```bash
python -m src.cli status --run-id <run_id> --stale-after 900
```

Disable auto-reconcile if you only want raw file content:

```bash
python -m src.cli status --run-id <run_id> --no-refresh
```

### Legacy Direct Mode

You can still run directly via:

```bash
python -m src.main
```

This writes to `runs/manual_run/` and is useful for quick local debugging, but `src.cli` is preferred for traceable, multi-run job management.

## Status Monitoring and Observability

Each run persists status and logs under `runs/<run_id>/`.

Key files:

- `status.json`: current status (`running`, `succeeded`, `failed`), timestamps, pid, error.
- `run_config.json`: snapshot of the exact config used for reproducibility.
- `run_memory.jsonl`: short-term memory trail used by workflow steps.
- `logs/events.jsonl`: structured lifecycle events (`run_started`, `run_succeeded`, `run_failed`).

Useful monitoring commands:

```bash
python -m src.cli list
python -m src.cli status --run-id <run_id>
```

PowerShell quick event tail:

```powershell
Get-Content runs/<run_id>/logs/events.jsonl -Tail 20
```

## Artifact Layout

Standardized output structure per run:

- `coding/`: generated step scripts (`001_Explore.py`, `002_Preprocess.py`, ...), `chat_history.json`, and merged training script snapshots.
- `data/`: csv/json/joblib/parquet/xlsx artifacts.
- `plot/`: image/chart artifacts.
- `logs/`: structured run events.

This avoids scattering generated files in the project root.

## Docker

### Docker Compose (Recommended)

Run the full project in one container with mounted workspace:

```bash
docker compose build
docker compose run --rm automl
```

Current compose command executes:

```bash
python -m src.cli run --foreground --config configs/container.yaml
```

Why `configs/container.yaml` uses `local-jupyter`:

- The app itself is already running inside a container.
- `local-jupyter` keeps code execution in the same filesystem namespace (`/app`).
- This avoids host/container path mismatch for dataset and outputs.

### Docker Image Direct Run (No Compose)

Build image:

```bash
docker build -t automl-kaggle .
```

Run image directly:

```bash
docker run --rm --env-file .env automl-kaggle
```

Optional with host mount for persistent outputs:

```bash
docker run --rm --env-file .env -v ${PWD}:/app automl-kaggle
```

### Use Docker as Code Executor Backend (from Host Python)

If running `python -m src.cli` on host but want sandboxed execution:

```powershell
$env:CODE_EXECUTOR_BACKEND="docker-jupyter"
python -m src.cli run --config configs/example.yaml --foreground
```

Optional custom executor image:

```powershell
$env:DOCKER_JUPYTER_IMAGE="your-custom-image"
python -m src.cli run --config configs/example.yaml --foreground
```

## AG2 References

- [GroupChat with customized speaker transitions](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_groupchat_customized#group-chat-with-customized-speaker-selection-method)
- [StateFlow design pattern](https://docs.ag2.ai/docs/blog/2024-02-29-StateFlow/index#stateflow-build-state-driven-workflows-with-customized-speaker-selection-in-groupchat)
- [Code Execution](https://docs.ag2.ai/docs/user-guide/advanced-concepts/code-execution#code-execution)

## Tags

data analysis, groupchat, stateflow, code execution, kaggle, automated machine learning, workflow automation, model training, data preprocessing, state machine, hyperparameter tuning

## License

This project is licensed under Apache License 2.0. See [LICENSE](../LICENSE) for details.
