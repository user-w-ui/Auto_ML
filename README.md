# Automated Machine Learning Workflow

An AG2-based multi-agent pipeline that automates tabular ML end-to-end: exploration, preprocessing, training, evaluation, and summarization.

## What This Project Does

The workflow uses a state-machine style orchestration with an explicit controller agent:

1. `Explore`: profile data and generate exploratory code.
2. `Preprocess`: clean and transform features.
3. `Train`: train and compare models in multiple trials.
4. `Evaluate`: check quality gates and decide next transition.
5. `Summarize`: produce final summary and artifacts.

Agent roles:

- `Data_Explorer`: data inspection and exploratory code.
- `Data_Processer`: preprocessing and feature preparation.
- `Model_Trainer`: iterative model training.
- `Evaluator`: state control and workflow analysis (quality gate, pass/fail, replan guidance).
- `Code_Summarizer`: final integration and summary output.

Current workflow states are not limited to four stages. The complete state set is:

- `EXPLORE`
- `PREPROCESS`
- `TRAIN`
- `EVALUATE`
- `REPLAN`
- `HUMAN_REVIEW`
- `DONE`
- `FAILED`

At each step, an LLM agent proposes code and a code executor runs it. Failed execution keeps the workflow in the same state for retry.

## Core Features

- Multi-agent orchestration with AG2 `GroupChat` custom speaker transitions.
- Stateful workflow transitions with quality-gated control (`Explore -> Preprocess -> Evaluate -> Train -> Evaluate -> Done/Retry/Replan`).
- Dedicated `Evaluator` agent for transition control and process diagnostics.
- Expanded state machine with retry/replan/human-review outcomes.
- Code execution backends: `local-jupyter` and `docker-jupyter`.
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

## Quick Start

Use this as the primary startup command:

```bash
python -m src.cli run --foreground --config configs/example.yaml
```

This runs the full state-machine workflow and writes outputs to `runs/<run_id>/`.

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

## Project Structure Visualization

Project layout (with brief file responsibilities):

```text
automate-ml-for-kaggle/
├── src/
│   ├── cli.py                         # CLI entry: run/list/status commands
│   ├── main.py                        # Legacy direct entry for quick local run
│   ├── utils.py                       # Shared utility functions (execution parsing, trial counting, quality checks)
│   ├── agent/
│   │   ├── definition.py              # Agent construction and system-message composition
│   │   ├── prompts.py                 # Role prompts and RAG queries (includes Evaluator)
│   │   └── tools.py                   # Optional tool registry and attachment
│   ├── config/
│   │   └── load.py                    # YAML config loader and validation
│   ├── jobs/
│   │   ├── artifacts.py               # Run directory layout and artifact paths
│   │   ├── events.py                  # Structured event logger (JSONL)
│   │   └── status.py                  # Run status model, heartbeat, persistence
│   ├── rag/
│   │   ├── embeddings_ollama.py       # Embedding client for Ollama
│   │   ├── injector.py                # RAG context builder for prompts
│   │   ├── kb_index.py                # Mini vector index build/search for kb/*.md
│   │   └── run_memory.py              # Short-term run memory and deduped context summarization
│   └── workflow/
│       ├── app.py                     # Core workflow state machine and speaker transitions
│       ├── runtime_helpers.py         # Runtime helper functions (env, logging, memory context, artifact organization)
│       ├── code_helpers.py            # Code block extraction/sanitization and state file naming
│       ├── prompts.py                 # Task prompt templates for workflow execution
│       ├── runner.py                  # Worker process entry: run lifecycle + heartbeat + status updates
│       └── subprocess.py              # Subprocess helpers for workflow orchestration
├── configs/
│   ├── example.yaml                   # Local default run configuration
│   └── container.yaml                 # Container-oriented run configuration
├── kb/
│   └── *.md                           # Knowledge base docs used by RAG retrieval
├── runs/
│   └── <run_id>/                      # Per-run artifacts (coding, data, plot, logs, memory, status)
├── Dockerfile                         # Container image definition
├── docker-compose.yml                 # Compose service for one-command container run
└── pyproject.toml                     # Python project metadata and dependencies
```

## AG2 References

- [GroupChat with customized speaker transitions](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_groupchat_customized#group-chat-with-customized-speaker-selection-method)
- [StateFlow design pattern](https://docs.ag2.ai/docs/blog/2024-02-29-StateFlow/index#stateflow-build-state-driven-workflows-with-customized-speaker-selection-in-groupchat)
- [Code Execution](https://docs.ag2.ai/docs/user-guide/advanced-concepts/code-execution#code-execution)

## Tags

data analysis, groupchat, stateflow, code execution, kaggle, automated machine learning, workflow automation, model training, data preprocessing, state machine, hyperparameter tuning
