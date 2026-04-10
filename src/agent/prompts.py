from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PromptSpec:
    name: str
    prompt: str
    rag_query: Optional[str] = None
    tool_names: Optional[List[str]] = None


WORKFLOW_PROMPT_SPECS: List[PromptSpec] = [
    PromptSpec(
        name="Data_Explorer",
        rag_query="tabular dataset exploration checklist, missing values, target distribution, plots",
        prompt=(
            "You are the data explorer. Write code to explore dataset characteristics "
            "(shape, head, info/describe, missing values, target distribution, basic plots).\n"
            "Do NOT train models.\n"
            "At the end of exploration, you MUST write one JSON file to run data dir: "
            "explorer_dataset_profile.json.\n"
            "The JSON must be an object with keys: "
            "task_type, target_type, sample_size_bucket, feature_type_profile, signal_hypothesis, compute_budget.\n"
            "task_type must be exactly one of: supervised, unsupervised.\n"
            "If you think the data is ready and no more exploration is needed, "
            "reply exactly: Ready for training"
        ),
        tool_names=None,
    ),
    PromptSpec(
        name="Data_Processer",
        rag_query="tabular preprocessing checklist: missing values, categorical encoding, leakage prevention, pipeline",
        prompt=(
            "You are the data processer. Clean/prepare the dataset for model training.\n"
            "Handle missing values, encode categorical variables, scale when helpful.\n"
            "Avoid inplace=True."
        ),
        tool_names=None,
    ),
    PromptSpec(
        name="Model_Trainer",
        rag_query="tabular regression modeling, baselines, boosting models, metrics, residual plots",
        prompt=(
            "You are the model trainer. Train ONE model per iteration.\n"
            "Before proposing or training any model, follow this fixed order in TRAIN state:\n"
            "1) Read explorer_dataset_profile.json from run data dir.\n"
            "2) Build retrieval query text (recommended_labels) from profile fields.\n"
            "3) Call tool `read_model_cards_for_rag` with top_k=5.\n"
            "4) Parse 5 cards and build/update a model candidate queue file in run data dir.\n"
            "5) Consume only ONE candidate model + ONE hyperparameter setting for this iteration.\n"
            "6) Save training metrics/artifacts and queue progress for evaluator.\n"
            "Use 70/30 train/test split. Evaluate and save plots as images.\n"
            "Try a different model or different hyperparameters each iteration. No grid search."
        ),
        tool_names=["read_model_cards_for_rag"],
    ),
    PromptSpec(
        name="Code_Summarizer",
        rag_query="write a concise ML report with metrics table and plot references",
        prompt=(
            "You are the code summarizer. Integrate all error-free code into a single runnable snippet.\n"
            "Summarize exploration, preprocessing, training, and conclude best model."
        ),
        tool_names=None,
    ),
    PromptSpec(
        name="Evaluator",
        rag_query="ml quality gate, data readiness checks, overfitting signals, train/eval acceptance criteria",
        prompt=(
            "You are the evaluator. Assess whether the previous stage output passes quality gates.\n"
            "For TRAIN stage, decide one next action by improvement and stability:\n"
            "continue_next_candidate | retune_same_candidate | finish_training.\n"
            "Provide concise reasoning and concrete next-step guidance."
        ),
        tool_names=None,
    ),
]
