# Automated Machine Learning Workflow

A personal project implementing an intelligent, automated machine learning workflow powered by AG2. This system orchestrates multiple AI agents to automatically analyze datasets, preprocess data, and train machine learning models end-to-end.

## Key Capabilities

The workflow automates the entire ML pipeline:

1. **Data Analysis & Exploration**: Automatically explore dataset characteristics, distributions, and patterns.
2. **Data Preprocessing**: Remove duplicates, handle missing values, encode categorical features, and scale numerical data.
3. **Model Training**: Train and compare multiple algorithms to find the best-performing model.

## Details

The workflow follows the steps of data analysis, preprocessing, and model training. Each step is executed by a specific agent, and the transition between steps is determined by the success or failure of the previous step.

We follow a state machine design to build the machine learning workflow:

- **`Init`** and **`End`**: Represent the start and end of the workflow.
- **`Explore`**: Analyze the dataset.
  - **Agents**: Data Explorer → Code Executor
  - **Transition**: If code execution is successful, move to `Preprocess`; otherwise, remain in `Explore`.
- **`Preprocess`**: Clean and prepare data.
  - **Agents**: Data Preprocessor → Code Executor
  - **Transition**: A language model determines whether all necessary preprocessing steps have been completed. If yes, move to `Train`; otherwise, return to `Explore` for further analysis.
- **`Train`**: Train a machine learning model.
  - **Agents**: Model Trainer → Code Executor
  - **Transition**: The model is trained in two iterations to compare performance. If the maximum trials are reached, move to `Summarize`. If code execution fails, remain in `Train` (failed trials do not count).
- **`Summarize`**: Generate a summary of the workflow.
  - **Agents**: Summarizer
  - **Transition**: Always moves to `End`.

At the `Explore`, `Preprocess`, and `Train` states:

- A **language model agent** is invoked first.
- A **code executor** then executes the generated code.
- If execution fails, the workflow remains in the same state.
- If execution succeeds, conditions are checked to determine whether to transition to the next state.

This structured workflow ensures an efficient and iterative approach to machine learning model building.

## AG2 Features

This project demonstrates the following AG2 features:

- [GroupChat with custom speaker transitions](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_groupchat_customized#group-chat-with-customized-speaker-selection-method)
- [Build with StateFlow design](https://docs.ag2.ai/docs/blog/2024-02-29-StateFlow/index#stateflow-build-state-driven-workflows-with-customized-speaker-selection-in-groupchat)
- [Code Execution](https://docs.ag2.ai/docs/user-guide/advanced-concepts/code-execution#code-execution)

## TAGS

TAGS: data analysis, groupchat, stateflow, code execution, kaggle, automated machine learning, workflow automation, model training, data preprocessing, state machine, hyperparameter tuning

## Prerequisites

- Python 3.12 or higher
- DeepSeek API key (or compatible OpenAI-format LLM API)
- Docker (optional, for isolated code execution)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/user-w-ui/Auto_ML.git
cd Auto_ML
```

2. Install dependencies:

```bash
pip install -e .
```

3. Create and configure `.env` file:

```bash
cp .env.example .env
# Edit .env with your DeepSeek API key
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

   Or set environment variables directly:
   ```powershell
   $env:DEEPSEEK_API_KEY="your_api_key"
   ```


## Usage

### Basic Example

Run the automated ML workflow:

```bash
python main.py
```

The workflow will automatically:
1. Analyze the dataset (`house_prices_train.csv`)
2. Preprocess and clean the data
3. Train multiple models and compare performance
4. Generate visualization charts
5. Output code and summary

### Docker Isolated Execution

For isolated code execution in a Docker container:

```powershell
$env:CODE_EXECUTOR_BACKEND="docker-jupyter"
python main.py
```

Or specify a custom image:

```powershell
$env:CODE_EXECUTOR_BACKEND="docker-jupyter"
$env:DOCKER_JUPYTER_IMAGE="your-custom-image"
python main.py
```

## References

- **AG2 Framework**: This project uses [AG2](https://github.com/ag2ai/ag2) for multi-agent orchestration - [AG2 Documentation](https://docs.ag2.ai/latest/)
- **Dataset**: House Prices dataset from Kaggle
- **Technologies**: Python, Jupyter, AG2, Scikit-learn, XGBoost, LightGBM, CatBoost

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](../LICENSE) for details.
