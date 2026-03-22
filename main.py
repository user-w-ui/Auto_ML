import os
import sys
import autogen
from autogen import OpenAIWrapper
from utils import is_ready_for_train, count_train_trials
from pathlib import Path
from dotenv import load_dotenv

# Ensure Jupyter CLI is discoverable when running Python via absolute path
# (common on Windows without activating the conda env in shell).
if os.name == "nt":
    scripts_dir = Path(sys.executable).parent / "Scripts"
    if scripts_dir.exists():
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{scripts_dir}{os.pathsep}{current_path}"

# Prevent kernel gateway websocket ping timeout during long-running code blocks.
os.environ.setdefault("KG_WS_PING_INTERVAL_SECS", "0")

from autogen.coding.jupyter import DockerJupyterServer, JupyterCodeExecutor, LocalJupyterServer

# 关键：从环境变量读取 API KEY 和 BASE URL
# 如果你运行后发现 DEEPSEEK_* 读不到，我再教你加 2 行 python-dotenv。
load_dotenv(Path(__file__).with_name(".env"))  # 加载项目根目录 .env
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
deepseek_base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
deepseek_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

if not deepseek_api_key:
    raise RuntimeError("Missing DEEPSEEK_API_KEY. Please set it in .env or environment variables.")

llm_config = {
    "api_type": "openai",
    "model": deepseek_model,
    "api_key": deepseek_api_key,
    "base_url": deepseek_base_url,
}

config_list = [llm_config]

initializer = autogen.UserProxyAgent(
    name="Init",
    code_execution_config=False,
)

data_explorer = autogen.AssistantAgent(
    name="Data_Explorer",
    llm_config=llm_config,
    system_message="""You are the data explorer. Given a dataset and a task, please write code to explore and understand the properties of the dataset.
For example, you can:
- get the shape of the dataset
- get the first several rows of the dataset
- get the information of the dataset use `df.info()` or `df.describe()`
- plot the plots as needed (i.e. histogram, distribution)
- check the missing values
Only perform necessary data exploration steps.

If a data preprocessing step is performed, you only need to check whether the changes are good. Perform the exploration on the data as needed.
You should not train models in any time. If you think the data is ready and there are no more exploration or processing steps needed, please reply with "Ready for training".
""",
)

data_processer = autogen.AssistantAgent(
    name="Data_Processer",
    llm_config=llm_config,
    system_message="""You are the data processer. Given a dataset and a task, please write code to clean up the dataset. You goal is to prepare the dataset for model training.
This includes but not limited to:
1. handle missing values
2. Remove unnecessary columns for model training
3. Convert categorical variables to numerical variables
4. Scale numerical variables
5. other data preprocessing steps
Please decide what data preprocessing steps are needed based on the data exploration results.
When transforming data, try not to use `inplace=True`, but instead assign the transformed data to a new variable.
""",
)

model_trainer = autogen.AssistantAgent(
    name="Model_Trainer",
    llm_config=llm_config,
    system_message="""You are the model trainer. Given a dataset and a task, please write code to train one model for the task.
Please split the train data into 30% test and 70% train. If it is already done, please use the existing split.
You don't need to repeat previous code snippets.
Please reason about the choice of the model and select the one you think is the best for the task. For example, you can use models like but not limited to LinearRegression, RandomForestModel, GradientBoostingModel, CartModel, DistributedGradientBoostingModel, etc.
Each time, based on previous results, you should try a different model, or a different set of hyperparameters if you think this current model can be improved. And then evaluate the model on the test split.
Do not perform any hyperparameter tuning like grid search. Please try different models or different hyperparameters directly based on your intuition.

If you are asked to never use particular models, please do not use them even if they are better.

When you run  model training, I would like you to generate the performance metrics, you can use these visualisations but not limited to loss curves, confusion matrix, auc, classification report, etc. Save it as an image.
""",
)

summarizer = autogen.AssistantAgent(
    name="Code_Summarizer",
    llm_config=llm_config,
    system_message="""You are the code summarizer. Given a machine learning task and previous code snippets, please integrate all error-free code into a single code snippet.
Please also provide a brief summary of the data exploration, data processing, and model training steps, and conclude what model is the best for the task.
You should give the full code to reproduce the data exploration, data processing, and model training steps, and show the results with different metrics.
""",
)

output_dir = Path("coding")
output_dir.mkdir(exist_ok=True)
executor_backend = os.environ.get("CODE_EXECUTOR_BACKEND", "local-jupyter").strip().lower()

if executor_backend == "docker-jupyter":
    docker_image = os.environ.get("DOCKER_JUPYTER_IMAGE") or None
    server = DockerJupyterServer(custom_image_name=docker_image)
else:
    server = LocalJupyterServer()

code_executor = autogen.UserProxyAgent(
    name="Code_Executor",
    system_message="Executor. Execute the code written by the Coder and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "executor": JupyterCodeExecutor(server, output_dir=output_dir),
    },
)


client = OpenAIWrapper(config_list=config_list)


def state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    # init state
    if last_speaker is initializer:
        # init -> explore
        return data_explorer

    # these states contains two steps, we will always call code_executor after the first step
    elif last_speaker in [data_explorer, data_processer, model_trainer]:
        return code_executor

    elif last_speaker is code_executor:
        last_second_speaker_name = groupchat.messages[-2]["name"]

        # if we get an error, we repeat the current state
        if "exitcode: 1" in messages[-1]["content"]:
            return groupchat.agent_by_name(last_second_speaker_name)

        # explore state
        elif last_second_speaker_name == "Data_Explorer":
            return data_processer

        # process state
        elif last_second_speaker_name == "Data_Processer":
            if is_ready_for_train(groupchat=groupchat, client=client):
                return model_trainer
            return data_explorer

        elif last_second_speaker_name == "Model_Trainer":
            if count_train_trials(groupchat) < 2:
                return model_trainer
            return summarizer

    # summarize state
    elif last_speaker is summarizer:
        return None  # end the conversation


groupchat = autogen.GroupChat(
    agents=[
        initializer,
        data_explorer,
        data_processer,
        model_trainer,
        summarizer,
        code_executor,
    ],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=None)


task_prompt = """Please help me to build a model predict the sales price for each house.
- The dataset is downloaded to this location: `./house_prices_train.csv`.
- All code will be executed in a Jupyter notebook, where previous states are saved.
"""

# Never use ensemble models | Use a neural network this time using pytorch

# task_prompt = """Please help me to build a model to classify Iris setosa, Iris versicolor, and Iris virginica.
# - The dataset is downloaded to this location: `./iris.csv`.
# - All code will be executed in a Jupyter notebook, where previous states are saved.
# """


chat_result = None
try:
    chat_result = initializer.initiate_chat(manager, message=task_prompt)
finally:
    # Always stop server to avoid orphan kernel gateway processes/containers.
    # For LocalJupyterServer on Windows, force-close subprocess if graceful stop hangs.
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


if chat_result and "```python" in chat_result.chat_history[-1]["content"]:

    content = chat_result.chat_history[-1]["content"]
    content = content.split("```python")[1].split("```")[0].strip()
    with open("train_file_by_agent.py", "w") as f:
        f.write(content)
