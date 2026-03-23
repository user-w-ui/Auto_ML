def _normalize_chat_messages(messages):
    """
    Normalize arbitrary chat messages into OpenAI-like format: {role, content}.

    Rules:
    - Keep role if provided and valid.
    - Infer role from name when role is missing.
    - Ensure content is always a string.
    """
    normalized = []
    valid_roles = {"system", "user", "assistant", "tool"}

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        if role not in valid_roles:
            name = str(msg.get("name", "")).lower()
            if name in {"init", "user", "userproxy", "user_proxy"}:
                role = "user"
            elif name in {"code_executor", "executor", "tool"}:
                role = "tool"
            else:
                role = "assistant"

        content = msg.get("content", "")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)

        normalized.append({"role": role, "content": content})

    return normalized


def is_ready_for_train(messages, client):
    normalized_messages = _normalize_chat_messages(messages)
    payload_messages = [
        {
            "role": "system",
            "content": """Based on the dataset exploration, and the data processing, please determine whether the data is ready for model training.
Please give a short summary of what we know about the dataset and what we have done so far.

Please follow this format:
Summary: <Your summary>
Decision: <choose from "Ready for training" or "Need more processing">
""",
        }
    ] + normalized_messages

    response = client.create(messages=payload_messages)
    response_str = client.extract_text_or_completion_object(response)[0]

    print("-" * 50)
    print(response_str)
    print("-" * 50)

    if "ready for training" in response_str.lower():
        return True
    return False


def count_train_trials(groupchat):
    messages = groupchat.messages

    tcount = 0
    for i, message in enumerate(messages):
        if message["name"] == "Model_Trainer":
            tcount += 1
        elif (
            message["name"] == "Code_Executor"
            and "exitcode: 1" in message["content"]
            and messages[i - 1]["name"] == "Model_Trainer"
        ):
            tcount -= 1

    return tcount
