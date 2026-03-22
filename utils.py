def is_ready_for_train(groupchat, client):
    messages = (
        [
            {
                "role": "system",
                "content": """Based on the dataset exploration, and the data processing, please determine whether the data is ready for model training.
Please give a short summary of what we know about the dataset and what we have done so far.

Please follow this format:
Summary: <Your summary>
Decision: <choose from "Ready for training" or "Need more processing">
""",
            }
        ]
        + groupchat.messages
    )

    response = client.create(messages=messages)
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
