from typing import Callable, Dict, List, Any


def format_for_tokenization(tokenizer) -> Callable:
    """Default formatting function that just passes through the text"""

    def format_func(examples: Dict[str, Any]) -> Dict[str, List[str]]:
        return {"text": examples["text"] if "text" in examples else examples}

    return format_func


def default_format(tokenizer) -> Callable:
    def format_func(examples: Dict[str, Any]) -> Dict[str, List[str]]:
        texts = []
        assert "messages" in examples.keys()
        for i in range(len(examples["messages"])):
            messages = examples["messages"][i]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
        return {"text": texts}

    return format_func


def comparison_format(tokenizer) -> Callable:
    """Format function for comparison-style datasets"""

    def format_func(examples: Dict[str, Any]) -> Dict[str, List[str]]:
        texts = []
        for i in range(len(examples["prompt"])):
            texts.append(
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": """Let's think step by step to judge which response is better for the given prompt. Please keep your thoughts clear and concise and at max around 300 words. The output should be in the following format:\n```## Rationale: <Your reasoning>\n## Winner: <model_a or model_b>```\n\n""",
                        },
                        {
                            "role": "user",
                            "content": f"Prompt: {examples['prompt'][i]}\n\nResponse A: ```{examples['response_a'][i]}```\n\nResponse B: ```{examples['response_b'][i]}```\n\n",
                        },
                        {
                            "role": "assistant",
                            "content": f"## Rationale: {examples['rationale'][i]}\n## Winner: {examples['winner'][i]}",
                        },
                    ],
                    tokenize=False,
                )
            )
        return {"text": texts}

    return format_func


def sharegpt_format(tokenizer) -> Callable:
    """Format function for ShareGPT-style datasets"""

    def format_func(examples: Dict[str, Any]) -> Dict[str, List[str]]:
        texts = []
        for conv in examples["conversations"]:
            messages = []
            if isinstance(conv, list):
                for message in conv:
                    if message.get("from") == "human":
                        messages.append(
                            {"role": "user", "content": message.get("value", "")}
                        )
                    elif message.get("from") == "gpt":
                        messages.append(
                            {"role": "assistant", "content": message.get("value", "")}
                        )
                    elif message.get("from") == "system":
                        messages.insert(
                            0, {"role": "system", "content": message.get("value", "")}
                        )

            if not any(msg.get("role") == "system" for msg in messages):
                messages.insert(
                    0, {"role": "system", "content": "You are a helpful assistant."}
                )

            texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
        return {"text": texts}

    return format_func


def get_formatter(format_function: str, tokenizer) -> Callable:
    """
    Returns the appropriate formatter function based on the provided format_function name

    Args:
        format_function (str): Name of the formatter function to use
        tokenizer: The tokenizer to pass to the formatter function

    Returns:
        Callable: The selected formatter function
    """
    formatters = {
        "default_format": default_format,
        "comparison_format": comparison_format,
        "sharegpt_format": sharegpt_format,
        "format_for_tokenization": format_for_tokenization,
    }

    if format_function in formatters:
        return formatters[format_function](tokenizer)
    else:
        print(
            f"Warning: Unknown formatter '{format_function}', falling back to default_format"
        )
        return default_format(tokenizer)
