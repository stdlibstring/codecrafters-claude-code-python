import argparse
import json
import os

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    }
                },
                "required": ["file_path"],
            },
        },
    }
]


def execute_read_tool(raw_arguments: str) -> str:
    try:
        tool_args = json.loads(raw_arguments)
    except json.JSONDecodeError as e:
        return f"invalid tool arguments: {raw_arguments}"

    file_path = tool_args.get("file_path")
    if not file_path:
        return "Read requires file_path"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"failed to read {file_path}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    parsed_args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [{"role": "user", "content": parsed_args.p}]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=TOOLS,
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        message = chat.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls or len(message.tool_calls) == 0:
            print(message.content or "")
            return

        for tool_call in message.tool_calls:
            fn = tool_call.function
            if fn.name == "Read":
                tool_output = execute_read_tool(fn.arguments)
            else:
                tool_output = f"unsupported tool: {fn.name}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                }
            )


if __name__ == "__main__":
    main()
