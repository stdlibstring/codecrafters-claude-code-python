import argparse
import json
import os
import subprocess
from typing import Any

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
MODEL_NAME = "anthropic/claude-haiku-4.5"
BASH_TIMEOUT_SECONDS = 30

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
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    }
                },
            },
        },
    },
]


def parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        tool_args = json.loads(raw_arguments)
    except json.JSONDecodeError:
        raise ValueError(f"invalid tool arguments: {raw_arguments}")

    if not isinstance(tool_args, dict):
        raise ValueError(f"tool arguments must be an object: {raw_arguments}")

    return tool_args


def execute_read_tool(raw_arguments: str) -> str:
    try:
        tool_args = parse_tool_arguments(raw_arguments)
    except ValueError as e:
        return str(e)

    file_path = tool_args.get("file_path")
    if not file_path:
        return "Read requires file_path"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"failed to read {file_path}: {e}"


def execute_write_tool(raw_arguments: str) -> str:
    try:
        tool_args = parse_tool_arguments(raw_arguments)
    except ValueError as e:
        return str(e)

    file_path = tool_args.get("file_path")
    content = tool_args.get("content")

    if not file_path:
        return "Write requires file_path"
    if content is None:
        return "Write requires content"

    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"wrote to {file_path}"
    except Exception as e:
        return f"failed to write {file_path}: {e}"


def execute_bash_tool(raw_arguments: str) -> str:
    try:
        tool_args = parse_tool_arguments(raw_arguments)
    except ValueError as e:
        return str(e)

    command = tool_args.get("command")
    if not command:
        return "Bash requires command"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"command timed out: {command}"
    except Exception as e:
        return f"failed to run command: {e}"

    output = (result.stdout or "") + (result.stderr or "")
    if output:
        return output

    if result.returncode != 0:
        return f"command failed with exit code {result.returncode}"

    return ""


def execute_tool_call(tool_name: str, raw_arguments: str) -> str:
    if tool_name == "Read":
        return execute_read_tool(raw_arguments)
    if tool_name == "Write":
        return execute_write_tool(raw_arguments)
    if tool_name == "Bash":
        return execute_bash_tool(raw_arguments)
    return f"unsupported tool: {tool_name}"


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
            model=MODEL_NAME,
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
            tool_output = execute_tool_call(fn.name, fn.arguments)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                }
            )


if __name__ == "__main__":
    main()
