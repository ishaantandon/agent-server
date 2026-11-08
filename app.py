"""Minimal tool-using agent server for a Spring Winter web server.

Bedrock credentials come from the ECS task role that Spring Winter attaches
once you select a model on the project's Bedrock page -- no API keys needed.
"""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import boto3
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_ID = os.environ["MODEL_ID"]  # copy from the project's Bedrock page
bedrock = boto3.client("bedrock-runtime")  # region comes from the task environment

app = FastAPI()

SYSTEM = [{"text": "You are a helpful assistant. Use tools when they help."}]

TOOLS = {
    "toolSpec": {
        "name": "get_current_time",
        "description": "Get the current time in an IANA timezone, e.g. Asia/Kolkata.",
        "inputSchema": {"json": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
        }},
    }
}


def run_tool(name: str, args: dict) -> dict:
    if name == "get_current_time":
        now = datetime.now(ZoneInfo(args["timezone"]))
        return {"time": now.strftime("%Y-%m-%d %H:%M:%S %Z")}
    return {"error": f"unknown tool {name}"}


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatRequest):
    messages = [{"role": "user", "content": [{"text": req.message}]}]
    tool_calls = []

    for _ in range(5):  # cap the agent loop
        resp = bedrock.converse(
            modelId=MODEL_ID,
            system=SYSTEM,
            messages=messages,
            toolConfig={"tools": [TOOLS]},
        )
        msg = resp["output"]["message"]
        messages.append(msg)

        if resp["stopReason"] != "tool_use":
            text = "".join(b.get("text", "") for b in msg["content"])
            return {"reply": text, "tool_calls": tool_calls}

        results = []
        for block in msg["content"]:
            if "toolUse" in block:
                use = block["toolUse"]
                out = run_tool(use["name"], use["input"])
                print(f"tool call: {use['name']}({use['input']}) -> {out}", flush=True)
                tool_calls.append({"tool": use["name"], "input": use["input"], "output": out})
                results.append({"toolResult": {
                    "toolUseId": use["toolUseId"],
                    "content": [{"json": out}],
                }})
        messages.append({"role": "user", "content": results})

    return {"reply": "Stopped after too many tool calls.", "tool_calls": tool_calls}
