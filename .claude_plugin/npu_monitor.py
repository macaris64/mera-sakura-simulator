"""Tool handlers for the sakura-npu-monitor Claude Code plugin."""

import json

from .telemetry import sample_npu_status, sample_telemetry


def get_npu_status(input_data: dict) -> str:
    return json.dumps(sample_npu_status().to_dict(), indent=2)


def get_telemetry(input_data: dict) -> str:
    window_ms = input_data.get("window_ms", 1000)
    return json.dumps(sample_telemetry(window_ms=window_ms).to_dict(), indent=2)


TOOL_HANDLERS = {
    "get_npu_status": get_npu_status,
    "get_telemetry": get_telemetry,
}


def dispatch(tool_name: str, input_data: dict) -> str:
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    return handler(input_data)
