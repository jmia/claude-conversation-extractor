"""Shared test helpers for Claude Conversation Extractor tests."""

import json
from pathlib import Path


def write_jsonl(path: Path, entries: list) -> Path:
    """Write a list of dicts to a JSONL file and return the path."""
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return path


def user_entry(text: str, timestamp: str = "2025-05-25T10:00:00Z", cwd: str = "/projects/myapp") -> dict:
    return {
        "type": "user",
        "message": {"role": "user", "content": text},
        "timestamp": timestamp,
        "cwd": cwd,
    }


def assistant_entry(text: str, timestamp: str = "2025-05-25T10:01:00Z") -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
        "timestamp": timestamp,
    }
