"""Unit tests for playground/chat.py helpers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "playground"))

from chat import (  # noqa: E402
    append_turn,
    build_chat_payload,
    should_submit_prompt,
)


def test_build_chat_payload_structure():
    payload = build_chat_payload(
        [{"role": "user", "content": "hi"}],
        "qwen2.5-0.5b",
    )
    assert payload["model"] == "qwen2.5-0.5b"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["stream"] is True
    assert "temperature" in payload
    assert "max_tokens" in payload


def test_messages_list_appends_user_then_assistant():
    messages: list[dict[str, str]] = []
    messages = append_turn(messages, "user", "hello")
    messages = append_turn(messages, "assistant", "hi there")
    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_messages_reset_on_new_conversation():
    messages = append_turn([], "user", "one")
    messages = []
    assert messages == []


def test_empty_prompt_not_submitted():
    assert should_submit_prompt("") is False
    assert should_submit_prompt("   ") is False
    assert should_submit_prompt("hello") is True
