"""Unit tests for playground/client.py.

All HTTP calls are mocked — no live server required.
"""
from __future__ import annotations

import json
import sys
import textwrap
import unittest.mock as mock
from io import StringIO
from pathlib import Path

import pytest

# Make playground importable without installing it as a package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "playground"))

import client as pg  # noqa: E402  (playground/client.py)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_response(
    content: str = "Hello!",
    model: str = "test-model",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
    total_tokens: int = 8,
) -> dict:
    return {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def _mock_urlopen(response_body: bytes, status: int = 200):
    """Return a context manager mock that yields a fake HTTP response."""
    resp_mock = mock.MagicMock()
    resp_mock.read.return_value = response_body
    resp_mock.status = status
    resp_mock.__enter__ = mock.Mock(return_value=resp_mock)
    resp_mock.__exit__ = mock.Mock(return_value=False)
    return resp_mock


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_normal(self):
        assert pg.extract_text(_make_response("Hi")) == "Hi"

    def test_empty_choices(self):
        assert pg.extract_text({"choices": []}) == "(no content in response)"

    def test_missing_key(self):
        assert pg.extract_text({}) == "(no content in response)"


# ---------------------------------------------------------------------------
# format_usage
# ---------------------------------------------------------------------------

class TestFormatUsage:
    def test_normal(self):
        result = pg.format_usage(_make_response(prompt_tokens=10, completion_tokens=20, total_tokens=30))
        assert "10 prompt" in result
        assert "20 completion" in result
        assert "30 total" in result

    def test_missing_usage(self):
        assert pg.format_usage({}) == "tokens: unknown"


# ---------------------------------------------------------------------------
# wrap_column
# ---------------------------------------------------------------------------

class TestWrapColumn:
    def test_short_text_unchanged(self):
        lines = pg.wrap_column("hello", 80)
        assert lines == ["hello"]

    def test_long_text_wrapped(self):
        long_text = "word " * 20
        lines = pg.wrap_column(long_text.strip(), 20)
        assert all(len(line) <= 20 for line in lines)
        assert len(lines) > 1

    def test_empty_string_returns_single_line(self):
        assert pg.wrap_column("", 40) == [""]

    def test_newlines_preserved_as_separate_paragraphs(self):
        text = "line one\nline two"
        lines = pg.wrap_column(text, 80)
        assert "line one" in lines
        assert "line two" in lines


# ---------------------------------------------------------------------------
# format_single
# ---------------------------------------------------------------------------

class TestFormatSingle:
    def test_contains_model_name(self):
        out = pg.format_single("hi", _make_response("ok", model="my-model"), "my-model")
        assert "my-model" in out

    def test_contains_response_text(self):
        out = pg.format_single("hi", _make_response("Hello there"), "m")
        assert "Hello there" in out

    def test_contains_usage(self):
        out = pg.format_single("hi", _make_response(total_tokens=99), "m")
        assert "99 total" in out

    def test_contains_latency_when_present(self):
        resp = _make_response()
        resp["_latency_ms"] = 142.5
        out = pg.format_single("hi", resp, "m")
        assert "142ms" in out


# ---------------------------------------------------------------------------
# format_compare
# ---------------------------------------------------------------------------

class TestFormatCompare:
    def test_contains_both_model_names(self):
        out = pg.format_compare(
            "prompt", "model-a", _make_response("A response"), "model-b", _make_response("B response")
        )
        assert "model-a" in out
        assert "model-b" in out

    def test_contains_both_responses(self):
        out = pg.format_compare(
            "p", "a", _make_response("Alpha answer"), "b", _make_response("Beta answer")
        )
        assert "Alpha answer" in out
        assert "Beta answer" in out

    def test_separator_present(self):
        out = pg.format_compare("p", "a", _make_response(), "b", _make_response())
        assert pg.SEPARATOR in out

    def test_both_usages_present(self):
        ra = _make_response(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        rb = _make_response(prompt_tokens=8, completion_tokens=4, total_tokens=12)
        out = pg.format_compare("p", "a", ra, "b", rb)
        assert "15 total" in out
        assert "12 total" in out

    def test_latency_shown_when_set(self):
        ra = _make_response()
        ra["_latency_ms"] = 250.0
        rb = _make_response()
        rb["_latency_ms"] = 180.0
        out = pg.format_compare("p", "a", ra, "b", rb)
        assert "250ms" in out
        assert "180ms" in out


# ---------------------------------------------------------------------------
# load_prompts
# ---------------------------------------------------------------------------

class TestLoadPrompts:
    def test_object_list(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text(json.dumps([{"text": "hi", "system": "be terse", "label": "test"}]))
        prompts = pg.load_prompts(str(f))
        assert len(prompts) == 1
        assert prompts[0].text == "hi"
        assert prompts[0].system == "be terse"
        assert prompts[0].label == "test"

    def test_string_list(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text(json.dumps(["first prompt", "second prompt"]))
        prompts = pg.load_prompts(str(f))
        assert len(prompts) == 2
        assert prompts[0].text == "first prompt"
        assert prompts[0].system is None

    def test_mixed_list(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text(json.dumps(["bare string", {"text": "obj prompt"}]))
        prompts = pg.load_prompts(str(f))
        assert prompts[0].text == "bare string"
        assert prompts[1].text == "obj prompt"

    def test_sample_prompts_file_parses(self):
        """The actual sample_prompts.json must load without errors."""
        sample_path = Path(__file__).parent.parent.parent / "playground" / "prompts" / "sample_prompts.json"
        prompts = pg.load_prompts(str(sample_path))
        assert len(prompts) >= 5
        assert all(p.text for p in prompts)


# ---------------------------------------------------------------------------
# chat_completion (mocked HTTP)
# ---------------------------------------------------------------------------

class TestChatCompletion:
    def test_sends_correct_payload(self):
        resp_body = json.dumps(_make_response("ok")).encode()
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(resp_body)):
            result = pg.chat_completion("hello", model="qwen", base_url="http://x:8000")
        assert result["choices"][0]["message"]["content"] == "ok"

    def test_includes_system_message_when_provided(self):
        captured_payload: list[bytes] = []

        class CapturingMock:
            def __init__(self, req, timeout):
                captured_payload.append(req.data)

            def read(self):
                return json.dumps(_make_response()).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with mock.patch("urllib.request.urlopen", CapturingMock):
            pg.chat_completion("hi", model="m", system="be brief")

        payload = json.loads(captured_payload[0])
        roles = [m["role"] for m in payload["messages"]]
        assert "system" in roles

    def test_http_error_raises_runtime_error(self):
        import urllib.error

        err = urllib.error.HTTPError("url", 500, "Internal Server Error", {}, None)
        err.read = lambda: b'{"error": "oops"}'
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                pg.chat_completion("hi", model="m")

    def test_url_error_raises_runtime_error_with_hint(self):
        import urllib.error

        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with pytest.raises(RuntimeError, match="scripts/dev.sh serve"):
                pg.chat_completion("hi", model="m")


# ---------------------------------------------------------------------------
# CLI argument parsing via main()
# ---------------------------------------------------------------------------

class TestCLI:
    def _run(self, argv: list[str], mock_resp=None) -> tuple[int, str]:
        """Run main() with mocked urlopen, capture stdout/stderr, return (exit_code, stdout)."""
        resp_body = json.dumps(mock_resp or _make_response("answer")).encode()
        buf = StringIO()
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(resp_body)):
            with mock.patch("sys.stdout", buf):
                code = pg.main(argv)
        return code, buf.getvalue()

    def test_inline_prompt_returns_0(self):
        code, out = self._run(["--model", "m", "hello"])
        assert code == 0
        assert "answer" in out

    def test_no_args_returns_1(self):
        with mock.patch("sys.stdout", StringIO()):
            code = pg.main([])
        assert code == 1

    def test_health_ok(self):
        buf = StringIO()
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(b"")):
            with mock.patch("sys.stdout", buf):
                code = pg.main(["--health"])
        assert code == 0
        assert "healthy" in buf.getvalue()

    def test_list_models(self):
        body = json.dumps({"data": [{"id": "qwen2.5-0.5b"}, {"id": "tinyllama-chat"}]}).encode()
        buf = StringIO()
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(body)):
            with mock.patch("sys.stdout", buf):
                code = pg.main(["--list-models"])
        assert code == 0
        out = buf.getvalue()
        assert "qwen2.5-0.5b" in out
        assert "tinyllama-chat" in out

    def test_compare_flag(self):
        resp_body = json.dumps(_make_response("response text")).encode()
        buf = StringIO()
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(resp_body)):
            with mock.patch("sys.stdout", buf):
                code = pg.main(["--compare", "model-a", "model-b", "some prompt"])
        assert code == 0
        out = buf.getvalue()
        assert "model-a" in out
        assert "model-b" in out

    def test_prompts_file(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text(json.dumps([{"text": "from file"}]))
        resp_body = json.dumps(_make_response("file answer")).encode()
        buf = StringIO()
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(resp_body)):
            with mock.patch("sys.stdout", buf):
                code = pg.main(["--prompts-file", str(f)])
        assert code == 0
        assert "file answer" in buf.getvalue()

    def test_bad_prompts_file_returns_1(self):
        buf = StringIO()
        with mock.patch("sys.stderr", buf):
            code = pg.main(["--prompts-file", "/nonexistent/path.json"])
        assert code == 1
