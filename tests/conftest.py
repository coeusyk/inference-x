"""Shared pytest fixtures.

Unit tests use stub engines via dependency overrides. Eager startup init would
load vLLM and break GPU-less test runs, so startup init is skipped here.
"""
import pytest


@pytest.fixture(autouse=True)
def _skip_eager_startup_init(monkeypatch):
    monkeypatch.setattr("inference_x.api.deps.initialize_app", lambda: None)
