"""Offline tests for the startup embedding-runtime guard."""

import logging

import pytest

from fastapi_rag_lab.api import app as app_module


def _check(match, **extra):
    return {
        "index": {"ollama_version": "0.20.7"},
        "query": {"ollama_version": "0.35.1"},
        "match": match,
        **extra,
    }


def test_refuses_on_mismatch(monkeypatch):
    monkeypatch.delenv("RAG_ALLOW_RUNTIME_MISMATCH", raising=False)
    monkeypatch.setattr(
        app_module, "check_runtime_matches_index", lambda *a: _check(False)
    )
    with pytest.raises(RuntimeError, match=r"0\.20\.7.*0\.35\.1"):
        app_module.verify_embedding_runtime()


def test_override_downgrades_to_warning(monkeypatch, caplog):
    monkeypatch.setenv("RAG_ALLOW_RUNTIME_MISMATCH", "1")
    monkeypatch.setattr(
        app_module, "check_runtime_matches_index", lambda *a: _check(False)
    )
    with caplog.at_level(logging.WARNING):
        app_module.verify_embedding_runtime()
    assert "mismatch" in caplog.text


def test_unknown_warns_and_match_is_silent(monkeypatch, caplog):
    monkeypatch.setattr(
        app_module,
        "check_runtime_matches_index",
        lambda *a: _check(None, reason="index has no runtime fields"),
    )
    with caplog.at_level(logging.WARNING):
        app_module.verify_embedding_runtime()
    assert "not verified" in caplog.text
    monkeypatch.setattr(
        app_module, "check_runtime_matches_index", lambda *a: _check(True)
    )
    app_module.verify_embedding_runtime()
