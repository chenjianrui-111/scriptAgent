from script_agent.config.settings import (
    _default_ollama_model_map,
    _resolve_qwen_model_by_env,
)


def test_resolve_qwen_model_local_default(monkeypatch):
    monkeypatch.delenv("QWEN_MODEL_LOCAL", raising=False)
    assert _resolve_qwen_model_by_env("development") == "qwen2.5:0.5b"


def test_resolve_qwen_model_production_default(monkeypatch):
    monkeypatch.delenv("QWEN_MODEL_PRODUCTION", raising=False)
    assert _resolve_qwen_model_by_env("production") == "qwen2.5:7b"


def test_default_ollama_model_map_uses_local_override(monkeypatch):
    monkeypatch.setenv("QWEN_MODEL_LOCAL", "qwen2.5:1.5b")
    mapping = _default_ollama_model_map("development")
    assert mapping["美妆"] == "qwen2.5:1.5b"
    assert mapping["通用"] == "qwen2.5:1.5b"
