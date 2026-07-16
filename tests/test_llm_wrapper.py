"""Tests for the unified LLM wrapper.

These verify provider/model resolution and error handling without needing
real API keys — the provider classes are stubbed so no network/auth occurs.
"""
import sys
import types

import pytest

from mednote.llm import wrapper


class _StubModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_stub(monkeypatch, module_name: str, class_name: str) -> None:
    """Register a fake langchain provider module exposing a stub chat class."""
    mod = types.ModuleType(module_name)
    setattr(mod, class_name, _StubModel)
    monkeypatch.setitem(sys.modules, module_name, mod)


def _install_all_provider_stubs(monkeypatch) -> None:
    """Stub every supported provider so tests track config.yml's provider choice."""
    _install_stub(monkeypatch, "langchain_anthropic", "ChatAnthropic")
    _install_stub(monkeypatch, "langchain_openai", "ChatOpenAI")
    _install_stub(monkeypatch, "langchain_google_genai", "ChatGoogleGenerativeAI")


def _max_tokens_kwarg(provider: str) -> str:
    """The google branch passes max_output_tokens; the others pass max_tokens."""
    return "max_output_tokens" if provider == "google" else "max_tokens"


def test_unsupported_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        wrapper.get_llm(provider="llama")


def test_get_llm_uses_config_defaults(monkeypatch) -> None:
    _install_all_provider_stubs(monkeypatch)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    model = wrapper.get_llm()

    cfg = wrapper.get_config().llm
    assert model.kwargs["model"] == cfg.model
    assert model.kwargs["temperature"] == cfg.temperature
    assert model.kwargs[_max_tokens_kwarg(cfg.provider)] == cfg.max_tokens


def test_explicit_args_win_over_env(monkeypatch) -> None:
    _install_stub(monkeypatch, "langchain_openai", "ChatOpenAI")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-from-env")

    model = wrapper.get_llm(provider="openai", model="gpt-4o")

    assert model.kwargs["model"] == "gpt-4o"


def test_env_overrides_config(monkeypatch) -> None:
    _install_stub(monkeypatch, "langchain_openai", "ChatOpenAI")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-from-env")

    model = wrapper.get_llm()

    assert model.kwargs["model"] == "gpt-from-env"


def test_get_fast_llm_uses_fast_config(monkeypatch) -> None:
    _install_all_provider_stubs(monkeypatch)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    model = wrapper.get_fast_llm()

    fast_cfg = wrapper.get_config().llm.fast
    assert model.kwargs["model"] == fast_cfg.model
    assert model.kwargs[_max_tokens_kwarg(fast_cfg.provider)] == fast_cfg.max_tokens
