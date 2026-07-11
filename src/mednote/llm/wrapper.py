"""Unified LLM wrapper — load OpenAI, Anthropic, or Google models via LangChain.

Defaults come from config.yml (Task 1B). ENV vars, if set, override the
provider/model at deploy time (e.g. CI) without editing the file.

Usage:
    from mednote.llm.wrapper import get_llm, get_fast_llm

    llm = get_llm()                                    # Main LLM (SOAP generation)
    fast_llm = get_fast_llm()                          # Fast/cheap tasks
    llm = get_llm(provider="openai", model="gpt-4o")   # Explicit override
"""
import os

from langchain_core.language_models import BaseChatModel

from mednote.config import get_config


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> BaseChatModel:
    """Load the main chat model.

    Resolution order for provider/model: explicit arg → env var
    (LLM_PROVIDER / LLM_MODEL) → config.yml. Temperature and max_tokens
    default to config.yml when not passed explicitly.
    """
    cfg = get_config().llm
    provider = provider or os.getenv("LLM_PROVIDER") or cfg.provider
    model = model or os.getenv("LLM_MODEL") or cfg.model
    temperature = cfg.temperature if temperature is None else temperature
    max_tokens = cfg.max_tokens if max_tokens is None else max_tokens

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            "Supported: 'anthropic', 'openai', 'google'"
        )


def get_fast_llm(**kwargs) -> BaseChatModel:
    """Get a fast/cheap LLM for lightweight tasks (entity extraction, classification)."""
    fast_cfg = get_config().llm.fast
    provider = kwargs.pop("provider", None) or fast_cfg.provider
    model = kwargs.pop("model", None) or fast_cfg.model
    max_tokens = kwargs.pop("max_tokens", None) or fast_cfg.max_tokens
    return get_llm(provider=provider, model=model, max_tokens=max_tokens, **kwargs)
