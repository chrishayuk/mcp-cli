# mcp_cli/llm/llm_client.py
"""
mcp_cli.llm.llm_client  - delegate to *chuk-llm*.

We do zero extra work here: every provider, default-model rule and
feature-flag now lives in chuk-llm.

Any code that used to do

    from mcp_cli.llm.llm_client import get_llm_client

continues to work unchanged.
"""
from __future__ import annotations
from typing import Optional, Any

# chuk-llm public API
from chuk_llm.llm.llm_client import (
    get_llm_client as _chuk_get_llm_client,
)
from chuk_llm.llm.configuration.provider_config import ProviderConfig   # re-export


def get_llm_client(                     # noqa: D401 – simple façade
    provider: str = "openai",
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    config: Optional[ProviderConfig] = None,
    **extra: Any,
):
    """
    Return an LLM client from **chuk-llm**.

    All keyword arguments are passed straight through.
    """
    return _chuk_get_llm_client(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
        config=config,
        **extra,
    )

