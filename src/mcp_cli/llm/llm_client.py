# mcp_cli/llm/llm_client.py
"""
Central factory for obtaining a provider-specific LLM client.

* Reads client class path from ProviderConfig (“client” key).
* Filters kwargs to match each adapter's __init__.
* Late host override via set_host().
* Loads .env (dotenv) so API-key env-vars are available before we touch them.
"""
from __future__ import annotations

# ── NEW: load variables from .env early ────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()                    # silently ignores if .env absent
except ModuleNotFoundError:
    # python-dotenv not installed – no big deal; continue
    pass
# -------------------------------------------------------------------------

import importlib
import inspect
import logging
from types import ModuleType
from typing import Any, Dict, Optional, Type, TypeVar

from mcp_cli.llm.providers.base import BaseLLMClient
from mcp_cli.provider_config import ProviderConfig

log = logging.getLogger(__name__)
_T = TypeVar("_T", bound=BaseLLMClient)

# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────
def _import_string(path: str) -> Any:
    module_path, _, attr = path.replace(":", ".").rpartition(".")
    if not module_path or not attr:
        raise ImportError(f"Invalid import path: {path!r}")
    module: ModuleType = importlib.import_module(module_path)
    return getattr(module, attr)


def _supports_param(cls: Type, param: str) -> bool:
    return param in inspect.signature(cls.__init__).parameters


def _constructor_kwargs(cls: Type[_T], cfg: Dict[str, Any]) -> Dict[str, Any]:
    cand = {
        "model":    cfg.get("model") or cfg.get("default_model"),
        "api_key":  cfg.get("api_key"),
        "api_base": cfg.get("api_base"),
    }
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {k: v for k, v in cand.items() if v is not None}
    return {k: v for k, v in cand.items() if k in params and v is not None}

# ────────────────────────────────────────────────────────────────────────────
# public factory
# ────────────────────────────────────────────────────────────────────────────
def get_llm_client(
    provider: str = "openai",
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    config: Optional[ProviderConfig] = None,
) -> BaseLLMClient:
    cfg_mgr = config or ProviderConfig()
    cfg = cfg_mgr.get_provider_config(provider.lower())  # raises if unknown

    for k, v in (("model", model), ("api_key", api_key), ("api_base", api_base)):
        if v is not None:
            cfg[k] = v

    client_path = cfg.get("client")
    if not client_path:
        raise ValueError(f"No 'client' class configured for provider '{provider}'")

    ClientCls: Type[_T] = _import_string(client_path)
    kwargs = _constructor_kwargs(ClientCls, cfg)

    try:
        client: BaseLLMClient = ClientCls(**kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"Error initialising '{provider}' client: {exc}") from exc

    if cfg.get("api_base") and not _supports_param(ClientCls, "api_base") and hasattr(client, "set_host"):
        try:
            log.info("Setting host via set_host() on %s → %s", provider, cfg["api_base"])
            client.set_host(cfg["api_base"])  # type: ignore[attr-defined]
        except Exception as exc:
            log.warning("Unable to set host on %s: %s", provider, exc)

    return client
