# src/mcp_cli/provider_config.py
"""
mcp_cli.provider_config
=======================

Hybrid configuration helper for MCP-CLI.

• **Defaults & schema** come straight from *chuk-llm*  
• Auto-sync with those defaults and persist **only the user’s overrides**
  to ``~/.mcp-cli/providers.json`` – so you can delete a broken file and the
  CLI heals itself on the next run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# ── canonical defaults from chuk-llm ────────────────────────────────────
from chuk_llm.llm.configuration.provider_config import DEFAULTS as CHUK_DEFAULTS

DEFAULTS: Dict[str, Dict[str, Any]] = CHUK_DEFAULTS           # public alias
CFG_PATH = Path(os.path.expanduser("~/.mcp-cli/providers.json"))


class ProviderConfig:
    """
    On-disk wrapper around *chuk-llm* provider defaults.

    • Deep-merges the stored JSON with fresh defaults on every load.  
    • Writes back the result when the structure changed (new provider/key).  
    • User overrides always win.
    """

    # ── construction ──────────────────────────────────────────────────
    def __init__(self, config_path: str | None = None) -> None:
        self._path = Path(os.path.expanduser(config_path)) if config_path else CFG_PATH
        self.providers: Dict[str, Dict[str, Any]] = self._load_and_sync()

    # ── private helpers ───────────────────────────────────────────────
    def _load_and_sync(self) -> Dict[str, Dict[str, Any]]:
        # read existing file (if any)
        on_disk: Dict[str, Dict[str, Any]] = {}
        if self._path.is_file():
            try:
                on_disk = json.loads(self._path.read_text())
            except json.JSONDecodeError:
                # corrupt file → ignore and rebuild
                pass

        # deep-copy defaults then overlay user config (user wins)
        merged: Dict[str, Dict[str, Any]] = json.loads(json.dumps(DEFAULTS))
        for prov, cfg in on_disk.items():
            merged.setdefault(prov, {}).update(cfg)

        # ── prune legacy MCP-CLI keys that confuse chuk-llm ───────────
        for prov_cfg in merged.values():
            prov_cfg.pop("client", None)          #  ← the single important line

        # write back if the structure changed (keeps file future-proof)
        if merged != on_disk:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(merged, indent=2))

        return merged

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self.providers, indent=2))

    def _ensure_section(self, name: str) -> None:
        if name not in self.providers:
            self.providers[name] = {}

    def _merge_env_key(self, cfg: Dict[str, Any]) -> None:
        """
        If an entry defines ``"api_key_env": "ENV_VAR"`` and ``api_key`` is
        still empty, pull the value from the environment each time we read.
        """
        if not cfg.get("api_key") and (env := cfg.get("api_key_env")):
            cfg["api_key"] = os.getenv(env)

    # ── public API ----------------------------------------------------
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        if provider not in DEFAULTS and provider not in self.providers:
            raise ValueError(f"Unknown provider: {provider}")
        self._ensure_section(provider)
        cfg = {**DEFAULTS.get(provider, {}), **self.providers[provider]}
        self._merge_env_key(cfg)
        return cfg

    def set_provider_config(self, provider: str, updates: Dict[str, Any]) -> None:
        self._ensure_section(provider)
        self.providers[provider].update(updates)
        self._save()

    # ── active provider / model ───────────────────────────────────────
    @property
    def _glob(self) -> Dict[str, Any]:
        self._ensure_section("__global__")
        return self.providers["__global__"]

    def get_active_provider(self) -> str:
        return self._glob.get("active_provider",
                              DEFAULTS["__global__"]["active_provider"])

    def set_active_provider(self, provider: str) -> None:
        self._glob["active_provider"] = provider
        self._save()

    def get_active_model(self) -> str:
        return self._glob.get("active_model",
                              DEFAULTS["__global__"]["active_model"])

    def set_active_model(self, model: str) -> None:
        self._glob["active_model"] = model
        self._save()

    # ── convenience getters ──────────────────────────────────────────
    def get_api_key(self, provider: str) -> Optional[str]:
        return self.get_provider_config(provider).get("api_key")

    def get_api_base(self, provider: str) -> Optional[str]:
        return self.get_provider_config(provider).get("api_base")

    def get_default_model(self, provider: str) -> str:
        return self.get_provider_config(provider).get("default_model", "")

    # explicit flush (rarely needed) -----------------------------------
    def save_config(self) -> None:
        self._save()
