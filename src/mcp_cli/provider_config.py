"""
Provider configuration management for MCP-CLI.

Enhancements
------------
* **Auto-sync with new DEFAULTS:** When MCP‑CLI ships new providers or updates
  default values, the user's on‑disk configuration is now *deep‑merged* with
  the baked‑in ``DEFAULTS`` on every load.
    • Missing providers are added.
    • Missing keys inside existing providers are filled in.
    • User‑overridden keys always take precedence.
* The merged structure is written back to disk if it differs, so subsequent
  runs start from an up‑to‑date baseline while still reflecting custom edits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# global defaults (unchanged except for illustrative whitespace tweak)
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "__global__": {
        "active_provider": "openai",
        "active_model": "gpt-4o-mini",
    },
    "openai": {
        "client": "mcp_cli.llm.providers.openai_client.OpenAILLMClient",
        "api_key_env": "OPENAI_API_KEY",
        "api_base": None,
        "api_key": None,
        "default_model": "gpt-4o-mini",
    },
    "groq": {
        "client": "mcp_cli.llm.providers.groq_client.GroqAILLMClient",
        "api_key_env": "GROQ_API_KEY",
        "api_key": None,
        "api_base": None,
        "default_model": "llama-3.3-70b-versatile",
    },
    "ollama": {
        "client": "mcp_cli.llm.providers.ollama_client.OllamaLLMClient",
        "api_key_env": None,
        "api_base": "http://localhost:11434",
        "api_key": None,
        "default_model": "qwen3",
    },
    "gemini": {
        "client": "mcp_cli.llm.providers.gemini_client.GeminiLLMClient",
        "api_key_env": None,
        "api_key": None,
        "default_model": "gemini-2.0-flash",
    },
    "anthropic": {
        "client": "mcp_cli.llm.providers.anthropic_client.AnthropicLLMClient",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_base": None,
        "api_key": None,
        "default_model": "claude-3-7-sonnet",
    },
}

CFG_PATH = Path(os.path.expanduser("~/.mcp-cli/providers.json"))


class ProviderConfig:
    """Load / mutate / persist provider configuration with default syncing."""

    # ------------------------------------------------------------------
    # construction & I/O
    # ------------------------------------------------------------------
    def __init__(self, config_path: Optional[str] = None) -> None:
        self._path = Path(os.path.expanduser(config_path)) if config_path else CFG_PATH
        self.providers: Dict[str, Dict[str, Any]] = self._load_and_sync()

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------
    def _load_and_sync(self) -> Dict[str, Dict[str, Any]]:
        """Return the merged provider map and flush to disk if it changed."""
        on_disk: Dict[str, Dict[str, Any]] = {}
        if self._path.is_file():
            try:
                on_disk = json.loads(self._path.read_text())
            except json.JSONDecodeError:
                pass  # bad JSON → ignore and rebuild

        # deep‑copy defaults then overlay anything from disk (user wins)
        merged: Dict[str, Dict[str, Any]] = json.loads(json.dumps(DEFAULTS))
        for prov, cfg in on_disk.items():
            if prov not in merged:
                merged[prov] = cfg  # custom provider entirely provided by user
                continue
            merged[prov].update(cfg)  # user overrides baked‑in defaults

        # write back if structure changed (keeps file current with new providers)
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
        if not cfg.get("api_key") and (env := cfg.get("api_key_env")):
            cfg["api_key"] = os.getenv(env)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        # Only allow known providers (in DEFAULTS or already present)
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

    # ── active provider / model ──────────────────────────────────────
    @property
    def _glob(self) -> Dict[str, Any]:
        self._ensure_section("__global__")
        return self.providers["__global__"]

    def get_active_provider(self) -> str:
        return self._glob.get("active_provider", DEFAULTS["__global__"]["active_provider"])

    def set_active_provider(self, provider: str) -> None:
        self._glob["active_provider"] = provider
        self._save()

    def get_active_model(self) -> str:
        return self._glob.get("active_model", DEFAULTS["__global__"]["active_model"])

    def set_active_model(self, model: str) -> None:
        self._glob["active_model"] = model
        self._save()

    # ── convenience getters ─────────────────────────────────────────
    def get_api_key(self, provider: str) -> Optional[str]:
        return self.get_provider_config(provider).get("api_key")

    def get_api_base(self, provider: str) -> Optional[str]:
        return self.get_provider_config(provider).get("api_base")

    def get_default_model(self, provider: str) -> str:
        return self.get_provider_config(provider).get("default_model", "")

    def save_config(self) -> None:
        """Public method to save the current provider config to disk."""
        self._save()
