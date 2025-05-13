# mcp_cli/cli/cli_options.py
"""
Shared option-processing helpers for MCP-CLI commands.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mcp_cli.provider_config import ProviderConfig   # ← NEW

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# basic JSON / filesystem helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_config(config_file: str) -> Optional[dict]:
    """Read *config_file* (if it exists) and return a dict or None."""
    try:
        if Path(config_file).is_file():
            with open(config_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        logger.warning("Config file '%s' not found.", config_file)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in config file '%s'", config_file)
    except OSError as exc:
        logger.error("Error loading config file: %s", exc)
    return None


def extract_server_names(cfg: Optional[dict], specified: list[str] | None = None) -> Dict[int, str]:
    """
    Convert the ``mcpServers`` dict in *cfg* into {index: name}.
    If *specified* is provided, keep only those servers (in the same order).
    """
    names: Dict[int, str] = {}
    if not cfg or "mcpServers" not in cfg:
        return names

    servers = cfg["mcpServers"]

    if specified:
        for i, name in enumerate(specified):
            if name in servers:
                names[i] = name
    else:
        for i, name in enumerate(servers.keys()):
            names[i] = name
    return names


# ──────────────────────────────────────────────────────────────────────────────
# main helper used by /chat and /cmd
# ──────────────────────────────────────────────────────────────────────────────
def _provider_default_model(provider: str) -> str:
    """
    Return the configured default_model for *provider*.
    Falls back to 'gpt-4o-mini' if the provider is unknown / mis-configured.
    """
    try:
        cfg = ProviderConfig().get_provider_config(provider.lower())
        return cfg.get("default_model", "gpt-4o-mini")
    except Exception:
        # ProviderConfig might throw if provider is absent – keep CLI resilient.
        return "gpt-4o-mini"


def process_options(
    server: Optional[str],
    disable_filesystem: bool,
    provider: str,
    model: Optional[str],
    config_file: str = "server_config.json",
) -> Tuple[List[str], List[str], Dict[int, str]]:
    """
    Process CLI options → (servers_list, user_specified, server_names).

    * Sets env-vars LLM_PROVIDER / LLM_MODEL / SOURCE_FILESYSTEMS.
    * Expands comma-separated ``--server`` argument.
    """
    servers_list: List[str] = []
    user_specified: List[str] = []

    logger.debug("Processing options: server=%s disable_fs=%s", server, disable_filesystem)

    # ------------------------------------------------------------------ servers
    if server:
        user_specified = [s.strip() for s in server.split(",")]
        servers_list.extend(user_specified)

    # ------------------------------------------------------------------ model
    effective_model = (
        model
        or os.getenv("LLM_MODEL")
        or _provider_default_model(provider)
    )

    # ------------------------------------------------------------------ env vars
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = effective_model
    if not disable_filesystem:
        os.environ["SOURCE_FILESYSTEMS"] = json.dumps([os.getcwd()])

    # ------------------------------------------------------------------ read cfg
    cfg = load_config(config_file)

    if not servers_list and cfg and "mcpServers" in cfg:
        servers_list = list(cfg["mcpServers"].keys())  # default: all configured

    server_names = extract_server_names(cfg, user_specified)

    logger.debug("Resolved model=%s servers=%s", effective_model, servers_list)
    return servers_list, user_specified, server_names
