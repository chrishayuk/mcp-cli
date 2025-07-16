# mcp_cli/cli_options.py
"""
Clean MCP CLI integration with ChukLLM.
Sets environment variables, triggers discovery, and gets out of the way.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Global flag to ensure we only set up once
_ENV_SETUP_COMPLETE = False
_DISCOVERY_TRIGGERED = False


def setup_chuk_llm_environment():
    """
    Set up environment variables for ChukLLM discovery.
    MUST be called before any chuk_llm imports.
    """
    global _ENV_SETUP_COMPLETE
    
    if _ENV_SETUP_COMPLETE:
        return
    
    # Set environment variables (only if not already set by user)
    env_vars = {
        "CHUK_LLM_DISCOVERY_ENABLED": "true",
        "CHUK_LLM_AUTO_DISCOVER": "true",
        "CHUK_LLM_DISCOVERY_ON_STARTUP": "true",
        "CHUK_LLM_DISCOVERY_TIMEOUT": "10",
        "CHUK_LLM_OLLAMA_DISCOVERY": "true",
        "CHUK_LLM_OPENAI_DISCOVERY": "true",
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    _ENV_SETUP_COMPLETE = True
    logger.debug("ChukLLM environment variables set")


def trigger_discovery_after_setup():
    """
    Trigger discovery after environment setup.
    Call this after setup_chuk_llm_environment() and before using models.
    """
    global _DISCOVERY_TRIGGERED
    
    if _DISCOVERY_TRIGGERED:
        return 0
    
    try:
        # Import discovery functions
        from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh
        
        logger.debug("Triggering Ollama discovery from cli_options...")
        
        # Trigger Ollama discovery to get all available models
        new_functions = trigger_ollama_discovery_and_refresh()
        
        _DISCOVERY_TRIGGERED = True
        
        if new_functions:
            logger.debug(f"CLI discovery: {len(new_functions)} new Ollama functions")
        else:
            logger.debug("CLI discovery: no new functions (may already be cached)")
        
        return len(new_functions)
        
    except Exception as e:
        logger.debug(f"CLI discovery failed: {e}")
        return 0


def get_available_models_quick(provider: str = "ollama") -> List[str]:
    """
    Quick function to get available models after discovery.
    """
    try:
        from chuk_llm.llm.client import list_available_providers
        providers = list_available_providers()
        return providers.get(provider, {}).get("models", [])
    except Exception as e:
        logger.debug(f"Could not get models for {provider}: {e}")
        return []


def validate_provider_exists(provider: str) -> bool:
    """Validate provider exists, potentially after discovery"""
    try:
        from chuk_llm.configuration import get_config
        config = get_config()
        config.get_provider(provider)  # This will raise if not found
        return True
    except Exception:
        return False


def load_config(config_file: str) -> Optional[dict]:
    """Load MCP server config file."""
    try:
        if Path(config_file).is_file():
            with open(config_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Error loading config file '%s': %s", config_file, exc)
    return None


def extract_server_names(cfg: Optional[dict], specified: List[str] = None) -> Dict[int, str]:
    """Extract server names from config."""
    if not cfg or "mcpServers" not in cfg:
        return {}
    
    servers = cfg["mcpServers"]
    
    if specified:
        return {i: name for i, name in enumerate(specified) if name in servers}
    else:
        return {i: name for i, name in enumerate(servers.keys())}


def inject_logging_env_vars(cfg: dict, quiet: bool = False) -> dict:
    """Inject logging environment variables into MCP server configs."""
    if not cfg or "mcpServers" not in cfg:
        return cfg
    
    log_level = "ERROR" if quiet else "WARNING"
    logging_env_vars = {
        "PYTHONWARNINGS": "ignore",
        "LOG_LEVEL": log_level,
        "CHUK_LOG_LEVEL": log_level,
        "MCP_LOG_LEVEL": log_level,
    }
    
    modified_cfg = json.loads(json.dumps(cfg))  # Deep copy
    
    for server_name, server_config in modified_cfg["mcpServers"].items():
        if "env" not in server_config:
            server_config["env"] = {}
        
        for env_key, env_value in logging_env_vars.items():
            if env_key not in server_config["env"]:
                server_config["env"][env_key] = env_value
    
    return modified_cfg


def process_options(
    server: Optional[str],
    disable_filesystem: bool,
    provider: str,
    model: Optional[str],
    config_file: str = "server_config.json",
    quiet: bool = False,
) -> Tuple[List[str], List[str], Dict[int, str]]:
    """
    Process CLI options. Sets up environment, triggers discovery, and parses config.
    ChukLLM handles everything else.
    """
    
    # STEP 1: Set up ChukLLM environment first
    setup_chuk_llm_environment()
    
    # STEP 2: Trigger discovery immediately after setup
    discovery_count = trigger_discovery_after_setup()
    
    if discovery_count > 0:
        logger.debug(f"Discovery found {discovery_count} new functions")
    
    # STEP 3: Set model environment for downstream use
    os.environ["LLM_PROVIDER"] = provider
    if model:
        os.environ["LLM_MODEL"] = model
    
    # STEP 4: Set filesystem environment if needed
    if not disable_filesystem:
        os.environ["SOURCE_FILESYSTEMS"] = json.dumps([os.getcwd()])
    
    # STEP 5: Parse server configuration
    user_specified = []
    if server:
        user_specified = [s.strip() for s in server.split(",")]
    
    cfg = load_config(config_file)
    
    # STEP 6: Handle MCP server logging
    if cfg:
        cfg = inject_logging_env_vars(cfg, quiet=quiet)
        
        # Save modified config for MCP tool manager
        temp_config_path = Path(config_file).parent / f"_modified_{Path(config_file).name}"
        try:
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
            os.environ["MCP_CLI_MODIFIED_CONFIG"] = str(temp_config_path)
        except Exception as e:
            logger.warning(f"Failed to create modified config: {e}")
    
    # STEP 7: Build server list
    servers_list = user_specified or (list(cfg["mcpServers"].keys()) if cfg and "mcpServers" in cfg else [])
    server_names = extract_server_names(cfg, user_specified)
    
    logger.debug(f"Options processed: provider={provider}, model={model}, servers={len(servers_list)}")
    
    return servers_list, user_specified, server_names


def get_discovery_status() -> Dict[str, any]:
    """Get discovery status for debugging"""
    return {
        "env_setup_complete": _ENV_SETUP_COMPLETE,
        "discovery_triggered": _DISCOVERY_TRIGGERED,
        "discovery_enabled": os.getenv("CHUK_LLM_DISCOVERY_ENABLED", "false"),
        "ollama_discovery": os.getenv("CHUK_LLM_OLLAMA_DISCOVERY", "false"),
        "auto_discover": os.getenv("CHUK_LLM_AUTO_DISCOVER", "false"),
    }


def force_discovery_refresh():
    """Force a fresh discovery (useful for debugging)"""
    global _DISCOVERY_TRIGGERED
    _DISCOVERY_TRIGGERED = False
    
    # Set force refresh environment variable
    os.environ["CHUK_LLM_DISCOVERY_FORCE_REFRESH"] = "true"
    
    # Trigger discovery again
    return trigger_discovery_after_setup()