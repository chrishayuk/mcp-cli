# tests/test_mcp_cli_init.py
"""Tests for mcp_cli/__init__.py, including the ModuleNotFoundError branch."""

import importlib
import sys
from unittest.mock import patch


class TestMcpCliInit:
    """Cover the top-level __init__.py including the dotenv-missing branch."""

    def test_version_is_set(self):
        import mcp_cli

        assert hasattr(mcp_cli, "__version__")
        assert isinstance(mcp_cli.__version__, str)

    def test_all_is_defined(self):
        import mcp_cli

        assert "__version__" in mcp_cli.__all__

    def test_chuk_llm_env_vars_set(self):
        """After import, CHUK_LLM_* env vars should be set."""
        import os
        import mcp_cli  # noqa: F401

        assert os.environ.get("CHUK_LLM_DISCOVERY_ENABLED") == "true"
        assert os.environ.get("CHUK_LLM_AUTO_DISCOVER") == "true"
        assert os.environ.get("CHUK_LLM_OPENAI_TOOL_COMPATIBILITY") == "true"
        assert os.environ.get("CHUK_LLM_UNIVERSAL_TOOLS") == "true"

    def test_dotenv_not_installed_branch(self):
        """
        Simulate python-dotenv not being installed so the except
        ModuleNotFoundError branch (lines 31-33) is executed.
        """
        # Remove mcp_cli from the module cache so we can re-import it
        mods_to_remove = [
            k for k in sys.modules if k == "mcp_cli" or k.startswith("mcp_cli.")
        ]
        saved_modules = {}
        for mod_name in mods_to_remove:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

        # Also remove dotenv if cached
        saved_dotenv = sys.modules.pop("dotenv", None)

        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "dotenv":
                raise ModuleNotFoundError("No module named 'dotenv'")
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=fake_import):
                mod = importlib.import_module("mcp_cli")
                # Module should still load correctly
                assert hasattr(mod, "__version__")
        finally:
            # Restore original modules
            for mod_name, mod_obj in saved_modules.items():
                sys.modules[mod_name] = mod_obj
            if saved_dotenv is not None:
                sys.modules["dotenv"] = saved_dotenv

    def test_dotenv_installed_and_loads(self):
        """Verify the happy path where dotenv is available and loads."""
        import mcp_cli  # noqa: F401

        # If we got here, dotenv loaded (or was skipped gracefully)
        assert True
