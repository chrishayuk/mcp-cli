#!/usr/bin/env python3
# diagnostics/provider_set_validation.py
"""
Comprehensive diagnostic script to test provider set functionality and model validation.
Proves that the system won't switch to non-existent models and handles edge cases properly.
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_mcp_command(cmd: List[str], expect_success: bool = True) -> Tuple[bool, str, str]:
    """
    Run an MCP CLI command and return (success, stdout, stderr).
    """
    try:
        result = subprocess.run(
            ["uv", "run", "mcp-cli"] + cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        success = (result.returncode == 0) if expect_success else (result.returncode != 0)
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def get_available_models_for_provider(provider: str) -> List[str]:
    """Get list of available models for a provider using the ModelManager."""
    try:
        from mcp_cli.model_manager import ModelManager
        manager = ModelManager()
        models = manager.get_available_models(provider)
        return models
    except Exception as e:
        print(f"⚠️  Could not get models via ModelManager: {e}")
        return []