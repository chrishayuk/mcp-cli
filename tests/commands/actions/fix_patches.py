#!/usr/bin/env python3
"""Script to fix all the patching issues in test files."""

import os
import re
from pathlib import Path


def fix_patches_in_file(filepath):
    """Fix patch paths in a test file."""
    with open(filepath, "r") as f:
        content = f.read()

    original_content = content

    # Get the module name from the filename
    filename = os.path.basename(filepath)
    if "test_models_action" in filename:
        module = "models"
    elif "test_ping_action" in filename:
        module = "ping"
    elif "test_providers_action" in filename:
        module = "providers"
    elif "test_theme_action" in filename:
        module = "theme"
    elif "test_tools_action" in filename:
        module = "tools"
    elif "test_tools_call" in filename:
        module = "tools_call"
    else:
        return False

    # Fix output patches
    content = re.sub(
        r'patch\("chuk_term\.ui\.output"\)',
        f'patch("mcp_cli.commands.actions.{module}.output")',
        content,
    )
    content = re.sub(
        r'with patch\("chuk_term\.ui\.output"\) as',
        f'with patch("mcp_cli.commands.actions.{module}.output") as',
        content,
    )

    # Fix format_table patches
    content = re.sub(
        r'patch\("chuk_term\.ui\.format_table"\)',
        f'patch("mcp_cli.commands.actions.{module}.format_table")',
        content,
    )
    content = re.sub(
        r'with patch\("chuk_term\.ui\.format_table"\) as',
        f'with patch("mcp_cli.commands.actions.{module}.format_table") as',
        content,
    )

    # Special fixes for theme
    if module == "theme":
        content = re.sub(
            r'patch\("chuk_term\.ui\.theme\.set_theme"\)',
            'patch("mcp_cli.commands.actions.theme.set_theme")',
            content,
        )
        content = re.sub(
            r'with patch\("chuk_term\.ui\.theme\.set_theme"\)',
            'with patch("mcp_cli.commands.actions.theme.set_theme")',
            content,
        )
        content = re.sub(
            r'patch\("chuk_term\.ui\.prompts\.ask"',
            'patch("mcp_cli.commands.actions.theme.ask"',
            content,
        )

    if content != original_content:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Fixed: {filepath}")
        return True
    else:
        print(f"No changes needed: {filepath}")
    return False


def main():
    test_dir = Path("tests/commands/actions")
    test_files = [
        "test_ping_action.py",
        "test_providers_action.py",
        "test_theme_action.py",
        "test_tools_action_improved.py",
        "test_tools_call_action.py",
    ]

    for test_file in test_files:
        filepath = test_dir / test_file
        if filepath.exists():
            fix_patches_in_file(filepath)
        else:
            print(f"Not found: {filepath}")


if __name__ == "__main__":
    main()
