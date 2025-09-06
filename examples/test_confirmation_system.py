#!/usr/bin/env python3
"""
Test script for the nuanced tool confirmation system.

This script demonstrates and tests:
1. Global confirmation modes (always, never, smart)
2. Per-tool confirmation settings
3. Risk-based confirmation in smart mode
4. Pattern-based rules
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.utils.preferences import PreferenceManager


def test_preference_manager():
    """Test the preference manager confirmation features."""

    print("=" * 60)
    print("Testing Tool Confirmation Preference System")
    print("=" * 60)

    # Create a test preference manager with a temp directory
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        prefs = PreferenceManager(config_dir=Path(tmpdir))

        # Test 1: Default mode should be smart
        print("\n1. Testing default mode:")
        mode = prefs.get_tool_confirmation_mode()
        print(f"   Default mode: {mode}")
        assert mode == "smart", f"Expected 'smart', got {mode}"
        print("   ✓ Default mode is 'smart'")

        # Test 2: Set and get different modes
        print("\n2. Testing mode changes:")
        for test_mode in ["always", "never", "smart"]:
            prefs.set_tool_confirmation_mode(test_mode)
            current = prefs.get_tool_confirmation_mode()
            assert current == test_mode, f"Expected {test_mode}, got {current}"
            print(f"   ✓ Set mode to '{test_mode}'")

        # Test 3: Test risk level detection
        print("\n3. Testing risk level detection:")
        test_tools = {
            "read_file": "safe",
            "list_tables": "safe",
            "get_resource": "safe",
            "write_file": "moderate",
            "create_database": "moderate",
            "update_config": "moderate",
            "delete_file": "high",
            "remove_database": "high",
            "execute_command": "high",
            "unknown_tool": "moderate",  # Default
        }

        for tool, expected_risk in test_tools.items():
            risk = prefs.get_tool_risk_level(tool)
            assert risk == expected_risk, (
                f"Tool {tool}: expected {expected_risk}, got {risk}"
            )
            print(f"   ✓ {tool}: {risk} risk")

        # Test 4: Test should_confirm_tool in smart mode
        print("\n4. Testing smart mode confirmation logic:")
        prefs.set_tool_confirmation_mode("smart")

        # Default thresholds: safe=False, moderate=True, high=True
        test_cases = {
            "read_file": False,  # Safe - no confirm
            "write_file": True,  # Moderate - confirm
            "delete_file": True,  # High - confirm
        }

        for tool, should_confirm in test_cases.items():
            result = prefs.should_confirm_tool(tool)
            assert result == should_confirm, (
                f"Tool {tool}: expected {should_confirm}, got {result}"
            )
            status = "requires confirmation" if should_confirm else "no confirmation"
            print(f"   ✓ {tool}: {status}")

        # Test 5: Test per-tool overrides
        print("\n5. Testing per-tool overrides:")

        # Override a safe tool to always confirm
        prefs.set_tool_confirmation("read_file", "always")
        assert prefs.should_confirm_tool("read_file") == True
        print("   ✓ read_file override to 'always' works")

        # Override a high-risk tool to never confirm
        prefs.set_tool_confirmation("delete_file", "never")
        assert prefs.should_confirm_tool("delete_file") == False
        print("   ✓ delete_file override to 'never' works")

        # Remove override
        prefs.set_tool_confirmation("read_file", None)
        assert prefs.should_confirm_tool("read_file") == False  # Back to default
        print("   ✓ Removing override works")

        # Test 6: Test risk threshold changes
        print("\n6. Testing risk threshold changes:")

        # Make safe tools require confirmation
        prefs.set_risk_threshold("safe", True)
        assert prefs.should_confirm_tool("list_tables") == True
        print("   ✓ Changed safe tools to require confirmation")

        # Make high-risk tools not require confirmation
        prefs.set_risk_threshold("high", False)
        prefs.set_tool_confirmation("delete_file", None)  # Remove previous override
        assert prefs.should_confirm_tool("delete_file") == False
        print("   ✓ Changed high-risk tools to not require confirmation")

        # Reset thresholds
        prefs.set_risk_threshold("safe", False)
        prefs.set_risk_threshold("moderate", True)
        prefs.set_risk_threshold("high", True)

        # Test 7: Test pattern rules
        print("\n7. Testing pattern-based rules:")

        # Add a pattern for SQL tools
        prefs.add_tool_pattern("sql_*", "always")
        patterns = prefs.preferences.ui.tool_confirmation.patterns
        assert len(patterns) > 0
        assert patterns[-1]["pattern"] == "sql_*"
        print("   ✓ Added pattern rule for 'sql_*'")

        # Remove pattern
        removed = prefs.remove_tool_pattern("sql_*")
        assert removed == True
        print("   ✓ Removed pattern rule")

        # Test 8: Test mode switching
        print("\n8. Testing global mode effects:")

        # Always mode - everything confirms
        prefs.set_tool_confirmation_mode("always")
        assert prefs.should_confirm_tool("read_file") == True
        assert prefs.should_confirm_tool("delete_file") == True
        print("   ✓ 'always' mode confirms all tools")

        # Never mode - nothing confirms (unless overridden)
        prefs.set_tool_confirmation_mode("never")
        prefs.set_tool_confirmation("delete_file", None)  # Clear override
        assert prefs.should_confirm_tool("read_file") == False
        assert prefs.should_confirm_tool("delete_file") == False
        print("   ✓ 'never' mode skips all confirmations")

        # Per-tool override still works in never mode
        prefs.set_tool_confirmation("delete_file", "always")
        assert prefs.should_confirm_tool("delete_file") == True
        print("   ✓ Per-tool override works even in 'never' mode")

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)


def test_ui_integration():
    """Test UI integration of confirmation system."""
    print("\n" + "=" * 60)
    print("Testing UI Integration")
    print("=" * 60)

    print("\nThe confirmation system integrates with the UI to:")
    print("1. Show risk levels (safe/moderate/high) in confirmation prompts")
    print("2. Allow users to set 'always allow' or 'never allow' for specific tools")
    print("3. Use the /confirm command to manage settings interactively")

    print("\n/confirm command usage:")
    print("  /confirm                     - Show current settings")
    print("  /confirm mode smart          - Set to smart (risk-based) mode")
    print("  /confirm mode always         - Always confirm all tools")
    print("  /confirm mode never          - Never confirm any tools")
    print("  /confirm tool read_file always - Always confirm read_file")
    print("  /confirm tool write_file never - Never confirm write_file")
    print("  /confirm tool delete_file remove - Remove override for delete_file")
    print("  /confirm risk safe on        - Enable confirmation for safe tools")
    print("  /confirm risk high off       - Disable confirmation for high-risk tools")
    print("  /confirm pattern 'sql_*' always - Always confirm SQL tools")
    print("  /confirm reset               - Reset to default settings")

    print("\nDuring tool execution:")
    print("  y = yes (execute tool)")
    print("  n = no (cancel execution)")
    print("  a = always allow this tool (no future confirmations)")
    print("  s = skip always (always require confirmation)")


if __name__ == "__main__":
    print("MCP-CLI Tool Confirmation System Test Suite\n")

    try:
        # Test preference manager
        test_preference_manager()

        # Show UI integration info
        test_ui_integration()

        print("\n✅ All tests completed successfully!")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
