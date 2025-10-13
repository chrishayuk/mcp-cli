#!/usr/bin/env python3
"""
End-to-end demonstration of custom provider management in MCP CLI.

This example shows how to:
1. Add a custom OpenAI-compatible provider
2. Use the provider for chat/commands
3. Remove the provider when done

For this demo, we'll use OpenAI as an example of an OpenAI-compatible provider.
In real usage, this would be LocalAI, a custom proxy, or any OpenAI-compatible API.
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chuk_term.ui import output
from chuk_term.ui.theme import set_theme
from mcp_cli.utils.preferences import get_preference_manager
from mcp_cli.model_manager import ModelManager


def run_command(cmd: str, capture=True):
    """Run a CLI command and show output."""
    output.print(f"\n[dim]$ {cmd}[/dim]")

    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            output.print(result.stdout)
        if result.stderr and result.returncode != 0:
            output.error(result.stderr)
        return result.returncode == 0
    else:
        # Run interactively
        return subprocess.run(cmd, shell=True).returncode == 0


async def demo_custom_provider():
    """Demonstrate custom provider management."""

    # Set a nice theme for the demo
    set_theme("dark")

    output.rule("🎭 Custom Provider Management Demo", style="primary")
    output.print()
    output.info(
        "This demo shows how to add, use, and remove custom OpenAI-compatible providers."
    )
    output.info("We'll use a mock 'localai' provider as an example.")
    output.print()

    # Initialize components
    prefs = get_preference_manager()
    model_manager = ModelManager()

    # ═══════════════════════════════════════════════════════════════════
    # Step 1: Show current state
    # ═══════════════════════════════════════════════════════════════════════
    output.rule("Step 1: Check Current Providers", style="cyan")
    output.info("First, let's see what custom providers are currently configured...")

    custom_providers = prefs.get_custom_providers()
    if custom_providers:
        output.warning(f"Found {len(custom_providers)} existing custom provider(s)")
        for name in custom_providers:
            output.info(f"  • {name}")
    else:
        output.success("No custom providers configured (clean state)")

    # List current providers
    run_command("uv run mcp-cli provider custom")

    # ═══════════════════════════════════════════════════════════════════
    # Step 2: Add a custom provider
    # ═══════════════════════════════════════════════════════════════════
    output.rule("Step 2: Add Custom Provider", style="cyan")
    output.info("Adding a custom OpenAI-compatible provider called 'localai'...")
    output.info("In real usage, this could be:")
    output.info("  • LocalAI running on localhost:8080")
    output.info("  • A custom proxy server")
    output.info("  • Any OpenAI-compatible API endpoint")

    # Add the provider with one model (CLI limitation - only one model can be passed)
    success = run_command(
        'uv run mcp-cli provider add localai "http://localhost:8080/v1" gpt-4'
    )

    output.info(
        "\nNote: Due to CLI limitations, we can only pass one model via command."
    )
    output.info(
        "In a real application, you could add multiple models programmatically."
    )

    if success:
        output.success("✅ Provider added successfully!")

    # ═══════════════════════════════════════════════════════════════════
    # Step 3: Verify provider was added
    # ═══════════════════════════════════════════════════════════════════
    output.rule("Step 3: Verify Provider Configuration", style="cyan")
    output.info("Let's check that our provider was added correctly...")

    # List custom providers
    run_command("uv run mcp-cli provider custom")

    # Check if it appears in the main provider list
    output.info("\nChecking main provider list...")
    run_command("uv run mcp-cli providers | grep localai")

    # ═══════════════════════════════════════════════════════════════════
    # Step 4: Show API key configuration
    # ═══════════════════════════════════════════════════════════════════
    output.rule("Step 4: API Key Configuration", style="cyan")
    output.info("Custom providers use environment variables for API keys:")
    output.print()

    env_var = "LOCALAI_API_KEY"
    output.info(f"Expected environment variable: [bold]{env_var}[/bold]")

    if os.environ.get(env_var):
        output.success(f"✅ {env_var} is set")
    else:
        output.warning(f"⚠️  {env_var} is not set")
        output.hint(f"Set it with: export {env_var}=your-api-key")

    output.print()
    output.info("For this demo, we'll set a dummy API key...")
    os.environ[env_var] = "demo-api-key-12345"
    output.success(f"Set {env_var} for demo purposes")

    # ═══════════════════════════════════════════════════════════════════
    # Step 5: Demonstrate provider usage
    # ═══════════════════════════════════════════════════════════════════
    output.rule("Step 5: Using the Custom Provider", style="cyan")
    output.info("Now we can use the custom provider in various ways:")
    output.print()

    # Show how to switch to the provider
    output.info("1️⃣ Switch to provider in chat mode:")
    output.print("   $ mcp-cli --server sqlite")
    output.print("   > /provider localai")
    output.print()

    output.info("2️⃣ Use directly from command line:")
    output.print("   $ mcp-cli --provider localai --server sqlite")
    output.print()

    output.info("3️⃣ Use with specific model:")
    output.print("   $ mcp-cli --provider localai --model gpt-4 --server sqlite")
    output.print()

    # Programmatically test the provider
    output.info("Testing provider programmatically...")
    try:
        # Reload the model manager to pick up new providers
        model_manager = ModelManager()

        # Check if provider is available
        providers = model_manager.get_available_providers()
        if "localai" in providers:
            output.success("✅ Provider 'localai' is available in ModelManager")

            # Get available models
            models = model_manager.get_available_models("localai")
            output.info(f"   Available models: {', '.join(models)}")

            # Check provider info
            provider_info = model_manager.list_available_providers().get("localai", {})
            if provider_info.get("has_api_key"):
                output.success("   ✅ API key is configured")
            else:
                output.warning("   ⚠️  No API key found")
        else:
            output.warning("⚠️  Provider 'localai' not found in current session")
            output.info("   (This is expected - providers are loaded at startup)")
    except Exception as e:
        output.error(f"Error testing provider: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 6: Update provider configuration
    # ═══════════════════════════════════════════════════════════════════
    output.rule("Step 6: Update Provider Configuration", style="cyan")
    output.info("Providers can be updated after creation...")
    output.print()

    # Update the provider programmatically
    output.info("Updating API endpoint and models...")
    prefs.update_custom_provider(
        "localai",
        api_base="http://localhost:9090/v1",  # New endpoint
        models=["gpt-4", "gpt-3.5-turbo", "mixtral", "llama3"],  # Updated models
    )
    output.success("✅ Provider updated")

    # Show updated configuration
    run_command("uv run mcp-cli provider custom")

    # ═══════════════════════════════════════════════════════════════════
    # Step 7: Runtime provider (temporary)
    # ═══════════════════════════════════════════════════════════════════
    output.rule("Step 7: Runtime Providers (Session-only)", style="cyan")
    output.info("You can also add providers at runtime without persisting them:")
    output.print()

    output.info("Example command:")
    output.print('   $ mcp-cli --provider tempai --api-base "http://temp.ai/v1" \\')
    output.print('              --api-key "temp-key" --server sqlite')
    output.print()
    output.info("This provider exists only for that session and is not saved.")

    # ═══════════════════════════════════════════════════════════════════
    # Step 8: Remove the custom provider
    # ═══════════════════════════════════════════════════════════════════
    output.rule("Step 8: Remove Custom Provider", style="cyan")
    output.info("Finally, let's clean up by removing the custom provider...")

    success = run_command("uv run mcp-cli provider remove localai")

    if success:
        output.success("✅ Provider removed successfully!")

    # Verify it's gone
    output.info("\nVerifying provider was removed...")
    run_command("uv run mcp-cli provider custom")

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    output.rule("📚 Summary", style="primary")
    output.print()
    output.success("Demo completed! You've learned how to:")
    output.info("  ✅ Add custom OpenAI-compatible providers")
    output.info("  ✅ Configure API keys via environment variables")
    output.info("  ✅ Use custom providers in chat/command modes")
    output.info("  ✅ Update provider configuration")
    output.info("  ✅ Remove providers when no longer needed")
    output.print()

    output.tip("💡 Real-world examples of custom providers:")
    output.info("  • LocalAI - Local AI server (http://localhost:8080)")
    output.info("  • LM Studio - Local model server")
    output.info("  • Text Generation WebUI - Gradio interface")
    output.info("  • Custom proxy servers")
    output.info("  • Corporate AI gateways")
    output.print()

    output.hint("Remember: API keys are NEVER stored in config files!")
    output.hint("Always use environment variables for security.")


def main():
    """Run the demo."""
    try:
        asyncio.run(demo_custom_provider())
    except KeyboardInterrupt:
        output.warning("\n\nDemo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
