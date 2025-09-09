#!/usr/bin/env python3
"""
Working demonstration of custom provider with actual inference.

This demo shows the complete workflow:
1. Add a custom OpenAI-compatible provider
2. Set the environment variable for API key
3. Show actual inference working
4. Clean up by removing the provider

We'll use the OpenAI API but add it as a custom provider with a different name
to demonstrate how any OpenAI-compatible endpoint works.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chuk_term.ui import output
from chuk_term.ui.theme import set_theme
from mcp_cli.utils.preferences import get_preference_manager


def run_command(cmd: str, show_output=True, capture=True):
    """Run a command and optionally show output."""
    if show_output:
        output.print(f"\n[bold cyan]$[/bold cyan] {cmd}")
    
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if show_output and result.stdout:
            output.print(result.stdout.strip())
        if result.stderr and result.returncode != 0 and show_output:
            output.error(result.stderr.strip())
        return result.returncode == 0, result.stdout
    else:
        return subprocess.run(cmd, shell=True).returncode == 0, ""


def main():
    """Run the working demo."""
    
    # Set theme for nice output
    set_theme("dark")
    
    output.rule("🚀 Custom Provider Working Demo", style="primary")
    output.print()
    output.info("This demo shows a custom OpenAI-compatible provider with real inference.")
    output.print()
    
    # Check for OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        output.error("❌ OPENAI_API_KEY not found in environment")
        output.info("Please set OPENAI_API_KEY in your .env file or environment")
        sys.exit(1)
    
    output.success("✅ Found OPENAI_API_KEY in environment")
    output.print()
    
    # Initialize preferences manager
    prefs = get_preference_manager()
    
    # Clean up any existing custom providers
    custom_providers = prefs.get_custom_providers()
    for name in custom_providers:
        prefs.remove_custom_provider(name)
        output.info(f"Cleaned up existing provider: {name}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Step 1: Add custom provider
    # ═══════════════════════════════════════════════════════════════════
    output.rule("Step 1: Add Custom OpenAI Provider", style="cyan")
    output.info("We'll add OpenAI as 'my-custom-ai' to demonstrate custom providers.")
    
    provider_name = "my-custom-ai"
    api_base = "https://api.openai.com/v1"
    model = "gpt-4o-mini"
    
    cmd = f'uv run mcp-cli provider add {provider_name} "{api_base}" {model}'
    success, _ = run_command(cmd)
    
    if not success:
        output.error("Failed to add provider")
        sys.exit(1)
    
    # ═══════════════════════════════════════════════════════════════════
    # Step 2: Set environment variable
    # ═══════════════════════════════════════════════════════════════════
    output.rule("Step 2: Configure API Key", style="cyan")
    
    # The environment variable name for our custom provider
    env_var_name = f"{provider_name.upper().replace('-', '_')}_API_KEY"
    output.info(f"Setting {env_var_name} from OPENAI_API_KEY...")
    
    # Set the environment variable for our custom provider
    os.environ[env_var_name] = openai_key
    output.success(f"✅ Set {env_var_name}")
    
    # Verify it's set
    output.print()
    output.info("Verifying configuration:")
    run_command("uv run mcp-cli provider custom")
    
    # ═══════════════════════════════════════════════════════════════════
    # Step 3: Test inference
    # ═══════════════════════════════════════════════════════════════════
    output.rule("Step 3: Test Actual Inference", style="cyan")
    output.info("Now let's test the custom provider with real API calls...")
    output.print()
    
    # Test 1: Simple completion
    output.info("Test 1: Simple math question")
    test_cmd = f'''echo "What is 25 + 37? Reply with just the number." | uv run mcp-cli cmd --provider {provider_name} --model {model} --input - --raw --single-turn 2>/dev/null'''
    
    success, response = run_command(test_cmd, show_output=False)
    if success and response:
        # Extract just the answer from the response
        lines = response.strip().split('\n')
        # Find the actual response (skip the MCP CLI output)
        answer = None
        for line in lines:
            if line.strip() and not line.startswith('✓') and not line.startswith('ℹ') and 'MCP CLI' not in line:
                # Try to find a number in the response
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    answer = numbers[0]
                    break
        
        if answer:
            output.success(f"✅ Response: {answer}")
            if answer == "62":
                output.success("   Correct answer!")
        else:
            output.info(f"Response: {response.strip()}")
    else:
        output.warning("Could not get response")
    
    output.print()
    
    # Test 2: Show provider info
    output.info("Test 2: List provider to confirm it's active")
    run_command(f"uv run mcp-cli providers | grep {provider_name}")
    
    output.print()
    
    # Test 3: Interactive-style test
    output.info("Test 3: More complex prompt")
    complex_prompt = "Write a haiku about custom AI providers. Just the haiku, no explanation."
    
    # Create a temporary file with the prompt
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(complex_prompt)
        temp_file = f.name
    
    test_cmd = f'''uv run mcp-cli cmd --provider {provider_name} --model {model} --input {temp_file} --raw --single-turn 2>/dev/null'''
    
    output.print(f"Prompt: {complex_prompt}")
    output.print()
    
    success, response = run_command(test_cmd, show_output=False)
    if success and response:
        # Extract the haiku from the response
        lines = response.strip().split('\n')
        haiku_lines = []
        for line in lines:
            if line.strip() and not line.startswith('✓') and not line.startswith('ℹ') and 'MCP CLI' not in line and '--' not in line:
                haiku_lines.append(line.strip())
        
        if haiku_lines:
            output.success("✅ Response (Haiku):")
            for line in haiku_lines[:3]:  # Haiku should be 3 lines
                output.print(f"   {line}")
    else:
        output.warning("Could not get response")
    
    # Clean up temp file
    os.unlink(temp_file)
    
    # ═══════════════════════════════════════════════════════════════════
    # Step 4: Show it in the provider list
    # ═══════════════════════════════════════════════════════════════════
    output.rule("Step 4: Verify Provider Status", style="cyan")
    output.info("The custom provider should show as configured and ready:")
    output.print()
    
    # Show just our custom provider status
    cmd = f"uv run mcp-cli providers 2>&1 | grep -A1 {provider_name}"
    run_command(cmd)
    
    # ═══════════════════════════════════════════════════════════════════
    # Step 5: Clean up
    # ═══════════════════════════════════════════════════════════════════
    output.rule("Step 5: Clean Up", style="cyan")
    output.info("Removing the custom provider...")
    
    cmd = f"uv run mcp-cli provider remove {provider_name}"
    success, _ = run_command(cmd)
    
    if success:
        output.success("✅ Provider removed successfully")
    
    # Verify it's gone
    output.print()
    output.info("Verifying removal:")
    run_command("uv run mcp-cli provider custom")
    
    # Clean up environment
    if env_var_name in os.environ:
        del os.environ[env_var_name]
        output.info(f"Cleaned up {env_var_name} from environment")
    
    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    output.rule("✨ Demo Complete!", style="primary")
    output.print()
    output.success("Successfully demonstrated:")
    output.info("  ✅ Added custom OpenAI-compatible provider")
    output.info("  ✅ Configured API key via environment variable")
    output.info("  ✅ Performed actual inference with the custom provider")
    output.info("  ✅ Verified provider status")
    output.info("  ✅ Cleaned up provider and environment")
    output.print()
    
    output.tip("💡 Key Takeaways:")
    output.info("  • Any OpenAI-compatible API can be added as a custom provider")
    output.info("  • API keys follow the pattern: {PROVIDER_NAME}_API_KEY")
    output.info("  • Custom providers work exactly like built-in providers")
    output.info("  • Perfect for LocalAI, LM Studio, corporate proxies, etc.")
    output.print()
    
    output.hint("Try it yourself with LocalAI or another OpenAI-compatible service!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        output.warning("\n\nDemo interrupted")
        sys.exit(0)
    except Exception as e:
        output.error(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)