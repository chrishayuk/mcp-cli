# src/mcp_cli/cli/commands/provider.py
"""
CLI binding for "provider" management commands.

All heavy-lifting is delegated to the shared helper:
    mcp_cli.commands.provider.provider_action_async
"""

from __future__ import annotations

import logging
from typing import Any

import typer
from chuk_term.ui import output

from mcp_cli.commands.provider import provider_action_async
from mcp_cli.model_manager import ModelManager
from mcp_cli.cli.commands.base import BaseCommand
from mcp_cli.utils.async_utils import run_blocking

log = logging.getLogger(__name__)

# ─── Typer sub-app ───────────────────────────────────────────────────────────
app = typer.Typer(help="Manage LLM provider configuration")


def _call_shared_helper(argv: list[str]) -> None:
    """Parse argv (after 'provider') and run shared async helper."""
    # Build a transient context dict (the shared helper expects it)
    ctx: dict[str, Any] = {
        "model_manager": ModelManager(),
        # The CLI path has no session client – we omit "client"
    }

    try:
        run_blocking(provider_action_async(argv, context=ctx))
    except Exception as e:
        output.print(f"[red]Provider command failed:[/red] {e}")
        log.exception("Provider command error")


@app.command("show", help="Show current provider & model")
def provider_show() -> None:
    """Show current provider and model status."""
    _call_shared_helper([])


@app.command("list", help="List configured providers")
def provider_list() -> None:
    """List all available providers with model counts and status."""
    _call_shared_helper(["list"])


@app.command("config", help="Display full provider config")
def provider_config() -> None:
    """Display detailed configuration for all providers."""
    _call_shared_helper(["config"])


@app.command("diagnostic", help="Run provider diagnostics")
def provider_diagnostic(
    provider_name: str = typer.Argument(None, help="Provider to diagnose (optional)"),
) -> None:
    """Run diagnostics on providers to check their configuration and connectivity."""
    args = ["diagnostic"]
    if provider_name:
        args.append(provider_name)
    _call_shared_helper(args)


@app.command("set", help="Set one configuration value")
def provider_set(
    provider_name: str = typer.Argument(..., help="Provider name"),
    key: str = typer.Argument(..., help="Configuration key"),
    value: str = typer.Argument(..., help="Configuration value"),
) -> None:
    """
    Set a configuration value for a provider.

    Examples:
        mcp-cli provider set openai api_key sk-your-key
        mcp-cli provider set anthropic api_base https://api.anthropic.com
        mcp-cli provider set groq default_model llama-3.3-70b-versatile
    """
    _call_shared_helper(["set", provider_name, key, value])


@app.command("switch", help="Switch to a provider")
def provider_switch(
    provider_name: str = typer.Argument(..., help="Provider name"),
    model: str = typer.Argument(None, help="Model name (optional)"),
) -> None:
    """
    Switch to a specific provider and optionally a model.

    Examples:
        mcp-cli provider switch anthropic
        mcp-cli provider switch openai gpt-4o
    """
    args = [provider_name]
    if model:
        args.append(model)
    _call_shared_helper(args)


# Add a convenience command that matches the pattern from main.py
@app.callback(invoke_without_command=True)
def provider_callback(
    ctx: typer.Context,
    provider_name: str = typer.Argument(None, help="Provider name to switch to"),
    model: str = typer.Option(None, "--model", help="Model name"),
) -> None:
    """
    Provider management. If no subcommand given, show status or switch provider.

    Usage:
        mcp-cli provider                    # Show current status
        mcp-cli provider list              # List providers
        mcp-cli provider anthropic         # Switch to Anthropic
        mcp-cli provider openai --model gpt-4o  # Switch to OpenAI with model
    """

    # If a subcommand was invoked, let it handle things
    if ctx.invoked_subcommand is not None:
        return

    # No subcommand provided
    if provider_name is None:
        # Show current status
        _call_shared_helper([])
    else:
        # Switch to provider
        args = [provider_name]
        if model:
            args.append(model)
        _call_shared_helper(args)


# ─── In-process command for CommandRegistry ──────────────────────────────────
class ProviderCommand(BaseCommand):
    """`provider` command usable from interactive or scripting contexts."""

    def __init__(self) -> None:
        super().__init__(
            name="provider",
            help_text="Manage LLM providers (show/list/config/set/switch).",
        )

    async def execute(self, tool_manager: Any, **params: Any) -> None:  # noqa: D401
        """
        Forward params to *provider_action_async*.

        Expected keys in **params:
          • subcommand (str)  – list | config | set | show | diagnostic
          • provider_name, key, value (for 'set')
          • model / provider (optional overrides)
        """
        argv: list[str] = []

        sub = params.get("subcommand", "show")

        if sub == "show":
            # Show current status
            argv = []
        elif sub in ["list", "config"]:
            # Simple commands
            argv = [sub]
        elif sub == "diagnostic":
            # Diagnostic command with optional provider
            argv = [sub]
            provider_name = params.get("provider_name")
            if provider_name:
                argv.append(provider_name)
        elif sub == "set":
            # Set command: set <provider> <key> <value>
            argv = [sub]
            for arg in ("provider_name", "key", "value"):
                val = params.get(arg)
                if val is None:
                    output.print(f"[red]Missing {arg} for 'set'[/red]")
                    return
                argv.append(str(val))
        else:
            # Treat it as "switch provider [model]"
            argv = [sub]  # sub is actually provider name
            maybe_model = params.get("model")
            if maybe_model:
                argv.append(maybe_model)

        context: dict[str, Any] = {
            "model_manager": ModelManager(),
        }

        try:
            await provider_action_async(argv, context=context)
        except Exception as e:
            output.print(f"[red]Provider command failed:[/red] {e}")
            log.exception("Provider command error in interactive mode")
