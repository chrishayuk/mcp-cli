# src/mcp_cli/main.py
"""Entry-point for the MCP CLI."""

from __future__ import annotations

import asyncio
import atexit
import gc
import os
import signal
import sys
from typing import Optional, List

import typer

# Module imports
from mcp_cli.logging_config import (
    setup_logging,
    get_logger,
    setup_silent_mcp_environment,
)

# Use unified command system
from mcp_cli.commands import register_all_commands as register_unified_commands
from mcp_cli.commands.registry import registry
from mcp_cli.commands.base import CommandMode
from mcp_cli.run_command import run_command_sync
from chuk_term.ui import (
    output,
    restore_terminal,
)
from chuk_term.ui.theme import set_theme
from mcp_cli.config import process_options
from mcp_cli.context import initialize_context

# ──────────────────────────────────────────────────────────────────────────────
# CRITICAL: Set up silent environment IMMEDIATELY after imports
# This prevents MCP server noise from appearing during module imports
# ──────────────────────────────────────────────────────────────────────────────
# FIRST: Set environment variables to silence MCP servers before they start
setup_silent_mcp_environment()

# THEN: Set up default clean logging immediately
# This will be overridden later if user specifies different options
setup_logging(level="ERROR", quiet=False, verbose=False)

# ──────────────────────────────────────────────────────────────────────────────
# Module logger
# ──────────────────────────────────────────────────────────────────────────────
logger = get_logger("main")

# ──────────────────────────────────────────────────────────────────────────────
# Typer root app
# ──────────────────────────────────────────────────────────────────────────────
app = typer.Typer(add_completion=False)


# ──────────────────────────────────────────────────────────────────────────────
# Default callback that handles no-subcommand case
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    config_file: str = typer.Option(
        "server_config.json", help="Configuration file path"
    ),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    provider: Optional[str] = typer.Option(None, help="LLM provider name"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging"
    ),
    log_level: str = typer.Option(
        "WARNING",
        "--log-level",
        help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
    theme: str = typer.Option(
        "default", "--theme", help="UI theme (default, dark, light, minimal, terminal)"
    ),
    confirm_mode: str = typer.Option(
        None,
        "--confirm-mode",
        help="Tool confirmation mode: always, never, or smart (risk-based)",
    ),
    init_timeout: float = typer.Option(
        120.0,
        "--init-timeout",
        help="Server initialization timeout in seconds",
    ),
    tool_timeout: Optional[float] = typer.Option(
        None,
        "--tool-timeout",
        help="Tool execution timeout in seconds (default: 120, can also set MCP_TOOL_TIMEOUT env var)",
    ),
    token_backend: Optional[str] = typer.Option(
        None,
        "--token-backend",
        help="Token storage backend: auto, keychain, windows, secretservice, encrypted, vault",
    ),
    max_turns: int = typer.Option(30, "--max-turns", help="Maximum conversation turns"),
) -> None:
    """MCP CLI - If no subcommand is given, start chat mode."""

    # Re-configure logging based on user options (this overrides the default ERROR level)
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)

    # Store tool timeout if specified
    if tool_timeout is not None:
        os.environ["MCP_TOOL_TIMEOUT"] = str(tool_timeout)

    # Store token backend preference if specified
    if token_backend:
        os.environ["MCP_CLI_TOKEN_BACKEND"] = token_backend

    # Set UI theme and confirmation mode - use preference if not specified
    from mcp_cli.utils.preferences import get_preference_manager

    pref_manager = get_preference_manager()

    # Set confirmation mode if specified
    if confirm_mode:
        if confirm_mode.lower() in ["always", "never", "smart"]:
            pref_manager.set_tool_confirmation_mode(confirm_mode.lower())
        else:
            output.print(f"[red]Invalid confirmation mode: {confirm_mode}[/red]")
            output.print("[dim]Valid modes: always, never, smart[/dim]")
            raise typer.Exit(1)

    if theme and theme != "default":
        # User specified theme via command line
        set_theme(theme)
        pref_manager.set_theme(theme)  # Save it as preference
    else:
        # Use saved preference
        saved_theme = pref_manager.get_theme()
        set_theme(saved_theme)

    # If a subcommand was invoked, let it handle things
    if ctx.invoked_subcommand is not None:
        return

    # IMPROVED: Better handling of --provider flag for common mistakes
    provider_commands = ["list", "config", "diagnostic", "set"]
    if provider and provider in provider_commands:
        logger.debug(f"Detected provider command in --provider flag: {provider}")
        output.tip(
            f"Use 'mcp-cli provider {provider}' or 'mcp-cli providers {provider}' instead"
        )
        output.info(f"Running: provider {provider}")

        # Execute the provider command
        from mcp_cli.commands.actions.providers import provider_action_async
        from mcp_cli.context import initialize_context

        # Initialize context for the provider command
        initialize_context(token_backend=token_backend)

        try:
            from mcp_cli.commands.models import ProviderActionParams

            params = ProviderActionParams(args=[provider])
            asyncio.run(provider_action_async(params))
        except Exception as e:
            output.error(f"Error: {e}")
        finally:
            restore_terminal()
        raise typer.Exit()

    # No subcommand - start chat mode (default behavior)
    logger.debug("Starting default chat mode")

    # Use ModelManager to get active provider/model if not specified
    from mcp_cli.model_manager import ModelManager

    model_manager = ModelManager()

    # Handle runtime custom provider if api_base is specified
    if api_base and provider:
        # Add as runtime provider (not persisted)
        logger.debug(f"Adding runtime provider: {provider} with base {api_base}")

        # Parse models if provided in model string (comma-separated)
        models = None
        if model and "," in model:
            models = [m.strip() for m in model.split(",")]
            model = models[0]  # Use first as default

        model_manager.add_runtime_provider(
            name=provider,
            api_base=api_base,
            api_key=api_key,  # Will be kept in memory only
            models=models,
        )

        output.info(f"Using runtime provider: {provider}")
        if api_key:
            output.success("API key provided (kept in memory only)")
        else:
            env_var = f"{provider.upper().replace('-', '_')}_API_KEY"
            if os.environ.get(env_var):
                output.success(f"Using API key from {env_var}")
            else:
                output.warning(f"No API key found. Set {env_var} environment variable.")

    # Validate provider if specified
    elif provider:
        if not model_manager.validate_provider(provider):
            available = ", ".join(model_manager.list_providers())
            output.error(f"Unknown provider: {provider}")
            output.info(f"Available providers: {available}")
            output.tip(f"Did you mean to run: mcp-cli provider {provider}")
            output.tip(
                "Or add a custom provider with: --provider <name> --api-base <url>"
            )
            raise typer.Exit(1)

    # Smart provider/model resolution:
    # 1. If both specified: use both
    # 2. If only provider specified: use provider + its default model
    # 3. If neither specified: use active provider + active model
    if provider and model:
        # Both specified explicitly
        effective_provider = provider
        effective_model = model
        logger.debug(f"Using explicit provider/model: {provider}/{model}")
    elif provider and not model:
        # Provider specified, get its default model
        effective_provider = provider
        effective_model = model_manager.get_default_model(provider)
        logger.debug(f"Using provider with default model: {provider}/{effective_model}")
    elif not provider and model:
        # Model specified, use current provider
        effective_provider = model_manager.get_active_provider()
        effective_model = model
        logger.debug(
            f"Using current provider with specified model: {effective_provider}/{model}"
        )
    else:
        # Neither specified, use active configuration
        effective_provider = model_manager.get_active_provider()
        effective_model = model_manager.get_active_model()
        logger.debug(
            f"Using active configuration: {effective_provider}/{effective_model}"
        )

    servers, _, server_names = process_options(
        server,
        disable_filesystem,
        effective_provider,
        effective_model,
        config_file,
        quiet=quiet,
    )

    from mcp_cli.chat.chat_handler import handle_chat_mode

    # Start chat mode directly with proper cleanup
    async def _start_chat():
        tm = None
        try:
            logger.debug("Initializing tool manager")
            from mcp_cli.run_command import _init_tool_manager

            tm = await _init_tool_manager(
                config_file, servers, server_names, init_timeout
            )

            logger.debug("Starting chat mode handler")
            success = await handle_chat_mode(
                tool_manager=tm,
                provider=effective_provider,  # Use effective values
                model=effective_model,  # Use effective values
                api_base=api_base,
                api_key=api_key,
                max_turns=max_turns,
            )
            logger.debug(f"Chat mode completed with success: {success}")
        except asyncio.TimeoutError:
            logger.error("Initialization or operation timed out")
            output.error("Operation timed out. Please check server configuration.")
            # Ensure cleanup happens before event loop closes
            if tm:
                logger.debug("Cleaning up after timeout")
                from mcp_cli.run_command import _safe_close

                await _safe_close(tm)
            raise
        except Exception as e:
            logger.error(f"Error in chat mode: {e}")
            # Ensure cleanup happens before event loop closes
            if tm:
                logger.debug("Cleaning up after error")
                from mcp_cli.run_command import _safe_close

                await _safe_close(tm)
            raise
        finally:
            if tm:
                logger.debug("Final cleanup of tool manager")
                from mcp_cli.run_command import _safe_close

                await _safe_close(tm)

    try:
        asyncio.run(_start_chat())
    except KeyboardInterrupt:
        output.warning("\nInterrupted")
        logger.debug("Chat mode interrupted by user")
    except asyncio.TimeoutError:
        # Already handled in _start_chat, just exit gracefully
        logger.debug("Exiting due to timeout")
    finally:
        restore_terminal()
        raise typer.Exit()


# ──────────────────────────────────────────────────────────────────────────────
# Built-in commands
# ──────────────────────────────────────────────────────────────────────────────


# Chat command - for backward compatibility (same as default behavior)
@app.command(
    "chat",
    help="Start chat mode (default behavior, provided for backward compatibility)",
)
def _chat_command(
    config_file: str = typer.Option(
        "server_config.json", help="Configuration file path"
    ),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    provider: Optional[str] = typer.Option(None, help="LLM provider name"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging"
    ),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
    theme: str = typer.Option(
        "default", "--theme", help="UI theme (default, dark, light, minimal, terminal)"
    ),
    confirm_mode: str = typer.Option(
        None,
        "--confirm-mode",
        help="Tool confirmation mode: always, never, or smart (risk-based)",
    ),
    init_timeout: float = typer.Option(
        120.0,
        "--init-timeout",
        help="Server initialization timeout in seconds",
    ),
) -> None:
    """Start chat mode (same as default behavior without subcommand)."""
    # Re-configure logging based on user options
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)

    # Set confirmation mode if specified
    if confirm_mode:
        from mcp_cli.utils.preferences import get_preference_manager

        pref_manager = get_preference_manager()
        if confirm_mode.lower() in ["always", "never", "smart"]:
            pref_manager.set_tool_confirmation_mode(confirm_mode.lower())
        else:
            output.print(f"[red]Invalid confirmation mode: {confirm_mode}[/red]")
            output.print("[dim]Valid modes: always, never, smart[/dim]")
            raise typer.Exit(1)

    # Set UI theme - use preference if not specified
    from mcp_cli.utils.preferences import get_preference_manager

    pref_manager = get_preference_manager()

    if theme and theme != "default":
        # User specified theme via command line
        set_theme(theme)
        pref_manager.set_theme(theme)  # Save it as preference
    else:
        # Use saved preference
        saved_theme = pref_manager.get_theme()
        set_theme(saved_theme)

    logger.debug("Starting chat mode (via explicit chat command)")

    # Use ModelManager to get active provider/model if not specified
    from mcp_cli.model_manager import ModelManager

    model_manager = ModelManager()

    # Handle runtime custom provider if api_base is specified
    if api_base and provider:
        # Add as runtime provider (not persisted)
        logger.debug(f"Adding runtime provider: {provider} with base {api_base}")

        # Parse models if provided in model string (comma-separated)
        models = None
        if model and "," in model:
            models = [m.strip() for m in model.split(",")]
            model = models[0]  # Use first as default

        model_manager.add_runtime_provider(
            name=provider,
            api_base=api_base,
            api_key=api_key,  # Will be kept in memory only
            models=models,
        )

        output.info(f"Using runtime provider: {provider}")
        if api_key:
            output.success("API key provided (kept in memory only)")
        else:
            import os

            env_var = f"{provider.upper().replace('-', '_')}_API_KEY"
            if os.environ.get(env_var):
                output.success(f"Using API key from {env_var}")
            else:
                output.warning(f"No API key found. Set {env_var} environment variable.")

    # Validate provider if specified
    elif provider:
        if not model_manager.validate_provider(provider):
            available = ", ".join(model_manager.list_providers())
            output.error(f"Unknown provider: {provider}")
            output.info(f"Available providers: {available}")
            raise typer.Exit(1)

    # Smart provider/model resolution
    if provider and model:
        effective_provider = provider
        effective_model = model
    elif provider and not model:
        effective_provider = provider
        effective_model = model_manager.get_default_model(provider)
    elif not provider and model:
        effective_provider = model_manager.get_active_provider()
        effective_model = model
    else:
        effective_provider = model_manager.get_active_provider()
        effective_model = model_manager.get_active_model()

    servers, _, server_names = process_options(
        server,
        disable_filesystem,
        effective_provider,
        effective_model,
        config_file,
        quiet=quiet,
    )

    from mcp_cli.chat.chat_handler import handle_chat_mode

    # Start chat mode directly with proper cleanup
    async def _start_chat():
        tm = None
        try:
            logger.debug("Initializing tool manager")
            from mcp_cli.run_command import _init_tool_manager

            tm = await _init_tool_manager(
                config_file, servers, server_names, init_timeout
            )

            logger.debug("Starting chat mode handler")
            success = await handle_chat_mode(
                tool_manager=tm,
                provider=effective_provider,
                model=effective_model,
                api_base=api_base,
                api_key=api_key,
            )
            logger.debug(f"Chat mode completed with success: {success}")
        except asyncio.TimeoutError:
            logger.error("Initialization or operation timed out")
            output.error("Operation timed out. Please check server configuration.")
            if tm:
                logger.debug("Cleaning up after timeout")
                from mcp_cli.run_command import _safe_close

                await _safe_close(tm)
            raise
        except Exception as e:
            logger.error(f"Error in chat mode: {e}")
            if tm:
                logger.debug("Cleaning up after error")
                from mcp_cli.run_command import _safe_close

                await _safe_close(tm)
            raise
        finally:
            if tm:
                logger.debug("Final cleanup of tool manager")
                from mcp_cli.run_command import _safe_close

                await _safe_close(tm)

    try:
        asyncio.run(_start_chat())
    except KeyboardInterrupt:
        output.warning("\nInterrupted")
        logger.debug("Chat mode interrupted by user")
    except asyncio.TimeoutError:
        logger.debug("Exiting due to timeout")
    finally:
        restore_terminal()
        raise typer.Exit()


@app.command("interactive", help="Start interactive command mode.")
def _interactive_command(
    config_file: str = typer.Option(
        "server_config.json", help="Configuration file path"
    ),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    provider: Optional[str] = typer.Option(None, help="LLM provider name"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging"
    ),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
    theme: str = typer.Option(
        "default", "--theme", help="UI theme (default, dark, light, minimal, terminal)"
    ),
    confirm_mode: str = typer.Option(
        None,
        "--confirm-mode",
        help="Tool confirmation mode: always, never, or smart (risk-based)",
    ),
    init_timeout: float = typer.Option(
        120.0,
        "--init-timeout",
        help="Server initialization timeout in seconds",
    ),
) -> None:
    """Start interactive command mode."""
    # Re-configure logging based on user options
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)

    # Set confirmation mode if specified
    if confirm_mode:
        from mcp_cli.utils.preferences import get_preference_manager

        pref_manager = get_preference_manager()
        if confirm_mode.lower() in ["always", "never", "smart"]:
            pref_manager.set_tool_confirmation_mode(confirm_mode.lower())
        else:
            output.print(f"[red]Invalid confirmation mode: {confirm_mode}[/red]")
            output.print("[dim]Valid modes: always, never, smart[/dim]")
            raise typer.Exit(1)

    # Set UI theme - use preference if not specified
    from mcp_cli.utils.preferences import get_preference_manager

    pref_manager = get_preference_manager()

    if theme and theme != "default":
        # User specified theme via command line
        set_theme(theme)
        pref_manager.set_theme(theme)  # Save it as preference
    else:
        # Use saved preference
        saved_theme = pref_manager.get_theme()
        set_theme(saved_theme)

    logger.debug("Starting interactive command mode")

    # Use ModelManager to get active provider/model if not specified
    from mcp_cli.model_manager import ModelManager

    model_manager = ModelManager()

    # Smart provider/model resolution:
    # 1. If both specified: use both
    # 2. If only provider specified: use provider + its default model
    # 3. If neither specified: use active provider + active model
    if provider and model:
        # Both specified explicitly
        effective_provider = provider
        effective_model = model
        logger.debug(f"Using explicit provider/model: {provider}/{model}")
    elif provider and not model:
        # Provider specified, get its default model
        effective_provider = provider
        effective_model = model_manager.get_default_model(provider)
        logger.debug(f"Using provider with default model: {provider}/{effective_model}")
    elif not provider and model:
        # Model specified, use current provider
        effective_provider = model_manager.get_active_provider()
        effective_model = model
        logger.debug(
            f"Using current provider with specified model: {effective_provider}/{model}"
        )
    else:
        # Neither specified, use active configuration
        effective_provider = model_manager.get_active_provider()
        effective_model = model_manager.get_active_model()
        logger.debug(
            f"Using active configuration: {effective_provider}/{effective_model}"
        )

    servers, _, server_names = process_options(
        server,
        disable_filesystem,
        effective_provider,
        effective_model,
        config_file,
        quiet=quiet,
    )

    from mcp_cli.interactive.shell import interactive_mode

    run_command_sync(
        interactive_mode,
        config_file,
        servers,
        extra_params={
            "provider": effective_provider,  # Use effective values
            "model": effective_model,  # Use effective values
            "server_names": server_names,
            "api_base": api_base,
            "api_key": api_key,
            "init_timeout": init_timeout,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Direct command registration with proper command structure
# ──────────────────────────────────────────────────────────────────────────────

# Register unified commands
logger.debug("Registering unified commands")
register_unified_commands()

# Register CLI commands from unified registry

logger.debug("Registering CLI commands from unified registry")
# Don't register all commands here - let each mode handle its own
# CLICommandAdapter.register_with_typer(app)

# Try unified registry for core commands
core_commands = ["cmd", "ping"]
registry_registered = []

for command_name in core_commands:
    cmd = registry.get(command_name, mode=CommandMode.CLI)
    if cmd:
        # For now, we'll handle these directly through existing subcommands
        registry_registered.append(command_name)
        logger.debug(f"Successfully found command in unified registry: {command_name}")

# Direct registration of tool-related commands
direct_registered = ["chat"]  # Chat is registered directly via @app.command


# Shared provider command function
def _run_provider_command(args, log_prefix="Provider command"):
    """Shared function to run provider commands."""
    from mcp_cli.commands.actions.providers import provider_action_async

    # Initialize context for the provider command
    initialize_context()

    try:
        asyncio.run(provider_action_async(args))
    except Exception as e:
        output.error(f"Error: {e}")
        raise typer.Exit(1)


# Function to configure logging for individual commands
def _setup_command_logging(
    quiet: bool, verbose: bool, log_level: str, theme: str = "default"
):
    """Set up logging and theme for individual commands."""
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)
    if theme:
        set_theme(theme)


# Provider command - FIXED to handle arguments properly
@app.command("provider", help="Manage LLM providers")
def provider_command(
    subcommand: Optional[str] = typer.Argument(
        None, help="Subcommand: list, config, diagnostic, set, or provider name"
    ),
    provider_name: Optional[str] = typer.Argument(
        None, help="Provider name (for set or switch commands)"
    ),
    key: Optional[str] = typer.Argument(None, help="Config key (for set command)"),
    value: Optional[str] = typer.Argument(None, help="Config value (for set command)"),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model name (for switch commands)"
    ),
    config_file: str = typer.Option(
        "server_config.json", help="Configuration file path"
    ),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging"
    ),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
    theme: str = typer.Option("default", "--theme", help="UI theme"),
) -> None:
    """Manage LLM providers."""
    # Configure logging and theme for this command
    _setup_command_logging(quiet, verbose, log_level, theme)

    # Build arguments list for the provider action
    args: List[str] = []

    # Handle different command patterns
    if subcommand is None:
        # No arguments - show status
        args = []
    elif subcommand in ["list", "config", "diagnostic", "custom"]:
        # Command without provider name
        args = [subcommand]
        if provider_name and subcommand == "diagnostic":
            # diagnostic can take a provider name
            args.append(provider_name)
    elif subcommand == "add":
        # add command: add <name> <api_base> [models...]
        if not provider_name or not key:
            output.error(
                "add command requires: provider add <name> <api_base> [model1 model2 ...]"
            )
            raise typer.Exit(1)
        args = [subcommand, provider_name, key]  # key is used as api_base
        if value:
            args.append(value)  # value is the first model
    elif subcommand == "remove":
        # remove command: remove <name>
        if not provider_name:
            output.error("remove command requires: provider remove <name>")
            raise typer.Exit(1)
        args = [subcommand, provider_name]
    elif subcommand == "set":
        # set command: set <provider> <key> <value>
        if not provider_name or not key or not value:
            output.error("set command requires: provider set <provider> <key> <value>")
            raise typer.Exit(1)
        args = [subcommand, provider_name, key, value]
    else:
        # Assume subcommand is a provider name for switching
        args = [subcommand]  # provider name
        if provider_name:
            # Second argument is model name
            args.append(provider_name)
        elif model:
            # Model specified via --model option
            args.append(model)

    _run_provider_command(args)


direct_registered.append("provider")


# ADD: providers command as alias to provider (for consistency)
@app.command("providers", help="List LLM providers (defaults to list)")
def providers_command(
    subcommand: Optional[str] = typer.Argument(
        None, help="Subcommand: list, config, diagnostic, set, or provider name"
    ),
    provider_name: Optional[str] = typer.Argument(
        None, help="Provider name (for set or switch commands)"
    ),
    key: Optional[str] = typer.Argument(None, help="Config key (for set command)"),
    value: Optional[str] = typer.Argument(None, help="Config value (for set command)"),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model name (for switch commands)"
    ),
    config_file: str = typer.Option(
        "server_config.json", help="Configuration file path"
    ),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging"
    ),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
    theme: str = typer.Option("default", "--theme", help="UI theme"),
) -> None:
    """List LLM providers (plural form defaults to 'list' command)."""
    # Configure logging and theme for this command
    _setup_command_logging(quiet, verbose, log_level, theme)

    # Build arguments list for the provider action
    args = []

    # Handle different command patterns
    if subcommand is None:
        # CHANGED: No arguments for "providers" defaults to list (not status)
        args = ["list"]
    elif subcommand in ["list", "config", "diagnostic"]:
        # Command without provider name
        args = [subcommand]
        if provider_name and subcommand == "diagnostic":
            # diagnostic can take a provider name
            args.append(provider_name)
    elif subcommand == "set":
        # set command: set <provider> <key> <value>
        if not provider_name or not key or not value:
            output.error("set command requires: providers set <provider> <key> <value>")
            raise typer.Exit(1)
        args = [subcommand, provider_name, key, value]
    else:
        # Assume subcommand is a provider name for switching
        args = [subcommand]  # provider name
        if provider_name:
            # Second argument is model name
            args.append(provider_name)
        elif model:
            # Model specified via --model option
            args.append(model)

    _run_provider_command(args, "Providers command")


direct_registered.append("providers")


# Tools command - create a direct command that wraps the tools functionality
@app.command("tools", help="List available tools")
def tools_command(
    all: bool = typer.Option(False, "--all", help="Show detailed tool information"),
    raw: bool = typer.Option(False, "--raw", help="Show raw JSON definitions"),
    config_file: str = typer.Option(
        "server_config.json", help="Configuration file path"
    ),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    provider: str = typer.Option("openai", help="LLM provider name"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging"
    ),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
    theme: str = typer.Option("default", "--theme", help="UI theme"),
) -> None:
    """List unique tools across all connected servers."""
    # Configure logging and theme for this command
    _setup_command_logging(quiet, verbose, log_level, theme)

    # Process options
    servers, _, server_names = process_options(
        server, disable_filesystem, provider, model, config_file, quiet=quiet
    )

    # Import and use the tools action - USE ASYNC VERSION
    from mcp_cli.commands.actions.tools import tools_action_async

    # Execute via run_command_sync with async wrapper
    async def _tools_wrapper(**params):
        return await tools_action_async(
            show_details=params.get("all", False),
            show_raw=params.get("raw", False),
        )

    run_command_sync(
        _tools_wrapper,
        config_file,
        servers,
        extra_params={
            "all": all,
            "raw": raw,
            "server_names": server_names,
        },
    )


direct_registered.append("tools")


# Servers command
@app.command(
    "servers", help="List connected MCP servers with comprehensive information"
)
def servers_command(
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed information including capabilities and transport details",
    ),
    capabilities: bool = typer.Option(
        False,
        "--capabilities",
        "--caps",
        "-c",
        help="Include server capability information in output",
    ),
    transport: bool = typer.Option(
        False,
        "--transport",
        "--trans",
        "-t",
        help="Include transport/connection details",
    ),
    output_format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, tree, or json"
    ),
    config_file: str = typer.Option(
        "server_config.json", help="Configuration file path"
    ),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    provider: str = typer.Option("openai", help="LLM provider name"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging"
    ),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
    theme: str = typer.Option("default", "--theme", help="UI theme"),
) -> None:
    """Show connected servers with comprehensive MCP information."""
    # Configure logging and theme for this command
    _setup_command_logging(quiet, verbose, log_level, theme)

    # Validate format
    valid_formats = ["table", "tree", "json"]
    if output_format.lower() not in valid_formats:
        output.error(
            f"Invalid format '{output_format}'. Must be one of: {', '.join(valid_formats)}"
        )
        raise typer.Exit(code=1)

    # Auto-enable features for detailed view
    if detailed:
        capabilities = True
        transport = True

    servers, _, server_names = process_options(
        server, disable_filesystem, provider, model, config_file, quiet=quiet
    )

    from mcp_cli.commands.actions.servers import servers_action_async

    async def _servers_wrapper(**params):
        return await servers_action_async(
            detailed=params.get("detailed", False),
            show_capabilities=params.get("capabilities", False),
            show_transport=params.get("transport", False),
            output_format=params.get("output_format", "table"),
        )

    run_command_sync(
        _servers_wrapper,
        config_file,
        servers,
        extra_params={
            "detailed": detailed,
            "capabilities": capabilities,
            "transport": transport,
            "output_format": output_format.lower(),
            "server_names": server_names,
        },
    )


direct_registered.append("servers")


# Resources command
@app.command("resources", help="List available resources")
def resources_command(
    config_file: str = typer.Option(
        "server_config.json", help="Configuration file path"
    ),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    provider: str = typer.Option("openai", help="LLM provider name"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging"
    ),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
    theme: str = typer.Option("default", "--theme", help="UI theme"),
) -> None:
    """Show all recorded resources."""
    # Configure logging and theme for this command
    _setup_command_logging(quiet, verbose, log_level, theme)

    servers, _, server_names = process_options(
        server, disable_filesystem, provider, model, config_file
    )

    from mcp_cli.commands.resources import resources_action_async

    async def _resources_wrapper(**params):
        return await resources_action_async()

    run_command_sync(
        _resources_wrapper,
        config_file,
        servers,
        extra_params={"server_names": server_names},
    )


direct_registered.append("resources")


# Prompts command
@app.command("prompts", help="List available prompts")
def prompts_command(
    config_file: str = typer.Option(
        "server_config.json", help="Configuration file path"
    ),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    provider: str = typer.Option("openai", help="LLM provider name"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging"
    ),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
    theme: str = typer.Option("default", "--theme", help="UI theme"),
) -> None:
    """Show all prompt templates."""
    # Configure logging and theme for this command
    _setup_command_logging(quiet, verbose, log_level, theme)

    servers, _, server_names = process_options(
        server, disable_filesystem, provider, model, config_file
    )

    from mcp_cli.commands.prompts import prompts_action_async

    async def _prompts_wrapper(**params):
        return await prompts_action_async()

    run_command_sync(
        _prompts_wrapper,
        config_file,
        servers,
        extra_params={"server_names": server_names},
    )


direct_registered.append("prompts")


# Models command - show available models for current or specified provider
@app.command("models", help="List available models for a provider")
def models_command(
    provider_name: Optional[str] = typer.Argument(
        None, help="Provider name (defaults to current)"
    ),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging"
    ),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
    theme: str = typer.Option("default", "--theme", help="UI theme"),
) -> None:
    """List available models for a provider."""
    # Configure logging and theme for this command
    _setup_command_logging(quiet, verbose, log_level, theme)

    from mcp_cli.model_manager import ModelManager
    from rich.table import Table

    model_manager = ModelManager()

    # Use specified provider or current active provider
    target_provider = provider_name or model_manager.get_active_provider()
    current_provider = model_manager.get_active_provider()
    current_model = model_manager.get_active_model()

    # Validate provider exists
    if not model_manager.validate_provider(target_provider):
        output.error(f"Unknown provider: {target_provider}")
        output.info(f"Available providers: {', '.join(model_manager.list_providers())}")
        return

    # Get provider info
    provider_info = model_manager.get_provider_info(target_provider)
    default_model = provider_info.get("default_model", "Not specified")
    available_models = model_manager.get_available_models(target_provider)

    # Create table
    is_current_provider = target_provider == current_provider
    title = f"Models for {target_provider}"
    if is_current_provider:
        title += " (Current Provider)"

    table = Table(title=title)
    table.add_column("Model", style="green")
    table.add_column("Status", style="yellow")

    # Add default model
    default_status = "Default"
    if is_current_provider and default_model == current_model:
        default_status = "Current & Default"
    elif is_current_provider and current_model in available_models:
        default_status = "Default"

    if default_model and default_model != "Not specified":
        table.add_row(default_model, default_status)

    # Add current model if different from default and this is current provider
    if is_current_provider and current_model != default_model and current_model:
        table.add_row(current_model, "Current")

    # Add other available models (first 10)
    other_models = [
        m for m in available_models[:10] if m not in [default_model, current_model]
    ]
    for model in other_models:
        table.add_row(model, "Available")

    if len(available_models) > 10:
        table.add_row(f"... and {len(available_models) - 10} more", "Available")

    output.print_table(table)

    # Show additional info
    output.print(f"\n[dim]Provider:[/dim] {target_provider}")
    output.print(f"[dim]Total models:[/dim] {len(available_models)}")
    output.print(f"[dim]API Base:[/dim] {provider_info.get('api_base') or 'Default'}")
    if provider_info.get("has_api_key"):
        output.print("[dim]API Key:[/dim] ********")
    else:
        output.print("[dim]API Key:[/dim] [red]Not configured[/red]")

    # Show switch command if not current provider
    if not is_current_provider:
        output.print(f"\n[dim]To switch:[/dim] mcp-cli provider {target_provider}")


direct_registered.append("models")


# Theme command - manage UI themes
@app.command("theme", help="Manage UI themes for MCP CLI")
def theme_command(
    theme_name: Optional[str] = typer.Argument(None, help="Theme name to switch to"),
    list_themes: bool = typer.Option(
        False, "--list", "-l", help="List all available themes"
    ),
    select: bool = typer.Option(
        False, "--select", "-s", help="Interactive theme selection"
    ),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging"
    ),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
) -> None:
    """Manage UI themes for MCP CLI."""
    # Configure logging for this command
    _setup_command_logging(
        quiet, verbose, log_level, "default"
    )  # Start with default theme

    from mcp_cli.commands.theme import theme_command as theme_cmd

    # Execute theme command
    theme_cmd(theme_name, list_themes, select)


direct_registered.append("theme")


# Token command - manage individual tokens (use 'tokens' to list all)
@app.command("token", help="Manage individual tokens (use 'tokens' to list)")
def token_command(
    action: str = typer.Argument(
        ...,
        help="Action: list, set, get, delete, clear, backends, set-provider, get-provider, delete-provider",
    ),
    name: Optional[str] = typer.Argument(
        None, help="Token/provider name (for set/get/delete actions)"
    ),
    token_type: str = typer.Option(
        "bearer", "--type", "-t", help="Token type (bearer, api-key, generic)"
    ),
    value: Optional[str] = typer.Option(None, "--value", help="Token value"),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Provider name (for API keys)"
    ),
    namespace: Optional[str] = typer.Option(
        None, "--namespace", "-n", help="Storage namespace"
    ),
    show_oauth: bool = typer.Option(
        True, "--show-oauth/--no-oauth", help="Show OAuth tokens (list)"
    ),
    show_bearer: bool = typer.Option(
        True, "--show-bearer/--no-bearer", help="Show bearer tokens (list)"
    ),
    show_api_keys: bool = typer.Option(
        True, "--show-api-keys/--no-api-keys", help="Show API keys (list)"
    ),
    show_providers: bool = typer.Option(
        True, "--show-providers/--no-providers", help="Show provider tokens (list)"
    ),
    is_oauth: bool = typer.Option(
        False, "--is-oauth", help="Delete OAuth token (delete)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation (clear)"
    ),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
) -> None:
    """Manage stored tokens and credentials."""
    # Configure logging for this command
    _setup_command_logging(quiet, verbose, log_level, "default")

    from mcp_cli.commands.actions.token import (
        token_list_action_async,
        token_set_action_async,
        token_get_action_async,
        token_delete_action_async,
        token_clear_action_async,
        token_backends_action_async,
        token_set_provider_action_async,
        token_get_provider_action_async,
        token_delete_provider_action_async,
    )
    import asyncio

    async def _token_wrapper():
        if action == "list":
            return await token_list_action_async(
                namespace=namespace,
                show_oauth=show_oauth,
                show_bearer=show_bearer,
                show_api_keys=show_api_keys,
                show_providers=show_providers,
            )
        elif action == "set":
            if not name:
                output.error("Token name is required for 'set' action")
                raise typer.Exit(1)
            from mcp_cli.commands.models import TokenSetParams

            # Use token_type as default namespace if not specified
            default_namespace = (
                token_type if token_type in ["bearer", "api-key"] else "generic"
            )
            return await token_set_action_async(
                TokenSetParams(
                    name=name,
                    token_type=token_type,
                    value=value,
                    provider=provider,
                    namespace=namespace or default_namespace,
                )
            )
        elif action == "get":
            if not name:
                output.error("Token name is required for 'get' action")
                raise typer.Exit(1)
            # Use token_type as default namespace if not specified
            default_namespace = (
                token_type if token_type in ["bearer", "api-key"] else "generic"
            )
            return await token_get_action_async(
                name=name,
                namespace=namespace or default_namespace,
            )
        elif action == "delete":
            if not name:
                output.error("Token name is required for 'delete' action")
                raise typer.Exit(1)
            from mcp_cli.commands.models import TokenDeleteParams

            # Use token_type as default namespace if not specified
            default_namespace = (
                token_type if token_type in ["bearer", "api-key"] else "generic"
            )
            return await token_delete_action_async(
                TokenDeleteParams(
                    name=name,
                    namespace=namespace or default_namespace,
                    oauth=is_oauth,
                )
            )
        elif action == "clear":
            return await token_clear_action_async(
                namespace=namespace,
                force=force,
            )
        elif action == "backends":
            return await token_backends_action_async()
        elif action == "set-provider":
            if not name:
                output.error("Provider name is required for 'set-provider' action")
                raise typer.Exit(1)
            return await token_set_provider_action_async(
                provider=name,
                api_key=value,
            )
        elif action == "get-provider":
            if not name:
                output.error("Provider name is required for 'get-provider' action")
                raise typer.Exit(1)
            return await token_get_provider_action_async(provider=name)
        elif action == "delete-provider":
            if not name:
                output.error("Provider name is required for 'delete-provider' action")
                raise typer.Exit(1)
            return await token_delete_provider_action_async(provider=name)
        else:
            output.error(f"Unknown action: {action}")
            output.hint(
                "Valid actions: list, set, get, delete, clear, backends, set-provider, get-provider, delete-provider"
            )
            raise typer.Exit(1)

    # Run the async function
    asyncio.run(_token_wrapper())


direct_registered.append("token")


# Tokens command - plural form defaults to list (for consistency with providers)
@app.command("tokens", help="List stored tokens (defaults to list)")
def tokens_command(
    action: Optional[str] = typer.Argument(
        None,
        help="Action: list, set, get, delete, clear, backends, set-provider, get-provider, delete-provider",
    ),
    name: Optional[str] = typer.Argument(
        None, help="Token/provider name (for set/get/delete actions)"
    ),
    token_type: str = typer.Option(
        "bearer", "--type", "-t", help="Token type (bearer, api-key, generic)"
    ),
    value: Optional[str] = typer.Option(None, "--value", help="Token value"),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Provider name (for API keys)"
    ),
    namespace: Optional[str] = typer.Option(
        None, "--namespace", "-n", help="Storage namespace"
    ),
    show_oauth: bool = typer.Option(
        True, "--show-oauth/--no-oauth", help="Show OAuth tokens (list)"
    ),
    show_bearer: bool = typer.Option(
        True, "--show-bearer/--no-bearer", help="Show bearer tokens (list)"
    ),
    show_api_keys: bool = typer.Option(
        True, "--show-api-keys/--no-api-keys", help="Show API keys (list)"
    ),
    show_providers: bool = typer.Option(
        True, "--show-providers/--no-providers", help="Show provider tokens (list)"
    ),
    is_oauth: bool = typer.Option(
        False, "--is-oauth", help="Delete OAuth token (delete)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation (clear)"
    ),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
) -> None:
    """List stored tokens (plural form defaults to 'list' action)."""
    # Configure logging for this command
    _setup_command_logging(quiet, verbose, log_level, "default")

    from mcp_cli.commands.actions.token import (
        token_list_action_async,
        token_set_action_async,
        token_get_action_async,
        token_delete_action_async,
        token_clear_action_async,
        token_backends_action_async,
        token_set_provider_action_async,
        token_get_provider_action_async,
        token_delete_provider_action_async,
    )
    import asyncio

    async def _tokens_wrapper():
        # Default to 'list' if no action specified (like providers command)
        effective_action = action or "list"

        if effective_action == "list":
            from mcp_cli.commands.models import TokenListParams
            import json
            from pathlib import Path

            # Load server names from config
            server_names = []
            try:
                config_path = Path("server_config.json")
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        server_names = list(config.get("mcpServers", {}).keys())
            except Exception:
                pass  # Silently ignore config load errors

            params = TokenListParams(
                namespace=namespace,
                show_oauth=show_oauth,
                show_bearer=show_bearer,
                show_api_keys=show_api_keys,
                show_providers=show_providers,
                server_names=server_names,
            )
            return await token_list_action_async(params)
        elif effective_action == "set":
            if not name:
                output.error("Token name is required for 'set' action")
                raise typer.Exit(1)
            from mcp_cli.commands.models import TokenSetParams

            params = TokenSetParams(
                name=name,
                token_type=token_type,
                value=value,
                provider=provider,
                namespace=namespace or "generic",
            )
            return await token_set_action_async(params)
        elif effective_action == "get":
            if not name:
                output.error("Token name is required for 'get' action")
                raise typer.Exit(1)
            return await token_get_action_async(
                name=name,
                namespace=namespace or "generic",
            )
        elif effective_action == "delete":
            if not name:
                output.error("Token name is required for 'delete' action")
                raise typer.Exit(1)
            from mcp_cli.commands.models import TokenDeleteParams

            params = TokenDeleteParams(
                name=name,
                namespace=namespace,
                oauth=is_oauth,
            )
            return await token_delete_action_async(params)
        elif effective_action == "clear":
            from mcp_cli.commands.models import TokenClearParams

            params = TokenClearParams(
                namespace=namespace,
                force=force,
            )
            return await token_clear_action_async(params)
        elif effective_action == "backends":
            return await token_backends_action_async()
        elif effective_action == "set-provider":
            if not name:
                output.error("Provider name is required for 'set-provider' action")
                raise typer.Exit(1)
            from mcp_cli.commands.models import TokenProviderParams

            params = TokenProviderParams(
                provider=name,
                api_key=value,
            )
            return await token_set_provider_action_async(params)
        elif effective_action == "get-provider":
            if not name:
                output.error("Provider name is required for 'get-provider' action")
                raise typer.Exit(1)
            from mcp_cli.commands.models import TokenProviderParams

            params = TokenProviderParams(provider=name)
            return await token_get_provider_action_async(params)
        elif effective_action == "delete-provider":
            if not name:
                output.error("Provider name is required for 'delete-provider' action")
                raise typer.Exit(1)
            from mcp_cli.commands.models import TokenProviderParams

            params = TokenProviderParams(provider=name)
            return await token_delete_provider_action_async(params)
        else:
            output.error(f"Unknown action: {effective_action}")
            output.hint(
                "Valid actions: list, set, get, delete, clear, backends, set-provider, get-provider, delete-provider"
            )
            raise typer.Exit(1)

    # Run the async function
    asyncio.run(_tokens_wrapper())


direct_registered.append("tokens")


# Cmd command - Unix-friendly automation mode
@app.command("cmd", help="Command mode for Unix-friendly automation and scripting")
def cmd_command(
    input_file: Optional[str] = typer.Option(
        None, "--input", "-i", help="Input file (use - for stdin)"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file (use - for stdout)"
    ),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Prompt text"),
    tool: Optional[str] = typer.Option(
        None, "--tool", "-t", help="Tool name to execute"
    ),
    tool_args: Optional[str] = typer.Option(
        None, "--tool-args", help="Tool arguments as JSON"
    ),
    system_prompt: Optional[str] = typer.Option(
        None, "--system-prompt", help="Custom system prompt"
    ),
    raw: bool = typer.Option(False, "--raw", help="Raw output without formatting"),
    single_turn: bool = typer.Option(
        False, "--single-turn", help="Disable multi-turn conversation"
    ),
    max_turns: int = typer.Option(30, "--max-turns", help="Maximum conversation turns"),
    config_file: str = typer.Option(
        "server_config.json", help="Configuration file path"
    ),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    provider: Optional[str] = typer.Option(None, help="LLM provider name"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
    theme: str = typer.Option("default", "--theme", help="UI theme"),
    init_timeout: float = typer.Option(
        120.0, "--init-timeout", help="Server initialization timeout in seconds"
    ),
) -> None:
    """Command mode for Unix-friendly automation and scripting."""
    # Configure logging and theme
    _setup_command_logging(quiet, verbose, log_level, theme)

    # Use ModelManager to resolve provider/model
    from mcp_cli.model_manager import ModelManager

    model_manager = ModelManager()

    # Smart provider/model resolution
    if provider and model:
        effective_provider = provider
        effective_model = model
    elif provider and not model:
        effective_provider = provider
        effective_model = model_manager.get_default_model(provider)
    elif not provider and model:
        effective_provider = model_manager.get_active_provider()
        effective_model = model
    else:
        effective_provider = model_manager.get_active_provider()
        effective_model = model_manager.get_active_model()

    # Process server options
    servers, _, server_names = process_options(
        server,
        disable_filesystem,
        effective_provider,
        effective_model,
        config_file,
        quiet=quiet,
    )

    # Import cmd action
    from mcp_cli.commands.actions.cmd import cmd_action_async

    # Execute via run_command_sync
    async def _cmd_wrapper(**params):
        return await cmd_action_async(
            input_file=params.get("input_file"),
            output_file=params.get("output_file"),
            prompt=params.get("prompt"),
            tool=params.get("tool"),
            tool_args=params.get("tool_args"),
            system_prompt=params.get("system_prompt"),
            raw=params.get("raw", False),
            single_turn=params.get("single_turn", False),
            max_turns=params.get("max_turns", 30),
        )

    run_command_sync(
        _cmd_wrapper,
        config_file,
        servers,
        extra_params={
            "input_file": input_file,
            "output_file": output_file,
            "prompt": prompt,
            "tool": tool,
            "tool_args": tool_args,
            "system_prompt": system_prompt,
            "raw": raw,
            "single_turn": single_turn,
            "max_turns": max_turns,
            "server_names": server_names,
            "provider": effective_provider,
            "model": effective_model,
            "api_base": api_base,
            "api_key": api_key,
            "init_timeout": init_timeout,
        },
    )


direct_registered.append("cmd")

# Show what we actually registered
all_registered = registry_registered + direct_registered
output.success("✓ MCP CLI ready")
if all_registered:
    output.info(f"  Available commands: {', '.join(sorted(all_registered))}")
else:
    output.warning("  Warning: No commands were successfully registered!")
output.hint("  Use --help to see all options")


# ──────────────────────────────────────────────────────────────────────────────
# Signal handling
# ──────────────────────────────────────────────────────────────────────────────
def _setup_signal_handlers() -> None:
    """Setup signal handlers for clean shutdown."""

    def handler(sig, _frame):
        logger.debug(f"Received signal {sig}, shutting down")
        restore_terminal()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, handler)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    _setup_signal_handlers()
    atexit.register(restore_terminal)

    try:
        app()
    finally:
        restore_terminal()
        gc.collect()
