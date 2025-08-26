# src/mcp_cli/main.py
"""Entry-point for the MCP CLI."""

from __future__ import annotations

import asyncio
import atexit
import gc
import signal
import sys
from typing import Optional

import typer

# ──────────────────────────────────────────────────────────────────────────────
# CRITICAL: Set up silent environment IMMEDIATELY before any other imports
# This prevents MCP server noise from appearing during module imports
# ──────────────────────────────────────────────────────────────────────────────
from mcp_cli.logging_config import (
    setup_logging,
    get_logger,
    setup_silent_mcp_environment,
)

# FIRST: Set environment variables to silence MCP servers before they start
setup_silent_mcp_environment()

# THEN: Set up default clean logging immediately
# This will be overridden later if user specifies different options
setup_logging(level="ERROR", quiet=False, verbose=False)

# ──────────────────────────────────────────────────────────────────────────────
# Now safe to import components that might start MCP servers
# ──────────────────────────────────────────────────────────────────────────────
from mcp_cli.cli.commands import register_all_commands
from mcp_cli.cli.registry import CommandRegistry
from mcp_cli.run_command import run_command_sync

# Import UI components from chuk_term
from chuk_term.ui import (
    output,
    restore_terminal,
)
from chuk_term.ui.theme import set_theme

from mcp_cli.cli_options import process_options

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
) -> None:
    """MCP CLI - If no subcommand is given, start chat mode."""

    # Re-configure logging based on user options (this overrides the default ERROR level)
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)

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
        from mcp_cli.commands.provider import provider_action_async
        from mcp_cli.model_manager import ModelManager

        context = {"model_manager": ModelManager()}

        try:
            asyncio.run(provider_action_async([provider], context=context))
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

    # Validate provider if specified
    if provider:
        if not model_manager.validate_provider(provider):
            available = ", ".join(model_manager.list_providers())
            output.error(f"Unknown provider: {provider}")
            output.info(f"Available providers: {available}")
            output.tip(f"Did you mean to run: mcp-cli provider {provider}")
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

    # Start chat mode directly
    async def _start_chat():
        tm = None
        try:
            logger.debug("Initializing tool manager")
            from mcp_cli.run_command import _init_tool_manager

            tm = await _init_tool_manager(config_file, servers, server_names)

            logger.debug("Starting chat mode handler")
            success = await handle_chat_mode(
                tool_manager=tm,
                provider=effective_provider,  # Use effective values
                model=effective_model,  # Use effective values
                api_base=api_base,
                api_key=api_key,
            )
            logger.debug(f"Chat mode completed with success: {success}")
        finally:
            if tm:
                logger.debug("Cleaning up tool manager")
                from mcp_cli.run_command import _safe_close

                await _safe_close(tm)

    try:
        asyncio.run(_start_chat())
    except KeyboardInterrupt:
        output.warning("\nInterrupted")
        logger.debug("Chat mode interrupted by user")
    finally:
        restore_terminal()
        raise typer.Exit()


# ──────────────────────────────────────────────────────────────────────────────
# Built-in commands
# ──────────────────────────────────────────────────────────────────────────────
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
) -> None:
    """Start interactive command mode."""
    # Re-configure logging based on user options
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)

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
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Direct command registration with proper command structure
# ──────────────────────────────────────────────────────────────────────────────

# Register all commands in the registry first (in case some work)
logger.debug("Registering commands from registry")
register_all_commands()

# Try registry-based registration first for core commands
core_commands = ["chat", "cmd", "ping"]  # Remove "provider" from registry
registry_registered = []

for command_name in core_commands:
    cmd = CommandRegistry.get_command(command_name)
    if cmd:
        try:
            cmd.register(app, run_command_sync)
            registry_registered.append(command_name)
            logger.debug(
                f"Successfully registered command via registry: {command_name}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to register command '{command_name}' via registry: {e}"
            )

# Direct registration of tool-related commands
direct_registered = []


# Shared provider command function
def _run_provider_command(args, log_prefix="Provider command"):
    """Shared function to run provider commands."""
    from mcp_cli.commands.provider import provider_action_async
    from mcp_cli.model_manager import ModelManager

    context = {"model_manager": ModelManager()}

    try:
        asyncio.run(provider_action_async(args, context=context))
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
    args = []

    # Handle different command patterns
    if subcommand is None:
        # No arguments - show status
        args = []
    elif subcommand in ["list", "config", "diagnostic"]:
        # Command without provider name
        args = [subcommand]
        if provider_name and subcommand == "diagnostic":
            # diagnostic can take a provider name
            args.append(provider_name)
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
    from mcp_cli.commands.tools import tools_action_async

    # Execute via run_command_sync with async wrapper
    async def _tools_wrapper(tool_manager, **params):
        return await tools_action_async(
            tool_manager,
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

    from mcp_cli.commands.servers import servers_action_async

    async def _servers_wrapper(tool_manager, **params):
        return await servers_action_async(
            tool_manager,
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

    async def _resources_wrapper(tool_manager, **params):
        return await resources_action_async(tool_manager)

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

    async def _prompts_wrapper(tool_manager, **params):
        return await prompts_action_async(tool_manager)

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
