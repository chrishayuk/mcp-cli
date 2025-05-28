# src/mcp_cli/main.py
"""Entry-point for the MCP CLI."""
from __future__ import annotations

import asyncio
import atexit
import gc
import logging
import os
import signal
import sys
from typing import Optional

import typer

# ──────────────────────────────────────────────────────────────────────────────
# local imports
# ──────────────────────────────────────────────────────────────────────────────
from mcp_cli.cli.commands import register_all_commands
from mcp_cli.cli.registry import CommandRegistry
from mcp_cli.run_command import run_command_sync
from mcp_cli.ui.ui_helpers import restore_terminal
from mcp_cli.cli_options import process_options
from mcp_cli.logging_config import setup_logging, get_logger

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
    config_file: str = typer.Option("server_config.json", help="Configuration file path"),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    provider: Optional[str] = typer.Option(None, help="LLM provider name"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
) -> None:
    """MCP CLI - If no subcommand is given, start chat mode."""
    
    # FIRST: Set up logging before anything else happens
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)
    
    # If a subcommand was invoked, let it handle things
    if ctx.invoked_subcommand is not None:
        return
    
    # No subcommand - start chat mode (default behavior)
    logger.debug("Starting default chat mode")
    
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
        logger.debug(f"Using current provider with specified model: {effective_provider}/{model}")
    else:
        # Neither specified, use active configuration
        effective_provider = model_manager.get_active_provider()
        effective_model = model_manager.get_active_model()
        logger.debug(f"Using active configuration: {effective_provider}/{effective_model}")
    
    servers, _, server_names = process_options(
        server, disable_filesystem, effective_provider, effective_model, config_file
    )

    from mcp_cli.chat.chat_handler import handle_chat_mode
    from mcp_cli.tools.manager import ToolManager

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
                model=effective_model,        # Use effective values
                api_base=api_base,
                api_key=api_key
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
        print("\n[yellow]Interrupted[/yellow]")
        logger.debug("Chat mode interrupted by user")
    except Exception as e:
        print(f"[red]Error:[/red] {e}")
        logger.error(f"Chat mode failed: {e}", exc_info=True)
    finally:
        restore_terminal()
        raise typer.Exit()


# ──────────────────────────────────────────────────────────────────────────────
# Built-in commands
# ──────────────────────────────────────────────────────────────────────────────
@app.command("interactive", help="Start interactive command mode.")
def _interactive_command(
    config_file: str = typer.Option("server_config.json", help="Configuration file path"),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    provider: Optional[str] = typer.Option(None, help="LLM provider name"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
) -> None:
    """Start interactive command mode."""
    # Set up logging first
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)
    
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
        logger.debug(f"Using current provider with specified model: {effective_provider}/{model}")
    else:
        # Neither specified, use active configuration
        effective_provider = model_manager.get_active_provider()
        effective_model = model_manager.get_active_model()
        logger.debug(f"Using active configuration: {effective_provider}/{effective_model}")
    
    servers, _, server_names = process_options(
        server, disable_filesystem, effective_provider, effective_model, config_file
    )

    from mcp_cli.interactive.shell import interactive_mode

    run_command_sync(
        interactive_mode,
        config_file,
        servers,
        extra_params={
            "provider": effective_provider,  # Use effective values
            "model": effective_model,        # Use effective values
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
            logger.debug(f"Successfully registered command via registry: {command_name}")
        except Exception as e:
            logger.warning(f"Failed to register command '{command_name}' via registry: {e}")

# Direct registration of tool-related commands
direct_registered = []

# Provider command - create a direct command
@app.command("provider", help="Manage LLM providers")
def provider_command(
    subcommand: str = typer.Argument("show", help="Subcommand: show, list, config, diagnostic, set"),
    provider_name: str = typer.Argument(None, help="Provider name (for set or switch commands)"),
    key: str = typer.Argument(None, help="Config key (for set command)"),
    value: str = typer.Argument(None, help="Config value (for set command)"),
    model: str = typer.Option(None, "--model", help="Model name (for switch commands)"),
    config_file: str = typer.Option("server_config.json", help="Configuration file path"),
    server: str = typer.Option(None, help="Server to connect to"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
) -> None:
    """Manage LLM providers."""
    # Set up logging
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)
    
    # Import the provider action
    from mcp_cli.commands.provider import provider_action_async
    from mcp_cli.model_manager import ModelManager
    
    # Build arguments list for the provider action
    args = []
    if subcommand and subcommand != "show":
        args.append(subcommand)
        if provider_name:
            args.append(provider_name)
            if key and value:  # For set command
                args.extend([key, value])
            elif model:  # For switch with model
                args.append(model)
    
    # Create context
    context = {
        "model_manager": ModelManager(),
    }
    
    # Run the provider action directly
    import asyncio
    asyncio.run(provider_action_async(args, context=context))

direct_registered.append("provider")

# Tools command - create a direct command that wraps the tools functionality
@app.command("tools", help="List available tools")
def tools_command(
    all: bool = typer.Option(False, "--all", help="Show detailed tool information"),
    raw: bool = typer.Option(False, "--raw", help="Show raw JSON definitions"),
    config_file: str = typer.Option("server_config.json", help="Configuration file path"),
    server: str = typer.Option(None, help="Server to connect to"),
    provider: str = typer.Option("openai", help="LLM provider name"),
    model: str = typer.Option(None, help="Model name"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
) -> None:
    """List unique tools across all connected servers."""
    # Set up logging
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)
    
    # Process options
    servers, _, server_names = process_options(
        server, disable_filesystem, provider, model, config_file
    )
    
    # Import and use the tools action - USE ASYNC VERSION
    from mcp_cli.commands.tools import tools_action_async
    
    # Execute via run_command_sync with async wrapper
    async def _tools_wrapper(tool_manager, **params):
        return await tools_action_async(tool_manager, show_details=params.get('all', False), show_raw=params.get('raw', False))
    
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
@app.command("servers", help="List connected MCP servers")
def servers_command(
    config_file: str = typer.Option("server_config.json", help="Configuration file path"),
    server: str = typer.Option(None, help="Server to connect to"),
    provider: str = typer.Option("openai", help="LLM provider name"),
    model: str = typer.Option(None, help="Model name"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
) -> None:
    """Show connected servers with status & tool counts."""
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)
    
    servers, _, server_names = process_options(
        server, disable_filesystem, provider, model, config_file
    )
    
    from mcp_cli.commands.servers import servers_action_async
    
    async def _servers_wrapper(tool_manager, **params):
        return await servers_action_async(tool_manager)
    
    run_command_sync(
        _servers_wrapper,
        config_file,
        servers,
        extra_params={"server_names": server_names},
    )

direct_registered.append("servers")

# Resources command  
@app.command("resources", help="List available resources")
def resources_command(
    config_file: str = typer.Option("server_config.json", help="Configuration file path"),
    server: str = typer.Option(None, help="Server to connect to"),
    provider: str = typer.Option("openai", help="LLM provider name"),
    model: str = typer.Option(None, help="Model name"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
) -> None:
    """Show all recorded resources."""
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)
    
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
    config_file: str = typer.Option("server_config.json", help="Configuration file path"),
    server: str = typer.Option(None, help="Server to connect to"),
    provider: str = typer.Option("openai", help="LLM provider name"),
    model: str = typer.Option(None, help="Model name"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
) -> None:
    """Show all prompt templates."""
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)
    
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
    provider_name: str = typer.Argument(None, help="Provider name (defaults to current)"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
) -> None:
    """List available models for a provider."""
    setup_logging(level=log_level, quiet=quiet, verbose=verbose)
    
    from mcp_cli.model_manager import ModelManager
    from mcp_cli.utils.rich_helpers import get_console
    from rich.table import Table
    
    console = get_console()
    model_manager = ModelManager()
    
    # Use specified provider or current active provider
    target_provider = provider_name or model_manager.get_active_provider()
    current_provider = model_manager.get_active_provider()
    current_model = model_manager.get_active_model()
    
    # Validate provider exists
    if not model_manager.validate_provider(target_provider):
        console.print(f"[red]Unknown provider:[/red] {target_provider}")
        console.print(f"[yellow]Available providers:[/yellow] {', '.join(model_manager.list_providers())}")
        return
    
    # Get provider configuration
    provider_config = model_manager.get_provider_config(target_provider)
    default_model = provider_config.get("default_model", "Not specified")
    
    # Create table
    is_current_provider = target_provider == current_provider
    title = f"Models for {target_provider}"
    if is_current_provider:
        title += " (Current Provider)"
        
    table = Table(title=title)
    table.add_column("Type", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Status", style="yellow")
    
    # Add default model
    default_status = "Available"
    if is_current_provider and default_model == current_model:
        default_status = "Current & Default"
    elif is_current_provider:
        default_status = "Default"
        
    table.add_row("Default", default_model, default_status)
    
    # Add current model if different from default and this is current provider
    if is_current_provider and current_model != default_model:
        table.add_row("Active", current_model, "Current")
    
    console.print(table)
    
    # Show additional info
    console.print(f"\n[dim]Provider:[/dim] {target_provider}")
    console.print(f"[dim]API Base:[/dim] {provider_config.get('api_base', 'Default')}")
    if provider_config.get('api_key'):
        console.print(f"[dim]API Key:[/dim] ********")
    else:
        console.print(f"[dim]API Key:[/dim] [red]Not configured[/red]")
    
    # Show switch command if not current provider
    if not is_current_provider:
        console.print(f"\n[dim]To switch:[/dim] mcp-cli provider {target_provider}")

direct_registered.append("models")

# Show what we actually registered
all_registered = registry_registered + direct_registered
print("✓ MCP CLI ready")
if all_registered:
    print(f"  Available commands: {', '.join(sorted(all_registered))}")
else:
    print("  Warning: No commands were successfully registered!")
print("  Use --help to see all options")

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