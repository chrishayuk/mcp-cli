# mcp_cli/cli/registry.py
"""Command registry for MCP CLI.

This module provides a centralized registry for all available CLI commands
in the MCP CLI system, with utilities for registration and discovery.
"""
from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional
import typer

# mcp cli commands
from mcp_cli.cli.commands.base import BaseCommand, CommandFunc, FunctionCommand

# loggeer
logger = logging.getLogger(__name__)

class CommandRegistry:
    """Central registry for all MCP CLI commands."""
    
    _commands: Dict[str, BaseCommand] = {}
    
    @classmethod
    def register(cls, command: BaseCommand) -> None:
        """Register a command in the registry."""
        cls._commands[command.name] = command
    
    @classmethod
    def register_function(cls, name: str, func: CommandFunc, help_text: str = "") -> None:
        """Register a function as a command."""
        command = FunctionCommand(name, func, help_text)
        cls.register(command)
    
    @classmethod
    def get_command(cls, name: str) -> Optional[BaseCommand]:
        """Get a command by name."""
        return cls._commands.get(name)
    
    @classmethod
    def get_all_commands(cls) -> List[BaseCommand]:
        """Get all registered commands."""
        return list(cls._commands.values())
    
    @classmethod
    def register_with_typer(cls, app: typer.Typer, run_command_func: Callable) -> None:
        """Register all commands with a Typer app."""
        for command in cls._commands.values():
            command.register(app, run_command_func)
    
    @classmethod
    def create_subcommand_group(cls, app: typer.Typer, group_name: str, 
                               sub_commands: List[str], run_command_func: Callable) -> None:
        """
        Create a subcommand group (e.g., 'tools') with multiple commands.
        
        Args:
            app: The Typer app to register with
            group_name: Name of the command group (e.g., 'tools')
            sub_commands: List of subcommand names in this group (e.g., ['list', 'call'])
            run_command_func: Function to run commands
        """
        # Create a new Typer subcommand group
        subapp = typer.Typer(help=f"{group_name.capitalize()} commands")
        
        # For each subcommand (like 'list', 'call')
        for sub_cmd_name in sub_commands:
            # Get the full command (e.g., 'tools list')
            full_name = f"{group_name} {sub_cmd_name}"
            cmd = cls.get_command(full_name)
            
            if not cmd:
                logger.warning(f"Command '{full_name}' not found in registry")
                continue
                
            # Define a wrapper function that will call the command
            def create_command_wrapper(command, full_cmd_name):
                # This is a factory function to capture the command variable properly
                
                # Get the original execute method
                original_execute = command.wrapped_execute
                
                # Define the actual wrapper that will become the Typer command
                def command_wrapper(
                    config_file: str = "server_config.json",
                    server: Optional[str] = None,
                    provider: str = "openai",
                    model: Optional[str] = None,
                    disable_filesystem: bool = False,
                    **kwargs
                ):
                    """Dynamically generated wrapper for subcommand."""
                    from mcp_cli.cli_options import process_options
                    
                    # Process options
                    servers, _, server_names = process_options(
                        server, disable_filesystem, provider, model, config_file
                    )
                    
                    # Prepare parameters
                    extra_params = {
                        "provider": provider,
                        "model": model,
                        "server_names": server_names,
                        **kwargs
                    }
                    
                    # Execute the command
                    run_command_func(
                        original_execute,
                        config_file,
                        servers,
                        extra_params=extra_params,
                    )
                
                # Add the docstring from the original command
                command_wrapper.__doc__ = f"{command.help}"
                return command_wrapper
            
            # Create and register the wrapper
            wrapper = create_command_wrapper(cmd, full_name)
            subapp.command(sub_cmd_name)(wrapper)
        
        # Add the subapp to the main app
        app.add_typer(subapp, name=group_name)