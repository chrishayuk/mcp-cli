# src/mcp_cli/commands/definitions/model.py
"""
Unified model command implementation.
Uses the existing enhanced model commands from mcp_cli.commands.model
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandGroup,
    CommandParameter,
    CommandResult,
)


class ModelCommand(CommandGroup):
    """Model command group."""

    def __init__(self):
        super().__init__()
        # Add subcommands
        self.add_subcommand(ModelListCommand())
        self.add_subcommand(ModelSetCommand())
        self.add_subcommand(ModelShowCommand())

    @property
    def name(self) -> str:
        return "models"

    @property
    def aliases(self) -> List[str]:
        return ["model"]

    @property
    def description(self) -> str:
        return "Manage LLM models"

    @property
    def help_text(self) -> str:
        return """
Manage LLM models for the current provider.

Subcommands:
  list  - List available models
  set   - Set the active model
  show  - Show current model

Usage:
  /model               - Show current model and available models
  /models              - List all models (preferred)
  /model <name>        - Switch to a different model
  /model list          - List all models (alternative)
  /model refresh       - Refresh model discovery
  
Examples:
  /model gpt-4o-mini   - Switch to gpt-4o-mini
  /model set gpt-4     - Explicitly set to gpt-4
  /model show          - Show current model
  /model list          - List all available models
"""

    async def execute(self, subcommand: str | None = None, **kwargs) -> CommandResult:
        """Execute the model command - handle direct model switching."""
        from mcp_cli.commands.actions.models import model_action_async
        from mcp_cli.commands.models import ModelActionParams

        # Check if we have args (could be model name or subcommand)
        args = kwargs.get("args", [])

        if not args:
            # No arguments - show current model status
            try:
                from mcp_cli.commands.models import ModelActionParams

                params = ModelActionParams(args=[])
                await model_action_async(params)
                return CommandResult(success=True)
            except Exception as e:
                return CommandResult(
                    success=False, error=f"Failed to show model status: {str(e)}"
                )

        # Check if the first arg is a known subcommand
        first_arg = args[0] if isinstance(args, list) else str(args)

        # Known subcommands that should be handled by subcommand classes
        if first_arg.lower() in [
            "list",
            "ls",
            "set",
            "use",
            "switch",
            "show",
            "current",
            "status",
        ]:
            # Let the parent class handle the subcommand routing
            return await super().execute(**kwargs)

        # Otherwise, treat it as a model name to switch to
        try:
            from mcp_cli.commands.models import ModelActionParams

            # Pass the model name directly to switch
            model_args = args if isinstance(args, list) else [str(args)]
            params = ModelActionParams(args=model_args)
            await model_action_async(params)

            return CommandResult(success=True)
        except Exception as e:
            return CommandResult(
                success=False, error=f"Failed to switch model: {str(e)}"
            )


class ModelListCommand(UnifiedCommand):
    """List available models."""

    @property
    def name(self) -> str:
        return "list"

    @property
    def aliases(self) -> List[str]:
        return ["ls"]

    @property
    def description(self) -> str:
        return "List available models for the current provider"

    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="provider",
                type=str,
                required=False,
                help="Provider to list models for (uses current if not specified)",
            ),
            CommandParameter(
                name="detailed",
                type=bool,
                default=False,
                help="Show detailed model information",
                is_flag=True,
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the model list command."""
        # Import the existing model implementation
        from mcp_cli.commands.actions.models import model_action_async

        try:
            # Use the existing enhanced implementation
            # Pass "list" as the command
            from mcp_cli.commands.models import ModelActionParams

            params = ModelActionParams(args=["list"])
            await model_action_async(params)

            # The existing implementation handles all output directly
            # Just return success
            return CommandResult(success=True, data={"command": "model list"})

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to list models: {str(e)}",
            )


class ModelSetCommand(UnifiedCommand):
    """Set the active model."""

    @property
    def name(self) -> str:
        return "set"

    @property
    def aliases(self) -> List[str]:
        return ["use", "switch"]

    @property
    def description(self) -> str:
        return "Set the active model"

    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="model_name",
                type=str,
                required=True,
                help="Name of the model to set",
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the model set command."""
        # Import the existing model implementation
        from mcp_cli.commands.actions.models import model_action_async

        # Get model name
        model_name = kwargs.get("model_name")
        if not model_name and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list):
                model_name = args_val[0] if args_val else None
            elif isinstance(args_val, str):
                model_name = args_val

        if not model_name:
            return CommandResult(
                success=False,
                error="Model name is required. Usage: /model set <name>",
            )

        try:
            from mcp_cli.commands.models import ModelActionParams

            # Use the existing enhanced implementation
            # Pass the model name directly to switch to it
            params = ModelActionParams(args=[model_name])
            await model_action_async(params)

            # The existing implementation handles all output directly
            return CommandResult(success=True, data={"model": model_name})

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to set model: {str(e)}",
            )


class ModelShowCommand(UnifiedCommand):
    """Show current model."""

    @property
    def name(self) -> str:
        return "show"

    @property
    def aliases(self) -> List[str]:
        return ["current", "status"]

    @property
    def description(self) -> str:
        return "Show the current active model"

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the model show command."""
        # Import the existing model implementation
        from mcp_cli.commands.actions.models import model_action_async

        try:
            # Use the existing enhanced implementation
            # Pass no arguments to show current status
            from mcp_cli.commands.models import ModelActionParams

            params = ModelActionParams(args=[])
            await model_action_async(params)

            # The existing implementation handles all output directly
            return CommandResult(success=True, data={"command": "model show"})

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to get model info: {str(e)}",
            )
