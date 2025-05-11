# mcp_cli/interactive/commands/model.py
from typing import Any, List
from .base import InteractiveCommand
from mcp_cli.commands.model import model_action_async


class ModelCommand(InteractiveCommand):
    def __init__(self):
        super().__init__(
            name="model",
            aliases=["m"],
            help_text="View or change the current LLM model.",
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: Any = None,   # unused
        **ctx: Any,
    ) -> None:
        await model_action_async(args, context=ctx)
