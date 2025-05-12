# mcp_cli/chat/commands/model.py
from typing import Dict, List
from mcp_cli.chat.commands import register_command
from mcp_cli.commands.model import model_action_async


async def cmd_model(parts: List[str], ctx: Dict[str, any]) -> bool:
    await model_action_async(parts[1:], context=ctx)
    return True


register_command("/model", cmd_model)
register_command("/m",     cmd_model)
