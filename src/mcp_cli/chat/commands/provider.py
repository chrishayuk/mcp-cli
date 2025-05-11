# src/mcp_cli/chat/commands/provider.py
from mcp_cli.commands.provider import provider_action_async
from mcp_cli.chat.commands import register_command

async def cmd_provider(parts, ctx):
    await provider_action_async(parts[1:], context=ctx)
    return True

register_command("/provider", cmd_provider)
register_command("/p", cmd_provider)
