# src/mcp_cli/commands/definitions/conversation.py
"""
Unified conversation command implementation (chat mode only).
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)


class ConversationCommand(UnifiedCommand):
    """Manage conversation history."""
    
    @property
    def name(self) -> str:
        return "conversation"
    
    @property
    def aliases(self) -> List[str]:
        return ["history", "ch"]
    
    @property
    def description(self) -> str:
        return "Manage conversation history"
    
    @property
    def help_text(self) -> str:
        return """
Manage conversation history in chat mode.

Usage:
  /conversation [action]
  
Actions:
  show    - Show conversation history
  clear   - Clear conversation history
  save    - Save conversation to file
  load    - Load conversation from file

Examples:
  /conversation show      - Display conversation
  /conversation clear     - Clear history
  /conversation save chat.json - Save to file
  /conversation load chat.json - Load from file
"""
    
    @property
    def modes(self) -> CommandMode:
        """This is a chat-only command."""
        return CommandMode.CHAT
    
    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="action",
                type=str,
                required=False,
                help="Action to perform",
                choices=["show", "clear", "save", "load"],
            ),
            CommandParameter(
                name="filename",
                type=str,
                required=False,
                help="Filename for save/load operations",
            ),
        ]
    
    async def execute(self, **kwargs) -> CommandResult:
        """Execute the conversation command."""
        # Get chat context
        chat_context = kwargs.get("chat_context")
        if not chat_context:
            return CommandResult(
                success=False,
                error="Conversation command requires chat context.",
            )
        
        # Get action
        action = kwargs.get("action", "show")
        if "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list) and args_val:
                action = args_val[0]
            elif isinstance(args_val, str):
                action = args_val
        
        if action == "show":
            # Show conversation history
            if hasattr(chat_context, "conversation_history"):
                history = chat_context.conversation_history
                if not history:
                    return CommandResult(
                        success=True,
                        output="No conversation history.",
                    )
                
                output_lines = ["Conversation History:", "=" * 40]
                for i, msg in enumerate(history):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    # Truncate long messages
                    if len(content) > 200:
                        content = content[:197] + "..."
                    output_lines.append(f"\n[{i+1}] {role.upper()}:")
                    output_lines.append(content)
                
                return CommandResult(
                    success=True,
                    output="\n".join(output_lines),
                )
            else:
                return CommandResult(
                    success=False,
                    error="Conversation history not available.",
                )
        
        elif action == "clear":
            # Clear conversation history
            if hasattr(chat_context, "clear_conversation"):
                chat_context.clear_conversation()
                return CommandResult(
                    success=True,
                    output="Conversation history cleared.",
                )
            else:
                return CommandResult(
                    success=False,
                    error="Cannot clear conversation history.",
                )
        
        elif action == "save":
            # Save conversation
            filename = kwargs.get("filename")
            if not filename and "args" in kwargs:
                args_val = kwargs["args"]
                if isinstance(args_val, list) and len(args_val) > 1:
                    filename = args_val[1]
            
            if not filename:
                return CommandResult(
                    success=False,
                    error="Filename required for save. Usage: /conversation save <filename>",
                )
            
            try:
                import json
                if hasattr(chat_context, "conversation_history"):
                    with open(filename, "w") as f:
                        json.dump(chat_context.conversation_history, f, indent=2)
                    return CommandResult(
                        success=True,
                        output=f"Conversation saved to {filename}",
                    )
                else:
                    return CommandResult(
                        success=False,
                        error="Conversation history not available.",
                    )
            except Exception as e:
                return CommandResult(
                    success=False,
                    error=f"Failed to save conversation: {str(e)}",
                )
        
        elif action == "load":
            # Load conversation
            filename = kwargs.get("filename")
            if not filename and "args" in kwargs:
                args_val = kwargs["args"]
                if isinstance(args_val, list) and len(args_val) > 1:
                    filename = args_val[1]
            
            if not filename:
                return CommandResult(
                    success=False,
                    error="Filename required for load. Usage: /conversation load <filename>",
                )
            
            try:
                import json
                with open(filename, "r") as f:
                    history = json.load(f)
                
                if hasattr(chat_context, "set_conversation_history"):
                    chat_context.set_conversation_history(history)
                    return CommandResult(
                        success=True,
                        output=f"Conversation loaded from {filename} ({len(history)} messages)",
                    )
                else:
                    return CommandResult(
                        success=False,
                        error="Cannot set conversation history.",
                    )
            except Exception as e:
                return CommandResult(
                    success=False,
                    error=f"Failed to load conversation: {str(e)}",
                )
        
        else:
            return CommandResult(
                success=False,
                error=f"Unknown action: {action}. Use show, clear, save, or load.",
            )