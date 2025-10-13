# MCP-CLI Command System Demo

This example demonstrates the MCP-CLI command system working independently of the main application.

## Purpose

The `command_system_demo.py` script shows:
- All chat commands working with the global context manager
- No need to pass `ctx` parameter to command functions
- Commands using `get_context()` to access application state
- The command system functioning independently

## Key Changes Made

### 1. Command Signatures
All command functions now have simplified signatures:
```python
# Before
async def command_name(parts: List[str], ctx: Dict[str, Any] = None) -> bool:
    # Use passed context
    
# After  
async def command_name(parts: List[str]) -> bool:
    context = get_context()  # Get global context
```

### 2. Global Context Manager
Commands now use the centralized `ApplicationContext`:
```python
from mcp_cli.context import get_context

async def my_command(parts: List[str]) -> bool:
    context = get_context()
    # Access context attributes directly
    provider = context.provider
    model = context.model
    tool_manager = context.tool_manager
```

### 3. Context Methods
`ApplicationContext` now includes:
- `update(**kwargs)` - Update context with keyword arguments
- `update_from_dict(data)` - Update from dictionary (backward compatibility)
- `to_dict()` - Convert to dictionary (backward compatibility)

## Running the Demo

```bash
python examples/command_system_demo.py
```

## Output

The demo will:
1. Initialize a mock context with tool manager and model manager
2. List all registered commands (37 total)
3. Execute each command category:
   - Help commands (`/help`, `/qh`)
   - Server commands (`/servers`)
   - Tool commands (`/tools`, `/tools-enable`, etc.)
   - Provider/Model commands (`/provider`, `/model`)
   - Conversation commands (`/conversation`, `/toolhistory`)
   - UI commands (`/theme`, `/verbose`, `/confirm`)
   - Utility commands (`/ping`, `/resources`, `/prompts`)
   - Session commands (`/cls`, `/clear`, `/save`)
   - Control commands (`/interrupt`, `/stop`, `/cancel`)
   - Exit command (`/exit`)

## Command Registration

Commands are auto-registered when their modules are imported:
```python
# In command module
from mcp_cli.chat.commands import register_command

async def my_command(parts: List[str]) -> bool:
    # Command implementation
    return True

register_command("/mycommand", my_command)
```

## Benefits

1. **Cleaner Code**: No need to pass context dictionaries around
2. **Type Safety**: ApplicationContext provides typed attributes
3. **Centralized State**: Single source of truth for application state
4. **Backward Compatibility**: Still supports dict-based operations where needed
5. **Independent Testing**: Commands can be tested without full app setup

## /exit Command Fix

The `/exit` command now works correctly. Previously, plain `exit` (without slash) was being intercepted before slash command processing. This has been fixed in the chat handler.