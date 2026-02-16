# MCP-CLI Architecture Principles

These principles govern all new code in mcp-cli. Existing code should be migrated toward these standards as it is touched.

## 1. Pydantic Native

Data structures are `BaseModel` subclasses, not raw dicts. Use `Field()` for defaults and documentation. Use `model_dump()` only at serialization boundaries (API calls, storage).

```python
# Yes
class ToolResult(BaseModel):
    name: str
    content: str
    success: bool = True

# No
result = {"name": "foo", "content": "bar", "success": True}
```

## 2. Async Native

Public APIs are `async def`. Synchronous code is only for pure computation with no I/O. Never block the event loop with synchronous I/O inside async functions.

```python
# Yes
async def execute_tool(self, name: str, args: dict) -> ToolResult: ...

# No
def execute_tool(self, name: str, args: dict) -> ToolResult:
    return asyncio.run(self._execute(name, args))
```

## 3. No Dictionary Goop

Typed models at every boundary. When receiving external data (API responses, config files), parse into models immediately. Pass models between functions, not dicts.

```python
# Yes
msg = Message.from_dict(raw_data)
process(msg)

# No
process(raw_data)  # passing a dict through the stack
```

## 4. No Magic Strings

Use `Enum`, `StrEnum`, or named constants from `config/defaults.py`. Never hardcode string literals that represent categories, field names, or configuration values.

```python
# Yes
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

# No
if msg["role"] == "assistant": ...
```

## 5. Core/UI Separation

Logic that is UI-independent must not import from `display/`, `interactive/`, or `commands/`. Core modules handle: tool execution, conversation management, session state, configuration.

UI modules handle: rendering, user input, theming, display formatting.

**Future goal:** core modules extractable into a standalone `mcp-cli-core` package.

```
src/mcp_cli/
    chat/               # Core: conversation, tool processing, context
    config/             # Core: defaults, configuration loading
    tools/              # Core: tool management, execution
    model_management/   # Core: provider/model resolution
    display/            # UI: rendering, formatting, streaming display
    interactive/        # UI: terminal interaction
    commands/           # UI: CLI command handlers
```

## 6. Single Source of Truth

All default values live in `config/defaults.py`. Business logic imports constants from there, never hardcodes values. Configuration flows: `defaults.py` -> CLI flags -> runtime config -> component init.

```python
# Yes
from mcp_cli.config.defaults import DEFAULT_MAX_TOOL_RESULT_CHARS
content = truncate(content, DEFAULT_MAX_TOOL_RESULT_CHARS)

# No
content = truncate(content, 100_000)  # magic number
```

## 7. Explicit Dependencies

Constructor injection over global singletons. When a component needs a dependency, accept it as a parameter. Global state is a last resort, not a first choice.

## 8. Fail Loudly at Boundaries, Recover Gracefully Inside

Validate inputs at system boundaries (CLI args, API responses, config files). Inside the core, trust the type system. Log errors with context, don't silently swallow exceptions with bare `except Exception`.

## 9. Linting and Type Checking

All code must pass `make check` (linting + type checking). No exceptions. Fix issues before merging, not after.

## 10. Test Coverage

Minimum **90% coverage per file**. New code ships with tests. Use `uv run pytest --cov=src/mcp_cli --cov-fail-under=90` to verify. Tests live alongside the code they test: `tests/chat/`, `tests/display/`, etc.

## 11. Working Examples

Every user-facing feature must have a working example in the `examples/` directory that demonstrates the functionality end-to-end. Examples serve as both documentation and integration tests.

---

## Known Violations (Tier 4 Backlog)

Architecture review performed after Tier 2. These are tracked for remediation in Tier 4 (Code Quality).

### Core/UI Separation (#5)

| File | Issue | Severity |
|------|-------|----------|
| `chat/ui_manager.py` | Imports `prompt_toolkit`, `display/`, `commands/` | HIGH — move to `interactive/` |
| `chat/command_completer.py` | Imports `prompt_toolkit`, `commands/` | HIGH — move to `interactive/` |
| `chat/streaming_handler.py:20` | Imports `StreamingDisplayManager` from `display/` | MEDIUM — use protocol |
| `chat/__main__.py:15` | Imports `register_commands` from `commands/` | MEDIUM — entry point, acceptable |

### Pydantic Native (#1, #3)

| File | Issue | Severity |
|------|-------|----------|
| `chat/chat_context.py` | `openai_tools: list[dict]` instead of typed model | MEDIUM |
| `chat/models.py` | `Message.tool_calls: list[dict]` instead of `list[ToolCallData]` | MEDIUM |
| `chat/conversation.py` | `_validate_tool_messages()` works on raw dicts | MEDIUM — by design at serialization boundary |

### Explicit Dependencies (#7)

| File | Issue | Severity |
|------|-------|----------|
| `chat/tool_processor.py` | Uses `get_tool_state()`, `get_search_engine()` globals | MEDIUM |
| `chat/conversation.py` | Uses `get_tool_state()` global | MEDIUM |
| `chat/tool_processor.py` | Uses `get_preference_manager()` global | LOW |

These are deferred to **Tier 4.1 (Replace Global Singletons)** and **Tier 4.2 (Consolidate Message Classes)**.
