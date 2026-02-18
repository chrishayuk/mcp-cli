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

## MCP Apps (SEP-1865)

MCP Apps are interactive HTML UIs served by MCP servers and rendered in the user's browser via sandboxed iframes. When a tool has a `_meta.ui` annotation, mcp-cli launches a local web server that bridges the browser and the MCP backend.

### Architecture

```
Browser                    Python Backend                MCP Server
┌─────────────────┐       ┌──────────────────┐       ┌──────────────┐
│  Host Page (JS)  │──WS──│  AppBridge        │──MCP──│  Tool Server │
│  ┌─────────────┐ │      │  (bridge.py)      │       │              │
│  │ App iframe  │ │      └──────────────────┘       └──────────────┘
│  │ (sandboxed) │ │              │
│  └─────────────┘ │      ┌──────────────────┐
│   postMessage ↕  │      │  AppHostServer   │
└─────────────────┘       │  (host.py)        │
                          └──────────────────┘
```

- **`host.py`** — `AppHostServer` manages lifecycle: port allocation, HTTP serving (host page + app HTML), WebSocket server, browser launch
- **`host_page.py`** — JavaScript host page template; bridges iframe postMessage ↔ WebSocket, handles `ui/initialize`, display modes, reconnection
- **`bridge.py`** — `AppBridge` handles JSON-RPC protocol: proxies `tools/call` and `resources/read` to MCP servers, manages message queue for disconnected WS, formats tool results per MCP spec
- **`models.py`** — Pydantic models: `AppInfo`, `AppState` (PENDING → INITIALIZING → READY → CLOSED), `HostContext`

### Security Model

- **Iframe sandbox:** `allow-scripts allow-forms allow-same-origin allow-popups allow-popups-to-escape-sandbox`
- **XSS prevention:** Tool names are `html.escape()`d before template injection
- **CSP domain sanitization:** Server-supplied domains validated against `^[a-zA-Z0-9\-.:/*]+$`
- **Tool name validation:** Bridge rejects tool names not matching `^[a-zA-Z0-9_\-./]+$`
- **URL scheme validation:** `ui/open-link` only allows `http://` and `https://` schemes
- **Safe JSON serialization:** `_safe_json_dumps()` with `_to_serializable()` fallback; circular reference protection

### Session Reliability

- **Message queue:** `_pending_notifications: deque[str]` (maxlen=50) queues notifications when WS is disconnected
- **Drain on reconnect:** `drain_pending()` flushes queued messages when WS reconnects
- **State reset:** `set_ws()` resets state to INITIALIZING, closes old WS
- **Reconnect notification:** Host page sends `ui/notifications/reconnected` to app iframe on WS reconnect
- **Exponential backoff:** WS reconnection uses 1s→30s exponential backoff with reset on success
- **Initialization timeout:** Configurable JS timeout (default 30s) shows "initialization timed out" if app never initializes
- **Deferred tool result delivery:** Initial tool results are stored on the bridge and pushed only after the app sends `ui/notifications/initialized`, preventing race conditions where postMessage is dropped before the app sets up its listener
- **Duplicate prevention:** `launch_app()` closes previous instance before launching new one
- **Push to existing:** `tool_processor.py` pushes new tool results to running apps instead of re-launching

### Spec Compliance

- `ui/initialize` response includes protocol version, host capabilities (with sandbox details), host info, host context
- `ui/resource-teardown` sent to iframe on `beforeunload`
- `ui/notifications/host-context-changed` sent after display mode changes
- `structuredContent` recovered from JSON text blocks when transport loses it (CTP normalization)

---

## Two Message Classes

The codebase has two classes that represent messages, serving different purposes:

- **`chuk_llm.core.models.Message`** (re-exported via `chat/response_models.py`) — canonical LLM message with typed `ToolCall` objects. Used by `tool_processor.py` and `conversation.py`.
- **`mcp_cli.chat.models.HistoryMessage`** (aliased as `Message` for backward compat) — SessionManager-compatible message with `tool_calls: list[dict]`. Used by `chat_context.py`.

The roundtrip: chuk_llm Message → `to_dict()` → SessionEvent → `from_dict()` → HistoryMessage → `to_dict()` → API.

## Secret Redaction

`SecretRedactingFilter` in `config/logging.py` is always active on all log handlers (console and file). It redacts:

- Bearer tokens (`Authorization: Bearer eyJ...`)
- API keys (`sk-proj-...`, `sk-...`)
- Generic `api_key=...` / `api-key: ...` values
- OAuth access tokens in JSON (`"access_token": "..."`)
- Authorization headers (`Authorization: Basic ...`)

The filter is a module-level singleton (`secret_filter`) that can be added to custom handlers.

---

## Known Violations (Remaining)

Architecture review performed after Tier 2. Tier 4 (Code Quality) resolved the most impactful issues. Remaining items are tracked here.

### Core/UI Separation (#5)

**Resolved in Tier 4.3:** `chat/conversation.py`, `chat/tool_processor.py`, and `chat/chat_context.py` no longer import `chuk_term.ui.output`. All core logging goes through the `logging` module.

**Remaining:**

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
| `chat/models.py` | `HistoryMessage.tool_calls: list[dict]` instead of `list[ToolCallData]` | MEDIUM — by design for SessionManager compat |
| `chat/conversation.py` | `_validate_tool_messages()` works on raw dicts | MEDIUM — by design at serialization boundary |

### Explicit Dependencies (#7)

**Resolved in Tier 4.1:** `_GLOBAL_TOOL_MANAGER` singleton removed. ToolManager is constructor-injected everywhere.

**Remaining (deferred — low impact):**

| File | Issue | Severity |
|------|-------|----------|
| `chat/tool_processor.py` | Uses `get_tool_state()`, `get_search_engine()` globals | MEDIUM — external library singletons |
| `chat/conversation.py` | Uses `get_tool_state()` global | MEDIUM — external library singleton |
| `chat/tool_processor.py` | Uses `get_preference_manager()` global | LOW — 15 call sites, marginal payoff |
