# MCP-CLI Roadmap

> **Vision:** Transform mcp-cli from an interactive MCP runtime into a programmable AI operations environment — the neutral runtime where agent capabilities live.

### The Stack

```
Model
  ↓
Skill    (what to do — portable behaviour)
  ↓
Plan     (how to do it — execution graph)
  ↓
Tools    (do it — MCP servers)
  ↓
World
```

Today most systems jump straight from model → tools. Skills become the **stable abstraction layer**. Plans make execution **repeatable**. MCP-CLI is the runtime that binds them.

### The Ecosystem Gap

| Layer               | Status                                |
|---------------------|---------------------------------------|
| Models              | Interchangeable                       |
| Tools (MCP servers) | Standardized                          |
| Agents              | Ad-hoc                                |
| **Skills**          | **Fragmented (Claude/Codex proprietary)** |

The missing piece: a **portable capability layer between prompts and tools**. If mcp-cli becomes the neutral runtime for that layer, it becomes the distribution platform for agent behaviour — Docker for AI capabilities.

---

## Tier 1: Foundation (Fix What Breaks Today) ✅ COMPLETE

### 1.1 Truncate Large Tool Results

**Problem:** Tool results stored in full in conversation history with no size limit. A single maritime API result can be 100K+ chars, causing 1.87M token overflow.

**Files:** `src/mcp_cli/config/defaults.py`, `src/mcp_cli/chat/tool_processor.py`

- Add `DEFAULT_MAX_TOOL_RESULT_CHARS = 100_000` (~25K tokens)
- Add `_truncate_tool_result(content, max_chars)` to `ToolProcessor`
  - Head + truncation notice + tail (preserves value binding `**RESULT: $vN = ...**`)
  - `max_chars=0` disables
- Call in `_add_tool_result_to_history()` before creating the Message
- Safe: value binding extraction uses raw result object, not formatted string

### 1.2 Strip Old Reasoning Content

**Problem:** Thinking models (kimi-k2.5, DeepSeek) produce 100K+ chars of reasoning per turn, all sent back on every subsequent API call.

**File:** `src/mcp_cli/chat/conversation.py`

- Add `_prepare_messages_for_api()` — replaces inline message serialization
- Add `_strip_old_reasoning_content(messages)` — keep only most recent reasoning
- Update 3 call sites in `_handle_streaming_completion` and `_handle_regular_completion`

### 1.3 Conversation History Sliding Window

**Problem:** `conversation_history` reads ALL events without limit. No eviction or compression.

**File:** `src/mcp_cli/chat/chat_context.py`

- Add `max_history_messages` parameter (default ~200)
- Keep system prompt + last N messages
- Warn when approaching threshold
- Summarize evicted messages into compact "earlier context" block

### 1.4 Enable Infinite Context Mode

**Problem:** SessionManager has `infinite_context`, context packing, and auto-summarization — all disabled. ChatContext hardcodes `infinite_context=False`.

**Files:** `src/mcp_cli/chat/chat_context.py`, `src/mcp_cli/chat/chat_handler.py`

- Make `infinite_context` configurable (CLI flag or per-provider default)
- Configure `token_threshold` and `max_turns_per_segment` per model
- Leverage SessionManager's built-in context packing

### 1.5 Streaming Buffer Caps

**Problem:** `StreamingState.accumulated_content` and `reasoning_content` are unbounded with O(n^2) concat.

**File:** `src/mcp_cli/display/models.py`

- Add `max_accumulated_size` cap (1MB) to StreamingState
- Chunk count limit for stalled stream detection
- Use list-join pattern instead of string concatenation

---

## Tier 2: Efficiency & Resilience ✅ COMPLETE

### 2.1 Eliminate Triple Tool Result Storage

**Problem:** Tool results stored 3x: SessionManager events, `tool_history` list, procedural memory.

**Files:** `src/mcp_cli/chat/tool_processor.py`, `src/mcp_cli/chat/chat_context.py`

- Remove `self.context.tool_history: list[ToolExecutionRecord]` (unbounded in-memory list)
- Read from `tool_memory.memory.tool_log[-limit:]` on demand
- Single source of truth: SessionManager for flow, procedural memory for learning

### 2.2 Enforce Procedural Memory Limits

**Problem:** `max_patterns_per_tool` exists but is never enforced. Patterns grow unbounded.

- Enforce with LRU eviction
- Store only first 100 chars of result, not full payloads
- Add per-session memory cap

### 2.3 System Prompt Optimization

**Problem:** With 50+ tools, system prompt includes all tool names. Rebuilds on every model change.

**Files:** `src/mcp_cli/chat/system_prompt.py`, `src/mcp_cli/chat/chat_context.py`

- For 20+ tool servers, show categories + "... and X more"
- Cache system prompt with dirty flag (only regenerate when tools change)

### 2.4 Stale Connection Recovery

**Problem:** No health checks for long-lived MCP connections. Server dies = permanent failure.

**File:** `src/mcp_cli/tools/manager.py`

- Health check on tool execution failure
- Automatic reconnection with backoff
- `--reconnect-on-failure` CLI flag

### 2.5 Tool Batch Timeout

**Problem:** If one tool hangs in a parallel batch, everything blocks. No global timeout envelope.

**File:** `src/mcp_cli/tools/execution.py`

- Wrap task gathering in `asyncio.wait_for` with configurable timeout
- Cancel remaining tasks on timeout

### 2.6 Narrower Exception Handlers

**Problem:** Broad `except Exception` masks critical issues. Inconsistent `CancelledError` handling.

**Files:** `src/mcp_cli/chat/conversation.py`, `src/mcp_cli/chat/streaming_handler.py`

- Catch specific exceptions (APIError, TimeoutError, ValueError)
- Consistent `asyncio.CancelledError` handling across all async loops

### 2.7 Provider Validation

**Problem:** Invalid API keys accepted silently. Chat fails only after user starts typing.

**Files:** `src/mcp_cli/model_management/model_manager.py`, `src/mcp_cli/chat/chat_handler.py`

- Quick auth validation in `ChatContext.initialize()`
- Optional connection test on `add_runtime_provider()`

### 2.8 LLM-Visible Context Management Notices

**Problem:** Tier 1 truncation, sliding window eviction, and reasoning stripping happen silently. The LLM sees truncated results but doesn't know data was removed, so it can't adjust its strategy (e.g., request smaller date ranges, fewer fields, or paginated results).

**Files:** `src/mcp_cli/chat/tool_processor.py`, `src/mcp_cli/chat/conversation.py`, `src/mcp_cli/chat/chat_context.py`

- **Tool result truncation notice:** When `_truncate_tool_result` fires, inject a system-level hint into the conversation: `"The previous tool result was truncated from {N} to {M} chars. Consider requesting less data (smaller date range, fewer fields, pagination)."`
- **Sliding window eviction notice:** When messages are evicted, inject: `"Context window: {N} older messages were evicted. Key context may need to be re-established."`
- **Reasoning stripping notice:** When old reasoning is stripped, inject a compact note so the model knows it lost its earlier chain of thought
- **Context compaction notice:** When SessionManager compacts/summarizes, surface the summary to the model so it knows what was compressed
- Design as injectable system messages placed just before the next API call, not permanently stored in history
- Configurable: `--context-notices / --no-context-notices` flag (default on)

---

## MCP Apps (SEP-1865) ✅ COMPLETE

Interactive HTML UIs served by MCP servers, rendered in sandboxed browser iframes.

### Implementation

- **Host/Bridge/HostPage architecture:** Local websockets server per app, iframe sandbox with postMessage ↔ WebSocket bridge
- **Tool meta propagation:** `_meta.ui` on tool definitions triggers automatic app launch on tool call
- **structuredContent recovery:** Extracts `structuredContent` from JSON text blocks when CTP transport discards it
- **Security hardening:** XSS prevention (html.escape), CSP domain sanitization, tool name validation, URL scheme validation
- **Session reliability:** Message queue with drain-on-reconnect, exponential backoff, state reset, duplicate prevention, push-to-existing-app
- **Spec compliance:** `ui/initialize`, `ui/resource-teardown`, `ui/notifications/host-context-changed`, sandbox capabilities
- **Robustness:** Tool execution timeout, safe JSON serialization, circular reference protection, initialization timeout
- **Test coverage:** 96 tests across bridge, host, security, session, models, and meta propagation

### Known Limitations

- Map and video views depend on server-side app JavaScript (not an mcp-cli issue)
- `ui/notifications/tool-input-partial` (streaming argument assembly) deferred to future work
- HTTPS/TLS for remote deployment not yet implemented
- CTP transport `_normalize_mcp_response` discards `structuredContent` — recovered via text block extraction

---

## Tier 3: Performance & Polish ✅ COMPLETE

### 3.1 Tool Lookup Index

**Problem:** `get_tool_by_name()` is O(n) per call — iterates all tools.

**File:** `src/mcp_cli/tools/manager.py`

- `_tool_index: dict[str, ToolInfo]` with lazy build on first access → O(1) lookups
- Dual-key indexing: fully qualified name + simple name
- Invalidated via `_invalidate_caches()` when tools change

### 3.2 Cache Tool Metadata Per Provider

**Problem:** `get_tools_for_llm()` re-filters and re-validates every call.

**File:** `src/mcp_cli/tools/manager.py`

- `_llm_tools_cache: dict[str, list[dict]]` keyed by provider name
- Returns cached tools on hit, rebuilds on miss
- Invalidated alongside tool index on tool state changes
- Bypassed when `MCP_CLI_DYNAMIC_TOOLS=1`

### 3.3 Startup Progress

**Problem:** No feedback during server init. Blank screen if >2s.

**Files:** `src/mcp_cli/tools/manager.py`, `src/mcp_cli/chat/chat_handler.py`, `src/mcp_cli/chat/chat_context.py`

- `on_progress` callback passed through initialization chain
- Reports: "Loading server configuration...", "Connecting to N server(s)...", "Discovering tools...", "Adapting N tools for {provider}..."
- Chat handler wires callback to `output.info()` for real-time display

### 3.4 Token & Cost Tracking

**Problem:** Zero visibility into token usage or API costs.

**Files:** `src/mcp_cli/chat/token_tracker.py`, `src/mcp_cli/commands/usage/usage.py`

- `TokenTracker` with `TurnUsage` Pydantic models
- Per-turn input/output tracking with chars/4 estimation fallback
- `/usage` command (aliases: `/tokens`, `/cost`) for cumulative display
- Integrated with conversation export

### 3.5 Session Persistence & Resume

**Problem:** Conversations lost on exit. No way to resume.

**Files:** `src/mcp_cli/chat/session_store.py`, `src/mcp_cli/commands/sessions/sessions.py`

- File-based persistence at `~/.mcp-cli/sessions/`
- `/sessions list`, `/sessions save`, `/sessions load <id>`, `/sessions delete <id>`
- Auto-save every 10 turns via `auto_save_check()` in ChatContext

### 3.6 Conversation Export

**Files:** `src/mcp_cli/commands/export/export.py`, `src/mcp_cli/chat/exporters.py`

- `/export markdown [filename]` and `/export json [filename]`
- Includes tool calls with arguments, tool results (truncated), token usage metadata
- Markdown: formatted sections by role; JSON: structured with version and timestamps

---

## Tier 4: Code Quality

### 4.1 Replace Global Singletons

`_GLOBAL_TOOL_MANAGER` and preference manager use global state → dependency injection.

### 4.2 Consolidate Message Classes

Two `Message` classes with different `tool_calls` types → type guards at boundaries, consider unification.

### 4.3 Standardize Logging

Mixed `logging.error()` / `output.error()` → library code uses `logging`, CLI code uses `output`.

### 4.4 Integration Tests

All ~3,164 tests are unit mocks → add end-to-end tests with a test MCP server.

### 4.5 Coverage Reporting

No CI coverage → `uv run pytest --cov=src/mcp_cli --cov-fail-under=80`

---

## Tier 5: Production Hardening

### 5.1 Structured File Logging

Logs only go to stderr → `RotatingFileHandler` at `~/.mcp-cli/logs/`, JSON format, secret redaction.

### 5.2 Server Health Monitoring

On-demand only → periodic polling, rate limit detection, automatic recovery.

### 5.3 Per-Server Configuration

Shared timeouts → per-server overrides, enable/disable without removal, documented token syntax.

### 5.4 Thread-Safe OAuth

`manager.py:548` mutates headers without locking → lock or thread-safe update.

---

## Tier 6: Execution Graphs & Plans

> **Shift:** conversation → reasoning → tools **becomes** intent → plan → execution → memory → replay

### 6.1 First-Class Plans

Today the AI reasons from scratch each time. Plans make workflows reproducible, shareable, schedulable, and testable — Terraform for agents.

```
mcp plan create "plan a coastal walk tomorrow"
mcp plan inspect 42
mcp plan run 42
mcp plan replay 42 --dry-run
```

- Plan = persistent, inspectable execution graph (DAG of tool calls + decisions)
- Plans are serialized (YAML/JSON) and version-controlled
- `replay --dry-run` shows what would execute without side effects
- Plans can be parameterized: `mcp plan run 42 --date 2026-03-01`

### 6.2 Simulation / Dry-Run Mode

Critical for trust. Show planned tool calls without executing them.

```
mcp run "delete inactive users" --simulate
```

- Traces the full execution path
- Shows tool calls that *would* happen, with estimated arguments
- No side effects — safe to run in production
- Foundation for plan creation: `--simulate` output becomes a plan

### 6.3 Deterministic Mode

Enterprise reliability — bounded, predictable execution.

```
mcp run "book cheapest train" --deterministic
```

- Fixed tool selection (no exploration)
- Bounded reasoning (max turns, max tokens)
- Structured outputs with schema validation
- Configurable retry policies
- Reproducible given same inputs

---

## Tier 7: Observability & Traces

> **Goal:** Turn mcp-cli into a debugger for AI behavior. No other agent CLI has a proper observability layer yet.

### 7.1 Tool Reasoning Traces

Operators need to understand *why* the AI chose specific tools and how data flowed.

```
mcp trace last
mcp trace step 4
mcp trace graph
```

Graph output:
```
User Intent
   |
Find location
   |
Weather API ---+
               +--- Decision -> unsafe
Tide API ------+
```

- Every tool call gets a trace ID linking intent → reasoning → tool → result → decision
- Traces are persistent (stored alongside session)
- Exportable for audit: `mcp trace export --format json`

### 7.2 Structured Outputs as First-Class

Allow output schemas for automation pipelines.

```
mcp run "analyse stocks" --schema portfolio.json
```

- Validate LLM output against JSON Schema
- Retry with schema hint on validation failure
- Composable with plans: plan steps can have typed inputs/outputs

---

## Tier 8: Memory Scopes

> **Shift:** Agents stop re-discovering facts every run. Unlocks long-running assistants.

### 8.1 Scoped Memory System

| Scope       | Purpose                  | Lifetime          |
|-------------|--------------------------|-------------------|
| `session`   | Current conversation     | Until exit        |
| `workspace` | Ongoing project context  | Until cleared     |
| `global`    | Personal knowledge base  | Persistent        |
| `plan`      | Workflow-specific memory | Tied to plan      |
| `skill`     | Skill-scoped knowledge   | Tied to skill     |

```
mcp memory show workspace
mcp memory edit global
mcp memory diff
```

- Memory injected into system prompt based on scope
- Workspace memory scoped to current directory / project
- Global memory shared across all sessions
- Plan and skill memory are portable — travel with the artifact

### 8.2 Memory-Aware Context Building

- When building messages for API, inject relevant memory by scope
- Workspace memory overrides global on conflict
- Plan/skill memory available during execution
- Memory compaction: summarize old entries, keep recent ones verbatim

---

## Tier 9: Skills & Capability Layer

> **The strategic keystone.** Skills make behaviour reusable the way MCP made tools reusable. This is the missing standard in the agent stack — the portable capability layer between prompts and tools.

### The Problem

Today's agent ecosystem:

| Ecosystem   | Problem                           |
|-------------|-----------------------------------|
| Claude Code | Locked skills (proprietary)       |
| OpenAI/Codex| Tool calls but no portability     |
| LangChain   | Code-level abstraction only       |
| Marketplaces| Shareable socially, not at runtime|

GitHub repos can be shared between ecosystems, but can't be *executed* across runtimes without rewriting glue. That's manual porting, not interoperability. The incompatibility isn't in prompts — it's in the **hidden runtime contract**: tool naming, approval models, execution semantics, memory models, error handling.

### The Solution: Declarative Capability Binding

Skills don't assume specific tools exist. They declare **capability intent**:

```yaml
capabilities:
  - web_search
  - file_edit
  - structured_reasoning
```

The runtime resolves capabilities to concrete tools:

```
web_search         → tavily server (or serpapi, or browser.search)
file_edit          → fs.mutate (or code.patch)
structured_reasoning → reasoning model
```

Same skill runs everywhere. Different infrastructure, same behaviour.

### Plugins vs Skills

|                      | Plugin (MCP Server) | Skill              |
|----------------------|---------------------|--------------------|
| Adds capability      | Yes                 | No                 |
| Defines behaviour    | No                  | Yes                |
| Implements functions | Yes                 | No                 |
| Reusable workflow    | No                  | Yes                |

MCP server = plugin. Skill = behaviour. mcp-cli hosts both.

### 9.1 Skill Format

A skill is a directory:

```
skills/
  travel-plan/
    skill.yaml         # manifest: capabilities, policy, metadata
    instructions.md     # reasoning instructions (natural language)
    schema.json         # structured output contract (optional)
```

Minimal `skill.yaml`:

```yaml
name: travel-plan
version: 1.0.0
description: Plan a day trip using weather and transit data

inputs:
  - name: query
    type: string
    description: What to plan

capabilities:
  required:
    - weather_forecast
    - route_planning
  optional:
    - tide_data

output:
  schema: schema.json

policy:
  allow_web: false
  allow_code_exec: false
  deterministic: true
  max_tool_calls: 20
```

The spec is intentionally tiny — small enough people actually adopt it.

### 9.2 Skill Runtime

```
mcp skill run travel-plan "day trip to Raglan tomorrow"
mcp skill inspect travel-plan
mcp skill validate travel-plan
```

**Runtime resolution flow:**

1. Load `skill.yaml` → parse capabilities
2. Resolve capabilities to available MCP tools:
   ```
   Skill needs: weather_forecast
   You have: 3 weather servers
   → mcp-cli asks: Use local weather, NOAA, or MetOffice?
   ```
3. Apply policy constraints (tool allowlist, deterministic mode, etc.)
4. Inject `instructions.md` as system prompt context
5. Execute with plan/trace support (Tiers 6-7)
6. Validate output against `schema.json`

Skills are portable across infrastructures because they bind to capabilities, not tools.

### 9.3 Skill Discovery & Installation

```
mcp skill search "finance"
mcp skill install mortgage-analyzer
mcp skill list
mcp skill recommend "analyse my expenses"
```

Registry is just Git repos:

```
mcp://skills/travel-plan → github.com/user/travel-plan-skill
```

Downloaded to `~/.mcp-cli/skills/`. No new marketplace needed — Git is the registry, `skill.yaml` is the manifest.

### 9.4 Skill Packaging & Publishing

```
mcp skill pack travel-plan        # validate + bundle
mcp skill publish travel-plan     # push to registry
```

- Validates `skill.yaml` schema
- Checks capability names against known capability vocabulary
- Bundles instructions, schema, and metadata
- Publishes to configured registry (Git, npm-style, or local)

### 9.5 Cross-Ecosystem Adapters

Don't copy proprietary formats — translate them:

```
Claude skill    → MCP skill adapter
Codex agent     → MCP skill adapter
LangChain chain → MCP skill adapter
```

mcp-cli becomes the **interoperability layer**. The Linux of agent tooling.

### 9.6 Capability Vocabulary

A small, shared vocabulary of abstract capabilities that skills declare and runtimes resolve:

| Capability             | Example tools that satisfy it         |
|------------------------|---------------------------------------|
| `web_search`           | tavily, serpapi, browser.search       |
| `fetch_url`            | fetch, browser.navigate               |
| `file_read`            | fs.read, code.view                    |
| `file_edit`            | fs.write, code.patch                  |
| `weather_forecast`     | openweather, noaa, metoffice          |
| `structured_reasoning` | reasoning model, chain-of-thought     |
| `code_execution`       | python.exec, sandbox.run              |
| `database_query`       | sql.query, sqlite.exec               |

Vocabulary grows organically. Skills can declare custom capabilities; the runtime warns if unresolvable.

### 9.7 Skill + Plan Integration

Skills can contain optional workflows (plans):

```yaml
# skill.yaml
workflow: workflow.yaml   # optional execution graph
```

When present, the skill uses the plan engine (Tier 6) for structured execution instead of free-form conversation. This bridges "chat-style" skills and "deterministic pipeline" skills.

### 9.8 Skill Memory

Each skill gets its own memory scope (Tier 8):

- Persists across runs of the same skill
- Stores learned patterns (which tools worked, common errors)
- Portable: travels with the skill package
- Separate from session/workspace/global memory

---

## Tier 10: Scheduling & Background Agents

> **Shift:** mcp-cli becomes cron + AI + tools.

### 10.1 Agent Definitions

```
mcp agent create surf-check --prompt "check surf conditions at Raglan"
mcp agent create daily-report --skill daily-summary
mcp agent list
mcp agent logs surf-check
```

- Agent = named prompt OR skill + server config + schedule (optional)
- Stored in `~/.mcp-cli/agents/`
- Each agent has its own memory scope
- Agents can reference skills for portable, reusable behaviour

### 10.2 Scheduling

```
mcp agent schedule surf-check "6:00 daily"
mcp agent schedule report "0 9 * * MON"   # cron syntax
mcp agent unschedule surf-check
```

- Lightweight daemon or OS-native scheduling (launchd/systemd/cron)
- Results stored in agent log, accessible via `mcp agent logs`
- Notifications on failure or interesting results (webhook, email, or local)

### 10.3 Background Execution

```
mcp agent run surf-check --background
mcp agent status surf-check
mcp agent stop surf-check
```

- Detached execution with log tailing
- Graceful interruption

---

## Tier 11: Multi-Agent Coordination

> **Shift:** Single conversation loop becomes agent pipelines. First CLI to naturally express this.

### 11.1 Workflow Definitions

```
mcp workflow run travel.yaml
mcp workflow inspect travel.yaml
mcp workflow validate travel.yaml
```

Example `travel.yaml`:
```yaml
agents:
  planner:
    model: gpt-5
    skill: travel-plan
  researcher:
    model: local-llama
    servers: [search]
  verifier:
    model: reasoning-model
    skill: safety-check

flow:
  - researcher: "find options for {destination}"
  - planner: "create itinerary from research"
  - verifier: "validate safety and feasibility"
  - user: "review and approve"
```

- Agents pass structured outputs between steps (via skill schemas)
- Each agent has its own tool set, model, and optional skill
- Workflow memory shared across agents (plan-scoped)
- Conditional branching: `if verifier.unsafe then planner.revise`

### 11.2 Tool Capability Discovery & Ranking

Instead of listing tools, rank them by relevance.

```
mcp tools suggest "calculate mortgage"
```

Output:
```
finance.mortgage_calculator  (confidence 0.91)
spreadsheet.compute          (confidence 0.52)
python.exec                  (confidence 0.12)
```

- Semantic search across tool descriptions
- Boosted by procedural memory (tools that worked before rank higher)
- Used by skill runtime for automatic capability resolution

---

## Tier 12: Remote Sessions

> **Shift:** ssh for AI systems. Run agents remotely, control locally.

### 12.1 Remote Agent Control

```
mcp connect prod-agent.company.net
mcp remote status
mcp remote logs --follow
```

- Local CLI connects to remote mcp-cli instance
- Authentication via SSH keys or OAuth
- Stream reasoning and tool traces in real-time
- Execute commands against remote tool servers

### 12.2 Shared Sessions

- Multiple operators can observe the same agent session
- Read-only mode for auditors
- Collaborative mode for pair-debugging agents

---

## Priority Summary

| Tier | Focus | What Changes | Status |
|------|-------|-------------|--------|
| **1** | Memory & context | Stops crashing on large payloads | ✅ Complete |
| **2** | Efficiency & resilience | Reliable under real workloads | ✅ Complete |
| **Apps** | MCP Apps (SEP-1865) | Interactive browser UIs from MCP servers | ✅ Complete |
| **3** | Performance & polish | Feels fast, saves work | ✅ Complete |
| **4** | Code quality | Maintainable, testable | Medium |
| **5** | Production hardening | Observable, auditable | Medium |
| **6** | Plans & execution graphs | Reproducible workflows | High |
| **7** | Observability & traces | Debugger for AI behavior | High |
| **8** | Memory scopes | Long-running assistants | High |
| **9** | Skills & capabilities | Portable behaviour layer | High |
| **10** | Scheduling & agents | Autonomous operations | High |
| **11** | Multi-agent coordination | Agent pipelines | Very High |
| **12** | Remote sessions | Distributed AI ops | Very High |

### If You Build Only Four Things

These change the category of the tool from **chat interface** to **agent operating system**:

1. **Plans** (Tier 6) — reproducible, inspectable execution
2. **Traces** (Tier 7) — explainable AI operations
3. **Skills** (Tier 9) — portable, reusable behaviour (the npm for agents)
4. **Scheduling** (Tier 10) — autonomous background agents

### The Strategic Position

| Ecosystem | Problem                           | mcp-cli Opportunity                  |
|-----------|-----------------------------------|--------------------------------------|
| Claude    | Locked skills                     | Open skill format + adapter          |
| OpenAI    | Tool calls, no portability        | Capability binding layer             |
| LangChain | Code-level only                   | Declarative behaviour packages       |
| All       | Prompts not reusable, tools are   | **Skills make behaviour reusable**   |

### The Meta Insight

Most tools optimize *how the model thinks*. The winning layer optimizes *how thinking is executed, repeated, inspected, and trusted*. mcp-cli is already positioned exactly there.

Skills are the keystone: they turn mcp-cli from a runtime into a **distribution platform for agent behaviour** — the neutral ground where capabilities live, regardless of which model or infrastructure runs them.
