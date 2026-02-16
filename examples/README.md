# MCP-CLI Examples

Organized examples demonstrating mcp-cli capabilities. Most examples run self-contained; those requiring API keys note it in their docstring.

## Directory Structure

```
examples/
  getting_started/       Basic examples to get up and running
  streaming/             Streaming response demos
  tools/                 Tool execution and round-trip demos
  commands/              Command system and slash commands
  servers/               Server management and custom providers
  safety/                Context safety mechanisms (Tier 1)
  sample_tools/          Reusable tool classes for demos
```

## Getting Started

```bash
# Run from the project root
cd /path/to/mcp-cli

# Basic LLM call (requires OPENAI_API_KEY)
python examples/getting_started/basic_llm_call.py

# Ollama local model (requires running Ollama)
python examples/getting_started/ollama_llm_call.py

# Simple streaming demo (requires OPENAI_API_KEY)
python examples/getting_started/streaming_simple.py

# UI banner demo (no API key needed)
python examples/getting_started/clear_with_banner_demo.py
```

## Streaming

```bash
# Full streaming showcase with multiple models
python examples/streaming/mcp_streaming_showcase.py

# Real LLM streaming with tool integration
python examples/streaming/real_llm_streaming_demo.py

# Working demo of streaming in mcp-cli context
python examples/streaming/mcp_cli_working_demo.py
```

## Tools

```bash
# LLM <-> local tool round-trip (requires OPENAI_API_KEY)
python examples/tools/tool_round_trip.py

# Tool execution in interactive mode
python examples/tools/demo_tool_execution.py

# Tool argument flow tracing (diagnostic)
python examples/tools/tool_args_flow_demo.py

# Runtime server add + tool execution
python examples/tools/server_add_and_execute_tool.py

# Playwright browser automation with tools
python examples/tools/playwright_server_with_tool_execution.py
```

## Commands

```bash
# Command system demo
python examples/commands/command_system_demo.py

# End-to-end command verification
python examples/commands/command_system_e2e_demo.py

# Slash command autocomplete
python examples/commands/slash_commands_demo.py

# Interactive mode capabilities
python examples/commands/demo_interactive_mode.py

# Command mode from shell
bash examples/commands/cmd_mode_demo.sh
```

## Servers

```bash
# Server lifecycle: list, add, use, disable, enable, remove
python examples/servers/server_management_e2e.py

# Custom provider integration
python examples/servers/custom_provider_working_demo.py
```

## Safety

### Tier 1: Context Safety

```bash
# All 5 context safety mechanisms — no API key needed
python examples/safety/tier1_context_safety_demo.py
```

Demonstrates:
1. Tool result truncation (100K char cap)
2. Reasoning content stripping
3. Conversation history sliding window
4. Infinite context configuration
5. Streaming buffer caps

### Tier 2: Efficiency & Resilience

```bash
# All 8 efficiency features — no API key needed
python examples/safety/tier2_efficiency_demo.py
```

Demonstrates:
1. Single source of truth for tool history (procedural memory)
2. Procedural memory pattern limits
3. System prompt optimization (caching + tool summary)
4. Connection error classification
5. Tool batch timeout
6. Narrower exception handlers
7. Provider validation at startup
8. LLM-visible context management notices
