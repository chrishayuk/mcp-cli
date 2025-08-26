# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development & Testing
```bash
# Install for development
make dev-install

# Run tests
make test                 # Run all tests
make test-cov            # Run tests with coverage report

# Code quality
make lint                # Run ruff linting
make format              # Auto-format code with ruff
make typecheck           # Run mypy type checking
make check               # Run all checks (lint + typecheck + test)

# Build & publish
make build               # Build package
make publish             # Publish to PyPI
make publish-test        # Publish to test PyPI

# Cleanup
make clean               # Basic cleanup
make clean-all           # Deep clean including caches
```

### Running the Application
```bash
# Default chat mode (uses Ollama with gpt-oss reasoning model)
mcp-cli --server sqlite

# Interactive mode
mcp-cli interactive --server sqlite

# Command mode for automation
mcp-cli cmd --server sqlite --tool list_tables --raw

# Direct commands
mcp-cli tools --server sqlite
mcp-cli provider list
mcp-cli models
```

## High-Level Architecture

### Core Component Interaction Flow

1. **Entry Point Flow** (`src/mcp_cli/main.py`):
   - Typer CLI parses arguments → Sets up logging/UI theme
   - Resolves provider/model through `ModelManager`
   - Routes to appropriate mode (chat/interactive/command/direct)

2. **Chat Mode Architecture**:
   - `ChatContext` manages state → `ChatUIManager` handles rich terminal UI
   - `ConversationProcessor` orchestrates conversation flow
   - `ModelManager` interfaces with chuk-llm for LLM providers
   - `ToolManager` handles MCP server communication via chuk-tool-processor

3. **Command System**:
   - Dual registration: `CommandRegistry` for most commands + direct registration for core commands
   - Commands follow consistent interface pattern with global options support
   - Async commands wrapped with `run_command_sync()` for synchronous execution

4. **Tool Execution Pipeline**:
   - `ToolManager` → CHUK Tool Processor → MCP Servers (STDIO/HTTP/SSE)
   - Supports concurrent execution, streaming responses, and progress tracking
   - Tool validation and sanitization for provider compatibility

### Key Architectural Patterns

- **Async-Native**: All core operations use asyncio; synchronous wrappers provided for CLI compatibility
- **Provider Abstraction**: LLM providers managed through chuk-llm with defaults to Ollama/gpt-oss
- **Registry Pattern**: Commands and tools use registry patterns for dynamic registration
- **Layered Dependencies**: Clear separation between UI (rich), CLI (typer), tool processing (chuk-tool-processor), and LLM management (chuk-llm)

### Directory Structure & Purpose

```
src/mcp_cli/
├── chat/                 # Chat mode implementation (streaming, conversation management)
├── cli/                  # CLI command structure and registry
├── commands/             # Core command implementations (provider, tools, servers, etc.)
├── interactive/          # Interactive shell mode with command processing
├── llm/                  # LLM integration layer wrapping chuk-llm
├── tools/                # Tool management, validation, and execution
├── ui/                   # Rich terminal UI components and formatting
└── utils/                # Shared utilities (async helpers, config management)
```

### Package Management

MCP-CLI uses **`uv`** for all package management. See [Package Management Guide](./docs/PACKAGE_MANAGEMENT.md) for detailed usage.

### UI System

MCP CLI features comprehensive UI management powered by the **chuk-term** library:

- **Output Management**: Centralized output system using chuk-term's output module with theme support and consistent formatting across the application.
- **Theme System**: Leverages chuk-term's built-in themes (default, dark, light, minimal, terminal, monokai, dracula, solarized) with automatic theme adaptation.
- **Terminal Management**: Cross-platform terminal operations provided by chuk-term including clearing, resizing, color detection, cursor control, and more.
- **Interactive Prompts**: User input handling through chuk-term's prompt system (ask, confirm, select_from_list, select_multiple).

The UI system has been migrated to use chuk-term (https://github.com/chrishayuk/chuk-term) for improved consistency and maintainability.

#### Demo Scripts

Interactive demonstrations of UI capabilities:

```bash
# Terminal management features
uv run examples/ui_terminal_demo.py

# Output system features  
uv run examples/ui_output_demo.py

# Streaming UI capabilities (NEW)
uv run examples/ui_streaming_demo.py
```


### Critical Files for Understanding Architecture

- `src/mcp_cli/main.py`: Entry point, command routing, mode selection
- `src/mcp_cli/llm/model_manager.py`: Provider/model management, defaults handling
- `src/mcp_cli/tools/tool_manager.py`: Tool discovery, execution, and validation
- `src/mcp_cli/chat/conversation_processor.py`: Chat conversation flow orchestration
- `src/mcp_cli/cli/registry.py`: Command registration and dispatch system
- `src/mcp_cli/ui/output.py`: Centralized output management with theme support
- `src/mcp_cli/ui/terminal.py`: Terminal operations and enhanced features
- `src/mcp_cli/ui/theme.py`: Theme system and style management

### Configuration Management

- **Server Config**: JSON files define MCP server connections (`server_config.json`)
- **Provider Config**: Managed through chuk-llm configuration system (`~/.chuk_llm/config.yaml`)
- **Environment Variables**: Support for API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- **Persistent State**: Active provider/model maintained by ModelManager

### Testing Approach

- Uses `pytest` with `pytest-asyncio` for async testing
- Mock-based testing for external dependencies
- Test files mirror source structure in `tests/` directory
- Focus on unit tests with module isolation
- Integration tests for core functionality without external service dependencies

### Default Behavior

- **Default Provider**: Ollama (local, no API key required)
- **Default Model**: gpt-oss (open-source reasoning model)
- **Default Mode**: Chat mode when no subcommand specified
- **Tool Confirmation**: Enabled by default in chat mode
- **Verbose Mode**: Enabled by default for tool execution visibility