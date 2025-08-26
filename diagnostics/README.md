# MCP CLI Diagnostic Scripts

This directory contains diagnostic and testing utilities for the MCP CLI system. These scripts help debug issues, validate functionality, and ensure proper system configuration.

## Quick Start

All diagnostic scripts can be run with `uv`:

```bash
# Run any diagnostic script
uv run diagnostics/<script_name>.py

# Example: Test all CLI commands
uv run diagnostics/cli_command_diagnostic.py

# Example: Check server health
uv run diagnostics/mcp_server_diagnostic.py
```

## Available Diagnostic Scripts

### 🔧 Core Functionality Tests

#### `cli_command_diagnostic.py`
**Purpose:** Comprehensive test suite for all MCP CLI commands  
**Tests:** Help commands, server commands, provider commands, tool discovery  
**Usage:** `uv run diagnostics/cli_command_diagnostic.py`  
**Output:** Pass/fail status for 24+ different CLI commands

#### `cli_tools_diagnostic.py`
**Purpose:** Tests tool discovery and execution functionality  
**Tests:** Tool manager initialization, tool execution with/without arguments  
**Usage:** `uv run diagnostics/cli_tools_diagnostic.py`  
**Output:** Execution traces and success metrics

#### `conversation_tools_diagnostic.py`
**Purpose:** Tests the conversation flow with tool calls (simplified version)  
**Tests:** ToolManager init, ChatContext creation, tool discovery, tool execution  
**Usage:** `uv run diagnostics/conversation_tools_diagnostic.py`  
**Output:** Minimal output showing each test step status

### 🔍 System Analysis

#### `mcp_cli_diagnostics.py`
**Purpose:** Analyzes MCP CLI installation and environment setup  
**Checks:** Package dependencies, API keys, configuration files, environment variables  
**Usage:** `uv run diagnostics/mcp_cli_diagnostics.py`  
**Output:** System configuration report with recommendations

#### `mcp_server_diagnostic.py`
**Purpose:** Comprehensive MCP server health check and capability analysis  
**Features:** Server connectivity, capability matrix, performance metrics, tool inventory, runtime server testing  
**Tests:** Runtime server addition/removal, preference storage, existing runtime server detection  
**Usage:** `uv run diagnostics/mcp_server_diagnostic.py`  
**Output:** Detailed server analysis with health status, runtime server management test results, and recommendations

#### `debug_models.py`
**Purpose:** Diagnoses model discovery and provider configuration issues  
**Checks:** chuk-llm version, API keys, provider status, Ollama detection  
**Usage:** `uv run diagnostics/debug_models.py`  
**Output:** Provider configuration analysis with model counts

### 🧪 Specialized Testing

#### `chained_tools_diagnostic.py`
**Purpose:** Tests complex chained tool call scenarios  
**Tests:** Sequential tool calls, streaming vs direct execution, error handling  
**Usage:** `uv run diagnostics/chained_tools_diagnostic.py [--config FILE] [--server NAME]`  
**Output:** Detailed execution traces with timing and error analysis

#### `provider_list_diagnostic.py`
**Purpose:** Validates provider listing and status detection logic  
**Tests:** Ollama detection, API key validation, model counting  
**Usage:** `uv run diagnostics/provider_list_diagnostic.py`  
**Output:** Provider status matrix and test results

#### `server_integration_test.py`
**Purpose:** Generic MCP server integration testing  
**Features:** Works with any configured server, tests tool discovery and execution  
**Usage:** `uv run diagnostics/server_integration_test.py [--server NAME]`  
**Output:** Server capabilities and tool test results

#### `timeout_investigation.py`
**Purpose:** Investigates and patches timeout issues in tool execution  
**Features:** Traces timeout sources, creates runtime patches  
**Usage:** `uv run diagnostics/timeout_investigation.py`  
**Output:** Timeout configuration analysis and patch creation

## Common Use Cases

### Debugging Tool Execution Issues
```bash
# Test basic tool execution
uv run diagnostics/cli_tools_diagnostic.py

# Test chained tool calls (complex scenarios)
uv run diagnostics/chained_tools_diagnostic.py

# Test conversation flow
uv run diagnostics/conversation_tools_diagnostic.py
```

### Server Health Checks
```bash
# Full server diagnostic
uv run diagnostics/mcp_server_diagnostic.py

# Test specific server integration
uv run diagnostics/server_integration_test.py --server sqlite
```

### System Configuration Issues
```bash
# Check MCP CLI installation
uv run diagnostics/mcp_cli_diagnostics.py

# Debug model/provider issues
uv run diagnostics/debug_models.py

# Check provider configuration
uv run diagnostics/provider_list_diagnostic.py
```

### Performance Issues
```bash
# Investigate timeouts
uv run diagnostics/timeout_investigation.py

# Check server performance
uv run diagnostics/mcp_server_diagnostic.py
```

## Output Interpretation

### Success Indicators
- ✅ **Green checkmarks** - Test passed
- 🎉 **Celebration emoji** - Excellent status
- 🚀 **Rocket** - Fast performance

### Warning Indicators
- ⚠️ **Warning sign** - Non-critical issue
- 💡 **Light bulb** - Suggestions available
- 🔍 **Magnifying glass** - Diagnostic in progress

### Error Indicators
- ❌ **Red X** - Test failed
- 🚨 **Siren** - Critical issue
- 📉 **Down chart** - Poor performance

## Troubleshooting

### Common Issues and Solutions

**Issue:** Scripts fail with import errors  
**Solution:** Ensure you're running from the project root with `uv run`

**Issue:** Server connection failures  
**Solution:** Check `server_config.json` exists and servers are properly configured

**Issue:** Tool execution timeouts  
**Solution:** Run `timeout_investigation.py` to diagnose and create patches

**Issue:** No models found  
**Solution:** Run `debug_models.py` to check provider configuration

## Contributing

When adding new diagnostic scripts:

1. Follow the naming convention: `<feature>_diagnostic.py` or `<feature>_test.py`
2. Include clear docstrings explaining purpose and usage
3. Use consistent output formatting with emoji indicators
4. Handle errors gracefully with helpful error messages
5. Update this README with the new script documentation

## Dependencies

All diagnostic scripts require:
- Python 3.8+
- `uv` package manager
- MCP CLI installed (`pip install -e .` or `make dev-install`)
- Valid `server_config.json` for server-related tests

Optional dependencies:
- `rich` - Enhanced terminal output (automatically used if available)
- Active Ollama installation for Ollama-specific tests
- API keys for provider-specific tests