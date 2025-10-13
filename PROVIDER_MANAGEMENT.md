# Provider Management for MCP CLI

## Overview
MCP CLI now supports adding and managing custom OpenAI-compatible providers at runtime. This allows users to connect to any OpenAI-compatible API endpoint without modifying chuk_llm configuration files.

## Features

### 1. Persistent Custom Providers
Custom providers can be added and saved to `~/.mcp-cli/preferences.json`. These providers persist across sessions.

**Add a custom provider:**
```bash
mcp-cli provider add <name> <api_base> [model1 model2 ...]

# Example
mcp-cli provider add localai http://localhost:8080/v1 gpt-4 gpt-3.5-turbo
```

**Remove a custom provider:**
```bash
mcp-cli provider remove <name>
```

**List custom providers:**
```bash
mcp-cli provider custom
```

### 2. Runtime Providers (Session-only)
Providers can be configured at runtime via CLI flags. These are not persisted and only exist for the current session.

```bash
mcp-cli --provider <name> --api-base <url> [--api-key <key>] --server <server>

# Example (API key in environment)
export LOCALAI_API_KEY=your-key
mcp-cli --provider localai --api-base http://localhost:8080/v1 --server sqlite

# Example (API key via CLI - kept in memory only)
mcp-cli --provider localai --api-base http://localhost:8080/v1 --api-key temp-key --server sqlite
```

### 3. API Key Management
**Security-first approach:** API keys are NEVER stored in configuration files.

- **Environment Variables**: Default pattern is `{PROVIDER_NAME}_API_KEY`
  - Example: Provider "localai" uses `LOCALAI_API_KEY`
  - Provider "my-ai" uses `MY_AI_API_KEY`
  
- **Runtime API Keys**: Can be passed via `--api-key` flag
  - Kept in memory only for the session
  - Not persisted to disk

### 4. Provider Discovery
Custom providers automatically appear in:
- `mcp-cli providers` - Full provider list
- `mcp-cli provider list` - Provider list with details  
- Model selection menus
- Provider switching commands

## Implementation Details

### Configuration Storage
- **Preferences**: `~/.mcp-cli/preferences.json`
  - Stores provider name, API base URL, models list
  - Does NOT store API keys
  
- **chuk_llm Config**: `~/.chuk_llm/providers.yaml`
  - Built-in providers (openai, anthropic, etc.)
  - Not modified by MCP CLI

### Code Structure
- `src/mcp_cli/utils/preferences.py` - CustomProvider class and persistence
- `src/mcp_cli/model_manager.py` - Provider loading and client creation
- `src/mcp_cli/commands/actions/providers.py` - Provider commands
- `src/mcp_cli/main.py` - CLI argument handling

### Custom Provider Client
Custom providers use the OpenAI Python client with:
- Custom `base_url` pointing to the provider's API endpoint
- API key from environment or runtime flag
- Full compatibility with OpenAI's API format

## Usage Examples

### Example 1: Local AI Server
```bash
# Add LocalAI as a provider
mcp-cli provider add localai http://localhost:8080/v1 gpt-4 gpt-3.5-turbo

# Set API key
export LOCALAI_API_KEY=dummy-key

# Use LocalAI
mcp-cli --provider localai --server sqlite
```

### Example 2: Custom OpenAI Proxy
```bash
# Add proxy with custom models
mcp-cli provider add myproxy https://proxy.example.com/v1 custom-model-1 custom-model-2

# Set API key
export MYPROXY_API_KEY=your-api-key

# Switch to proxy
mcp-cli provider myproxy
```

### Example 3: Temporary Provider
```bash
# Use a provider for just one session
mcp-cli --provider temp-ai --api-base https://api.temp.com/v1 --api-key test-key --server sqlite
```

## Limitations

1. **Model Discovery**: Custom providers don't support automatic model discovery. Models must be specified when adding the provider.

2. **Feature Detection**: All custom providers are assumed to support standard OpenAI features (streaming, tools, text).

3. **Runtime Providers in Subcommands**: Runtime providers configured via `--provider` and `--api-base` are only available in the main chat mode, not in subcommands like `providers list`.

## Future Enhancements

- Model discovery for custom providers
- Feature detection based on provider capabilities
- Support for non-OpenAI compatible providers
- Provider templates for common configurations