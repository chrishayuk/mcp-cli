# MCP CLI - Model Context Protocol Command Line Interface

A powerful, feature-rich command-line interface for interacting with Model Context Protocol servers. This client enables seamless communication with LLMs through integration with the [CHUK Tool Processor](https://github.com/chrishayuk/chuk-tool-processor) and [CHUK-LLM](https://github.com/chrishayuk/chuk-llm), providing tool usage, conversation management, and multiple operational modes.

**Default Configuration**: MCP CLI defaults to using Ollama with the `gpt-oss` reasoning model for local, privacy-focused operation without requiring API keys.

## 🔄 Architecture Overview

The MCP CLI is built on a modular architecture with clean separation of concerns:

- **[CHUK Tool Processor](https://github.com/chrishayuk/chuk-tool-processor)**: Async-native tool execution and MCP server communication
- **[CHUK-LLM](https://github.com/chrishayuk/chuk-llm)**: Unified LLM provider configuration and client management with 200+ auto-generated functions
- **[CHUK-Term](https://github.com/chrishayuk/chuk-term)**: Enhanced terminal UI with themes, cross-platform terminal management, and rich formatting
- **MCP CLI**: Command orchestration and integration layer (this project)

## 🌟 Features

### Multiple Operational Modes
- **Chat Mode**: Conversational interface with streaming responses and automated tool usage (default: Ollama/gpt-oss)
- **Interactive Mode**: Command-driven shell interface for direct server operations
- **Command Mode**: Unix-friendly mode for scriptable automation and pipelines
- **Direct Commands**: Run individual commands without entering interactive mode

### Advanced Chat Interface
- **Streaming Responses**: Real-time response generation with live UI updates
- **Reasoning Visibility**: See AI's thinking process with reasoning models (gpt-oss, GPT-5, Claude 4)
- **Concurrent Tool Execution**: Execute multiple tools simultaneously while preserving conversation order
- **Smart Interruption**: Interrupt streaming responses or tool execution with Ctrl+C
- **Performance Metrics**: Response timing, words/second, and execution statistics
- **Rich Formatting**: Markdown rendering, syntax highlighting, and progress indicators

### Comprehensive Provider Support

MCP CLI supports all providers and models from CHUK-LLM, including cutting-edge reasoning models:

| Provider | Key Models | Special Features |
|----------|------------|------------------|
| **Ollama** (Default) | 🧠 gpt-oss, llama3.3, llama3.2, qwen3, qwen2.5-coder, deepseek-coder, granite3.3, mistral, gemma3, phi3, codellama | Local reasoning models, privacy-focused, no API key required |
| **OpenAI** | 🚀 GPT-5 family (gpt-5, gpt-5-mini, gpt-5-nano), GPT-4o family, O3 series (o3, o3-mini) | Advanced reasoning, function calling, vision |
| **Anthropic** | 🧠 Claude 4 family (claude-4-1-opus, claude-4-sonnet), Claude 3.5 Sonnet | Enhanced reasoning, long context |
| **Azure OpenAI** 🏢 | Enterprise GPT-5, GPT-4 models | Private endpoints, compliance, audit logs |
| **Google Gemini** | Gemini 2.0 Flash, Gemini 1.5 Pro | Multimodal, fast inference |
| **Groq** ⚡ | Llama 3.1 models, Mixtral | Ultra-fast inference (500+ tokens/sec) |
| **Perplexity** 🌐 | Sonar models | Real-time web search with citations |
| **IBM watsonx** 🏢 | Granite, Llama models | Enterprise compliance |
| **Mistral AI** 🇪🇺 | Mistral Large, Medium | European, efficient models |

### Robust Tool System
- **Automatic Discovery**: Server-provided tools are automatically detected and catalogued
- **Provider Adaptation**: Tool names are automatically sanitized for provider compatibility
- **Concurrent Execution**: Multiple tools can run simultaneously with proper coordination
- **Rich Progress Display**: Real-time progress indicators and execution timing
- **Tool History**: Complete audit trail of all tool executions
- **Streaming Tool Calls**: Support for tools that return streaming data

### Advanced Configuration Management
- **Environment Integration**: API keys and settings via environment variables
- **File-based Config**: YAML and JSON configuration files
- **User Preferences**: Persistent settings for active providers and models
- **Validation & Diagnostics**: Built-in provider health checks and configuration validation

### Enhanced User Experience
- **Cross-Platform Support**: Windows, macOS, and Linux with platform-specific optimizations via chuk-term
- **Rich Console Output**: Powered by chuk-term with 8 built-in themes (default, dark, light, minimal, terminal, monokai, dracula, solarized)
- **Advanced Terminal Management**: Cross-platform terminal operations including clearing, resizing, color detection, and cursor control
- **Interactive UI Components**: User input handling through chuk-term's prompt system (ask, confirm, select_from_list, select_multiple)
- **Command Completion**: Context-aware tab completion for all interfaces
- **Comprehensive Help**: Detailed help system with examples and usage patterns
- **Graceful Error Handling**: User-friendly error messages with troubleshooting hints

## 📋 Prerequisites

- **Python 3.11 or higher**
- **For Local Operation (Default)**:
  - Ollama: Install from [ollama.ai](https://ollama.ai)
  - Pull the default reasoning model: `ollama pull gpt-oss`
- **For Cloud Providers** (Optional):
  - OpenAI: `OPENAI_API_KEY` environment variable (for GPT-5, GPT-4, O3 models)
  - Anthropic: `ANTHROPIC_API_KEY` environment variable (for Claude 4, Claude 3.5)
  - Azure: `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` (for enterprise GPT-5)
  - Google: `GEMINI_API_KEY` (for Gemini models)
  - Groq: `GROQ_API_KEY` (for fast Llama models)
  - Custom providers: Provider-specific configuration
- **MCP Servers**: Server configuration file (default: `server_config.json`)

## 🚀 Installation

### Quick Start with Ollama (Default)

1. **Install Ollama** (if not already installed):
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or visit https://ollama.ai for other installation methods
```

2. **Pull the default reasoning model**:
```bash
ollama pull gpt-oss  # Open-source reasoning model with thinking visibility
```

3. **Install and run MCP CLI**:
```bash
# Using uvx (recommended)
uvx mcp-cli --help

# Or install from source
git clone https://github.com/chrishayuk/mcp-cli
cd mcp-cli
pip install -e "."
mcp-cli --help
```

### Using Different Models

```bash
# === LOCAL MODELS (No API Key Required) ===

# Use default reasoning model (gpt-oss)
mcp-cli --server sqlite

# Use other Ollama models
mcp-cli --model llama3.3              # Latest Llama
mcp-cli --model qwen2.5-coder         # Coding-focused
mcp-cli --model deepseek-coder        # Another coding model
mcp-cli --model granite3.3            # IBM Granite

# === CLOUD PROVIDERS (API Keys Required) ===

# GPT-5 Family (requires OpenAI API key)
mcp-cli --provider openai --model gpt-5          # Full GPT-5 with reasoning
mcp-cli --provider openai --model gpt-5-mini     # Efficient GPT-5 variant
mcp-cli --provider openai --model gpt-5-nano     # Ultra-lightweight GPT-5

# GPT-4 Family
mcp-cli --provider openai --model gpt-4o         # GPT-4 Optimized
mcp-cli --provider openai --model gpt-4o-mini    # Smaller GPT-4

# O3 Reasoning Models
mcp-cli --provider openai --model o3             # O3 reasoning
mcp-cli --provider openai --model o3-mini        # Efficient O3

# Claude 4 Family (requires Anthropic API key)
mcp-cli --provider anthropic --model claude-4-1-opus    # Most advanced Claude
mcp-cli --provider anthropic --model claude-4-sonnet    # Balanced Claude 4
mcp-cli --provider anthropic --model claude-3-5-sonnet  # Claude 3.5

# Enterprise Azure (requires Azure configuration)
mcp-cli --provider azure_openai --model gpt-5    # Enterprise GPT-5

# Other Providers
mcp-cli --provider gemini --model gemini-2.0-flash      # Google Gemini
mcp-cli --provider groq --model llama-3.1-70b          # Fast Llama via Groq
```

## 🧰 Global Configuration

### Default Configuration

MCP CLI defaults to:
- **Provider**: `ollama` (local, no API key required)
- **Model**: `gpt-oss` (open-source reasoning model with thinking visibility)

### Command-line Arguments

Global options available for all modes and commands:

- `--server`: Specify server(s) to connect to (comma-separated)
- `--config-file`: Path to server configuration file (default: `server_config.json`)
- `--provider`: LLM provider (default: `ollama`)
- `--model`: Specific model to use (default: `gpt-oss` for Ollama)
- `--disable-filesystem`: Disable filesystem access (default: enabled)
- `--api-base`: Override API endpoint URL
- `--api-key`: Override API key (not needed for Ollama)
- `--verbose`: Enable detailed logging
- `--quiet`: Suppress non-essential output

### Environment Variables

```bash
# Override defaults
export LLM_PROVIDER=ollama              # Default provider (already the default)
export LLM_MODEL=gpt-oss                # Default model (already the default)

# For cloud providers (optional)
export OPENAI_API_KEY=sk-...           # For GPT-5, GPT-4, O3 models
export ANTHROPIC_API_KEY=sk-ant-...    # For Claude 4, Claude 3.5
export AZURE_OPENAI_API_KEY=sk-...     # For enterprise GPT-5
export AZURE_OPENAI_ENDPOINT=https://...
export GEMINI_API_KEY=...              # For Gemini models
export GROQ_API_KEY=...                # For Groq fast inference

# Tool configuration
export MCP_TOOL_TIMEOUT=120            # Tool execution timeout (seconds)
```

## 🌐 Available Modes

### 1. Chat Mode (Default)

Provides a natural language interface with streaming responses and automatic tool usage:

```bash
# Default mode with Ollama/gpt-oss reasoning model (no API key needed)
mcp-cli --server sqlite

# See the AI's thinking process with reasoning models
mcp-cli --server sqlite --model gpt-oss     # Open-source reasoning
mcp-cli --server sqlite --provider openai --model gpt-5  # GPT-5 reasoning
mcp-cli --server sqlite --provider anthropic --model claude-4-1-opus  # Claude 4 reasoning

# Use different local models
mcp-cli --server sqlite --model llama3.3
mcp-cli --server sqlite --model qwen2.5-coder

# Switch to cloud providers (requires API keys)
mcp-cli chat --server sqlite --provider openai --model gpt-5
mcp-cli chat --server sqlite --provider anthropic --model claude-4-sonnet
```

### 2. Interactive Mode

Command-driven shell interface for direct server operations:

```bash
mcp-cli interactive --server sqlite

# With specific models
mcp-cli interactive --server sqlite --model gpt-oss       # Local reasoning
mcp-cli interactive --server sqlite --provider openai --model gpt-5  # Cloud GPT-5
```

### 3. Command Mode

Unix-friendly interface for automation and scripting:

```bash
# Process text with reasoning models
mcp-cli cmd --server sqlite --model gpt-oss --prompt "Think through this step by step" --input data.txt

# Use GPT-5 for complex reasoning
mcp-cli cmd --server sqlite --provider openai --model gpt-5 --prompt "Analyze this data" --input data.txt

# Execute tools directly
mcp-cli cmd --server sqlite --tool list_tables --output tables.json

# Pipeline-friendly processing
echo "SELECT * FROM users LIMIT 5" | mcp-cli cmd --server sqlite --tool read_query --input -
```

### 4. Direct Commands

Execute individual commands without entering interactive mode:

```bash
# List available tools
mcp-cli tools --server sqlite

# Show provider configuration
mcp-cli provider list

# Show available models for current provider
mcp-cli models

# Show models for specific provider
mcp-cli models openai    # Shows GPT-5, GPT-4, O3 models
mcp-cli models anthropic # Shows Claude 4, Claude 3.5 models
mcp-cli models ollama    # Shows gpt-oss, llama3.3, etc.

# Ping servers
mcp-cli ping --server sqlite

# List resources
mcp-cli resources --server sqlite

# UI Theme Management
mcp-cli theme                     # Show current theme and list available
mcp-cli theme dark                # Switch to dark theme
mcp-cli theme --select            # Interactive theme selector
mcp-cli theme --list              # List all available themes
```

## 🤖 Using Chat Mode

Chat mode provides the most advanced interface with streaming responses and intelligent tool usage.

### Starting Chat Mode

```bash
# Simple startup with default reasoning model (gpt-oss)
mcp-cli --server sqlite

# Multiple servers
mcp-cli --server sqlite,filesystem

# With advanced reasoning models
mcp-cli --server sqlite --provider openai --model gpt-5
mcp-cli --server sqlite --provider anthropic --model claude-4-1-opus
```

### Chat Commands (Slash Commands)

#### Provider & Model Management
```bash
/provider                           # Show current configuration (default: ollama)
/provider list                      # List all providers
/provider config                    # Show detailed configuration
/provider diagnostic               # Test provider connectivity
/provider set ollama api_base http://localhost:11434  # Configure Ollama endpoint
/provider openai                   # Switch to OpenAI (requires API key)
/provider anthropic                # Switch to Anthropic (requires API key)
/provider openai gpt-5             # Switch to OpenAI GPT-5

/model                             # Show current model (default: gpt-oss)
/model llama3.3                    # Switch to different Ollama model
/model gpt-5                       # Switch to GPT-5 (if using OpenAI)
/model claude-4-1-opus             # Switch to Claude 4 (if using Anthropic)
/models                            # List available models for current provider
```

#### Tool Management
```bash
/tools                             # List available tools
/tools --all                       # Show detailed tool information
/tools --raw                       # Show raw JSON definitions
/tools call                        # Interactive tool execution

/toolhistory                       # Show tool execution history
/th -n 5                          # Last 5 tool calls
/th 3                             # Details for call #3
/th --json                        # Full history as JSON
```

#### Conversation Management
```bash
/conversation                      # Show conversation history
/ch -n 10                         # Last 10 messages
/ch 5                             # Details for message #5
/ch --json                        # Full history as JSON

/save conversation.json            # Save conversation to file
/compact                          # Summarize conversation
/clear                            # Clear conversation history
/cls                              # Clear screen only
```

#### UI Customization
```bash
/theme                            # Interactive theme selector with preview
/theme dark                       # Switch to dark theme
/theme monokai                    # Switch to monokai theme

# Available themes: default, dark, light, minimal, terminal, monokai, dracula, solarized
# Themes are persisted across sessions
```

#### Session Control
```bash
/verbose                          # Toggle verbose/compact display (Default: Enabled)
/confirm                          # Toggle tool call confirmation (Default: Enabled)
/interrupt                        # Stop running operations
/servers                          # List connected servers
/help                            # Show all commands
/help tools                       # Help for specific command
/exit                            # Exit chat mode
```

### Chat Features

#### Streaming Responses with Reasoning Visibility
- **🧠 Reasoning Models**: See the AI's thinking process with gpt-oss, GPT-5, Claude 4
- **Real-time Generation**: Watch text appear token by token
- **Performance Metrics**: Words/second, response time
- **Graceful Interruption**: Ctrl+C to stop streaming
- **Progressive Rendering**: Markdown formatted as it streams

#### Tool Execution
- Automatic tool discovery and usage
- Concurrent execution with progress indicators
- Verbose and compact display modes
- Complete execution history and timing

#### Provider Integration
- Seamless switching between providers
- Model-specific optimizations
- API key and endpoint management
- Health monitoring and diagnostics

## 🖥️ Using Interactive Mode

Interactive mode provides a command shell for direct server interaction.

### Starting Interactive Mode

```bash
mcp-cli interactive --server sqlite
```

### Interactive Commands

```bash
help                              # Show available commands
exit                              # Exit interactive mode
clear                             # Clear terminal

# Provider management
provider                          # Show current provider
provider list                     # List providers
provider anthropic                # Switch provider
provider openai gpt-5             # Switch to GPT-5

# Model management
model                             # Show current model
model gpt-oss                     # Switch to reasoning model
model claude-4-1-opus             # Switch to Claude 4
models                            # List available models

# Tool operations
tools                             # List tools
tools --all                       # Detailed tool info
tools call                        # Interactive tool execution

# Server operations
servers                           # List servers
ping                              # Ping all servers
resources                         # List resources
prompts                           # List prompts
```

## 📄 Using Command Mode

Command mode provides Unix-friendly automation capabilities.

### Command Mode Options

```bash
--input FILE                      # Input file (- for stdin)
--output FILE                     # Output file (- for stdout)
--prompt TEXT                     # Prompt template
--tool TOOL                       # Execute specific tool
--tool-args JSON                  # Tool arguments as JSON
--system-prompt TEXT              # Custom system prompt
--raw                             # Raw output without formatting
--single-turn                     # Disable multi-turn conversation
--max-turns N                     # Maximum conversation turns
```

### Examples

```bash
# Text processing with reasoning models
echo "Analyze this data" | mcp-cli cmd --server sqlite --model gpt-oss --input - --output analysis.txt

# Use GPT-5 for complex analysis
mcp-cli cmd --server sqlite --provider openai --model gpt-5 --prompt "Provide strategic analysis" --input report.txt

# Tool execution
mcp-cli cmd --server sqlite --tool list_tables --raw

# Complex queries
mcp-cli cmd --server sqlite --tool read_query --tool-args '{"query": "SELECT COUNT(*) FROM users"}'

# Batch processing with GNU Parallel
ls *.txt | parallel mcp-cli cmd --server sqlite --input {} --output {}.summary --prompt "Summarize: {{input}}"
```

## 🔧 Provider Configuration

### Ollama Configuration (Default)

Ollama runs locally by default on `http://localhost:11434`. To use reasoning and other models:

```bash
# Pull reasoning and other models for Ollama
ollama pull gpt-oss          # Default reasoning model
ollama pull llama3.3         # Latest Llama
ollama pull llama3.2         # Llama 3.2
ollama pull qwen3            # Qwen 3
ollama pull qwen2.5-coder    # Coding-focused
ollama pull deepseek-coder   # DeepSeek coder
ollama pull granite3.3       # IBM Granite
ollama pull mistral          # Mistral
ollama pull gemma3           # Google Gemma
ollama pull phi3             # Microsoft Phi
ollama pull codellama        # Code Llama

# List available Ollama models
ollama list

# Configure remote Ollama server
mcp-cli provider set ollama api_base http://remote-server:11434
```

### Cloud Provider Configuration

To use cloud providers with advanced models, configure API keys:

```bash
# Configure OpenAI (for GPT-5, GPT-4, O3 models)
mcp-cli provider set openai api_key sk-your-key-here

# Configure Anthropic (for Claude 4, Claude 3.5)
mcp-cli provider set anthropic api_key sk-ant-your-key-here

# Configure Azure OpenAI (for enterprise GPT-5)
mcp-cli provider set azure_openai api_key sk-your-key-here
mcp-cli provider set azure_openai api_base https://your-resource.openai.azure.com

# Configure other providers
mcp-cli provider set gemini api_key your-gemini-key
mcp-cli provider set groq api_key your-groq-key

# Test configuration
mcp-cli provider diagnostic openai
mcp-cli provider diagnostic anthropic
```

### Manual Configuration

The `chuk_llm` library configuration in `~/.chuk_llm/config.yaml`:

```yaml
ollama:
  api_base: http://localhost:11434
  default_model: gpt-oss

openai:
  api_base: https://api.openai.com/v1
  default_model: gpt-5

anthropic:
  api_base: https://api.anthropic.com
  default_model: claude-4-1-opus

azure_openai:
  api_base: https://your-resource.openai.azure.com
  default_model: gpt-5

gemini:
  api_base: https://generativelanguage.googleapis.com
  default_model: gemini-2.0-flash

groq:
  api_base: https://api.groq.com
  default_model: llama-3.1-70b
```

API keys (if using cloud providers) in `~/.chuk_llm/.env`:

```bash
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
AZURE_OPENAI_API_KEY=sk-your-azure-key-here
GEMINI_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key
```

## 📂 Server Configuration

Create a `server_config.json` file with your MCP server configurations:

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "python",
      "args": ["-m", "mcp_server.sqlite_server"],
      "env": {
        "DATABASE_PATH": "database.db"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"],
      "env": {}
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-brave-api-key"
      }
    }
  }
}
```

## 📈 Advanced Usage Examples

### Reasoning Model Comparison

```bash
# Compare reasoning across different models
> /provider ollama
> /model gpt-oss
> Think through this problem step by step: If a train leaves New York at 3 PM...
[See the complete thinking process with gpt-oss]

> /provider openai
> /model gpt-5
> Think through this problem step by step: If a train leaves New York at 3 PM...
[See GPT-5's reasoning approach]

> /provider anthropic
> /model claude-4-1-opus
> Think through this problem step by step: If a train leaves New York at 3 PM...
[See Claude 4's analytical process]
```

### Local-First Workflow with Reasoning

```bash
# Start with default Ollama/gpt-oss (no API key needed)
mcp-cli chat --server sqlite

# Use reasoning model for complex problems
> Think through this database optimization problem step by step
[gpt-oss shows its complete thinking process before answering]

# Try different local models for different tasks
> /model llama3.3              # General purpose
> /model qwen2.5-coder         # For coding tasks
> /model deepseek-coder        # Alternative coding model
> /model granite3.3            # IBM's model
> /model gpt-oss               # Back to reasoning model

# Switch to cloud when needed (requires API keys)
> /provider openai
> /model gpt-5
> Complex enterprise architecture design...

> /provider anthropic
> /model claude-4-1-opus
> Detailed strategic analysis...

> /provider ollama
> /model gpt-oss
> Continue with local processing...
```

### Multi-Provider Workflow

```bash
# Start with local reasoning (default, no API key)
mcp-cli chat --server sqlite

# Compare responses across providers
> /provider ollama
> What's the best way to optimize this SQL query?

> /provider openai gpt-5        # Requires API key
> What's the best way to optimize this SQL query?

> /provider anthropic claude-4-sonnet  # Requires API key
> What's the best way to optimize this SQL query?

# Use each provider's strengths
> /provider ollama gpt-oss      # Local reasoning, privacy
> /provider openai gpt-5        # Advanced reasoning
> /provider anthropic claude-4-1-opus  # Deep analysis
> /provider groq llama-3.1-70b  # Ultra-fast responses
```

### Complex Tool Workflows with Reasoning

```bash
# Use reasoning model for complex database tasks
> /model gpt-oss
> I need to analyze our database performance. Think through what we should check first.
[gpt-oss shows thinking: "First, I should check the table structure, then indexes, then query patterns..."]
[Tool: list_tables] → products, customers, orders

> Now analyze the indexes and suggest optimizations
[gpt-oss thinks through index analysis]
[Tool: describe_table] → Shows current indexes
[Tool: read_query] → Analyzes query patterns

> Create an optimization plan based on your analysis
[Complete reasoning process followed by specific recommendations]
```

### Automation and Scripting

```bash
# Batch processing with different models
for file in data/*.csv; do
  # Use reasoning model for analysis
  mcp-cli cmd --server sqlite \
    --model gpt-oss \
    --prompt "Analyze this data and think through patterns" \
    --input "$file" \
    --output "analysis/$(basename "$file" .csv)_reasoning.txt"
  
  # Use coding model for generating scripts
  mcp-cli cmd --server sqlite \
    --model qwen2.5-coder \
    --prompt "Generate Python code to process this data" \
    --input "$file" \
    --output "scripts/$(basename "$file" .csv)_script.py"
done

# Pipeline with reasoning
cat complex_problem.txt | \
  mcp-cli cmd --model gpt-oss --prompt "Think through this step by step" --input - | \
  mcp-cli cmd --model llama3.3 --prompt "Summarize the key points" --input - > solution.txt
```

### Performance Monitoring

```bash
# Check provider and model performance
> /provider diagnostic
Provider Diagnostics
Provider      | Status      | Response Time | Features      | Models
ollama        | ✅ Ready    | 56ms         | 📡🔧         | gpt-oss, llama3.3, qwen3, ...
openai        | ✅ Ready    | 234ms        | 📡🔧👁️      | gpt-5, gpt-4o, o3, ...
anthropic     | ✅ Ready    | 187ms        | 📡🔧         | claude-4-1-opus, claude-4-sonnet, ...
azure_openai  | ✅ Ready    | 198ms        | 📡🔧👁️      | gpt-5, gpt-4o, ...
gemini        | ✅ Ready    | 156ms        | 📡🔧👁️      | gemini-2.0-flash, ...
groq          | ✅ Ready    | 45ms         | 📡🔧         | llama-3.1-70b, ...

# Check available models
> /models
Models for ollama (Current Provider)
Model                | Status
gpt-oss             | Current & Default (Reasoning)
llama3.3            | Available
llama3.2            | Available
qwen2.5-coder       | Available
deepseek-coder      | Available
granite3.3          | Available
... and 6 more

# Monitor tool execution with reasoning
> /verbose
> /model gpt-oss
> Analyze the database and optimize the slowest queries
[Shows complete thinking process]
[Tool execution with timing]
```

## 🔍 Troubleshooting

### Common Issues

1. **Ollama not running** (default provider):
   ```bash
   # Start Ollama service
   ollama serve
   
   # Or check if it's running
   curl http://localhost:11434/api/tags
   ```

2. **Model not found**:
   ```bash
   # For Ollama (default), pull the model first
   ollama pull gpt-oss      # Reasoning model
   ollama pull llama3.3     # Latest Llama
   ollama pull qwen2.5-coder # Coding model
   
   # List available models
   ollama list
   
   # For cloud providers, check supported models
   mcp-cli models openai     # Shows GPT-5, GPT-4, O3 models
   mcp-cli models anthropic  # Shows Claude 4, Claude 3.5 models
   ```

3. **Provider not found or API key missing**:
   ```bash
   # Check available providers
   mcp-cli provider list
   
   # For cloud providers, set API keys
   mcp-cli provider set openai api_key sk-your-key
   mcp-cli provider set anthropic api_key sk-ant-your-key
   
   # Test connection
   mcp-cli provider diagnostic openai
   ```

4. **Connection issues with Ollama**:
   ```bash
   # Check Ollama is running
   ollama list
   
   # Test connection
   mcp-cli provider diagnostic ollama
   
   # Configure custom endpoint if needed
   mcp-cli provider set ollama api_base http://localhost:11434
   ```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
mcp-cli --verbose chat --server sqlite
mcp-cli --log-level DEBUG interactive --server sqlite
```

## 🔒 Security Considerations

- **Local by Default**: Ollama with gpt-oss runs locally, keeping your data private
- **API Keys**: Only needed for cloud providers (OpenAI, Anthropic, etc.), stored securely
- **File Access**: Filesystem access can be disabled with `--disable-filesystem`
- **Tool Validation**: All tool calls are validated before execution
- **Timeout Protection**: Configurable timeouts prevent hanging operations
- **Server Isolation**: Each server runs in its own process

## 🚀 Performance Features

- **Local Processing**: Default Ollama provider minimizes latency
- **Reasoning Visibility**: See AI thinking process with gpt-oss, GPT-5, Claude 4
- **Concurrent Tool Execution**: Multiple tools can run simultaneously
- **Streaming Responses**: Real-time response generation
- **Connection Pooling**: Efficient reuse of client connections
- **Caching**: Tool metadata and provider configurations are cached
- **Async Architecture**: Non-blocking operations throughout

## 📦 Dependencies

Core dependencies are organized into feature groups:

- **cli**: Terminal UI and command framework (Rich, Typer, chuk-term)
- **dev**: Development tools, testing utilities, linting
- **chuk-tool-processor**: Core tool execution and MCP communication
- **chuk-llm**: Unified LLM provider management with 200+ auto-generated functions
- **chuk-term**: Enhanced terminal UI with themes, prompts, and cross-platform support

Install with specific features:
```bash
pip install "mcp-cli[cli]"        # Basic CLI features
pip install "mcp-cli[cli,dev]"    # CLI with development tools
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/chrishayuk/mcp-cli
cd mcp-cli
pip install -e ".[cli,dev]"
pre-commit install
```

### UI Demo Scripts

Explore the UI capabilities powered by chuk-term:

```bash
# Terminal management features
uv run examples/ui_terminal_demo.py

# Output system with themes
uv run examples/ui_output_demo.py

# Streaming UI capabilities
uv run examples/ui_streaming_demo.py
```

### Running Tests

```bash
pytest
pytest --cov=mcp_cli --cov-report=html
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[CHUK Tool Processor](https://github.com/chrishayuk/chuk-tool-processor)** - Async-native tool execution
- **[CHUK-LLM](https://github.com/chrishayuk/chuk-llm)** - Unified LLM provider management with GPT-5, Claude 4, and reasoning model support
- **[CHUK-Term](https://github.com/chrishayuk/chuk-term)** - Enhanced terminal UI with themes and cross-platform support
- **[Rich](https://github.com/Textualize/rich)** - Beautiful terminal formatting
- **[Typer](https://typer.tiangolo.com/)** - CLI framework
- **[Prompt Toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit)** - Interactive input

## 🔗 Related Projects

- **[Model Context Protocol](https://modelcontextprotocol.io/)** - Core protocol specification
- **[MCP Servers](https://github.com/modelcontextprotocol/servers)** - Official MCP server implementations
- **[CHUK Tool Processor](https://github.com/chrishayuk/chuk-tool-processor)** - Tool execution engine
- **[CHUK-LLM](https://github.com/chrishayuk/chuk-llm)** - LLM provider abstraction with GPT-5, Claude 4, O3 series support
- **[CHUK-Term](https://github.com/chrishayuk/chuk-term)** - Terminal UI library with themes and cross-platform support