# Model Context Provider CLI
This repository contains a protocol-level CLI designed to interact with a Model Context Provider server. The client allows users to send commands, query data, and interact with various resources provided by the server.

## Features
- Protocol-level communication with the Model Context Provider.
- Dynamic tool and resource exploration.
- Support for multiple providers and models:
  - Providers: OpenAI, Ollama.
  - Default models: `gpt-4o-mini` for OpenAI, `qwen2.5-coder` for Ollama.
- Enhanced modular chat system with server-aware tools.
- Rich command system with context-aware completions.
- Two operational modes:
  - **Interactive Mode**: Command-line interface with slash commands
  - **Chat Mode**: Conversational interface with LLM

## Prerequisites
- Python 3.8 or higher.
- Required dependencies (see [Installation](#installation))
- If using ollama you should have ollama installed and running.
- If using openai you should have an api key set in your environment variables (OPENAI_API_KEY=yourkey)

## Installation
1. Clone the repository:

```bash
git clone https://github.com/chrishayuk/mcp-cli
cd mcp-cli
```

2. Install UV:

```bash
pip install uv
```

3. Resynchronize dependencies:

```bash
uv sync --reinstall
```

## Usage
To start the client and interact with the SQLite server, you can use either chat mode or interactive mode:

### Chat Mode
```bash
uv run mcp-cli chat --server sqlite
```

### Interactive Mode
```bash
uv run mcp-cli interactive --server sqlite
```

### Command-line Arguments
- `--server`: Specifies the server configuration to use. Required.
- `--config-file`: (Optional) Path to the JSON configuration file. Defaults to `server_config.json`.
- `--provider`: (Optional) Specifies the provider to use (`openai` or `ollama`). Defaults to `openai`.
- `--model`: (Optional) Specifies the model to use. Defaults depend on the provider:
  - `gpt-4o-mini` for OpenAI.
  - `llama3.2` for Ollama.

### Examples
Run in chat mode with the default OpenAI provider and model:

```bash
uv run mcp-cli chat --server sqlite
```

Run in interactive mode with a specific provider and model:

```bash
uv run mcp-cli interactive --server sqlite --provider ollama --model llama3.2
```

## Interactive Mode
Interactive mode provides a command-line interface with slash commands:

```bash
uv run mcp-cli interactive --server sqlite
```

### Interactive Commands
In interactive mode, you can use the following slash commands:

- `/ping`: Check if server is responsive
- `/prompts`: List available prompts
- `/tools`: List available tools
- `/tools-all`: Show detailed tool information with parameters
- `/tools-raw`: Show raw tool definitions in JSON
- `/resources`: List available resources
- `/chat`: Enter chat mode
- `/cls`: Clear the screen
- `/clear`: Clear the screen and show welcome message
- `/help`: Show this help message
- `/exit` or `/quit`: Exit the program

You can also exit by typing `exit` or `quit` without the slash prefix.

## Chat Mode
Chat mode provides a conversational interface with the LLM:

```bash
uv run mcp-cli chat --server sqlite
```

In chat mode, you can interact with the model in natural language, and it will automatically use the available tools when needed. You can also use the following slash commands:

- `/tools`: Display all available tools with their server information
  - `/tools --all`: Show detailed tool information including parameters
  - `/tools --raw`: Show raw tool definitions
- `/toolhistory` or `/th`: Show history of tool calls in the current session
  - `/th -n 5`: Show only the last 5 tool calls
  - `/th --json`: Show tool calls in JSON format
- `/cls`: Clear the screen while keeping conversation history
- `/clear`: Clear both the screen and conversation history
- `/compact`: Condense conversation history into a summary
- `/save <filename>`: Save conversation history to a JSON file
- `/help`: Show available commands
- `/help <command>`: Show detailed help for a specific command
- `/quickhelp` or `/qh`: Display a quick reference of common commands
- `/interrupt`, `/stop`, or `/cancel`: Interrupt running tool execution

Type `exit` or `quit` to leave chat mode.

### Using OpenAI Provider:
If you wish to use openai models, you should:

- Set the `OPENAI_API_KEY` environment variable before running the client, either in .env or as an environment variable.

## Project Structure

The project follows a modular architecture:

```
src/
├── cli/
│   ├── chat/
│   │   ├── commands/         # Chat slash commands
│   │   ├── chat_context.py   # Chat state management
│   │   ├── chat_handler.py   # Main chat logic
│   │   ├── conversation.py   # Conversation processing
│   │   ├── ui_helpers.py     # UI utilities
│   │   └── ui_manager.py     # User interface management
│   ├── commands/             # Main CLI commands (including interactive mode)
│   └── ...
├── llm/                      # LLM client and tools
└── ...
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your proposed changes.

## License
This project is licensed under the [MIT License](license.md).