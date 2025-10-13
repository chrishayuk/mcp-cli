# MCP-CLI Command Patterns

## Consistent Command Structure

The MCP-CLI command system follows a consistent pattern for singular vs plural commands:

### Core Pattern

| Command Type | Purpose | Example | Result |
|-------------|---------|---------|--------|
| **Singular** | Show current status | `/model` | Shows current model with tips |
| **Plural** | List all available | `/models` | Lists all models in a table |
| **Singular + Name** | Direct action | `/model gpt-4o` | Switches to gpt-4o |
| **Singular + Subcommand** | Specific action | `/model refresh` | Refreshes model discovery |

### Commands Following This Pattern

#### Models
- `/model` - Show current model status
- `/models` - List all available models  
- `/model gpt-4o` - Switch to gpt-4o directly
- `/model refresh` - Refresh model discovery

#### Providers
- `/provider` - Show current provider status
- `/providers` - List all available providers
- `/provider ollama` - Switch to Ollama provider
- `/provider set openai api_key KEY` - Configure provider

#### Themes
- `/theme` - Show current theme with preview
- `/themes` - List all available themes
- `/theme dark` - Switch to dark theme

#### Servers
- `/server echo` - Show details about echo server
- `/servers` - List all connected MCP servers
- `/ping` - Test connectivity to all servers

## Key Features

### 1. Visual Indicators
- Commands with subcommands show `▸` indicator in help
- Example: `models ▸`, `providers ▸`, `tools ▸`

### 2. Consistent Tips
Every command output includes a `💡 Tip:` line showing:
- How to perform related actions
- Alternative commands
- Next steps

Examples:
```
💡 Tip: Use: /model <name> to switch  |  /models to list all  |  /model refresh to discover
💡 Tip: Use: /provider <name> to switch  |  /providers to list all  |  /provider set <name> for config
💡 Tip: Use: /theme <name> to switch  |  /themes to see all available themes
```

### 3. Panels for Status
Status commands use panels to display information clearly:
```
╭─────────────────────────────────────────╮
│ ℹ Current provider: openai              │
│ ℹ Current model   : gpt-4o-mini         │
│ ℹ Status          : ✅ Ready            │
│ ℹ Features        : 📡 🔧 👁️            │
╰─────────────────────────────────────────╯
```

### 4. Tables for Lists
List commands use formatted tables:
```
┌──────────┬───────────┬──────────────┐
│ Provider │ Status    │ Models       │
├──────────┼───────────┼──────────────┤
│ openai   │ ✅ Ready  │ 15 models    │
│ ollama   │ ✅ Ready  │ 51 models    │
└──────────┴───────────┴──────────────┘
```

## Design Principles

1. **Predictability**: Users can guess commands based on the pattern
2. **Discoverability**: Tips and indicators guide users to related commands
3. **Efficiency**: Direct actions without unnecessary subcommands
4. **Consistency**: Same patterns across all command groups
5. **Theme-Aware**: Uses theme colors, not hardcoded values

## Command Menu

Typing just `/` shows a menu of all available commands with descriptions and subcommand indicators.

## Examples

```bash
# Show current status
/model                  # Current model
/provider               # Current provider  
/theme                  # Current theme with preview

# List all available
/models                 # All models
/providers              # All providers
/themes                 # All themes
/servers                # All servers

# Direct actions
/model gpt-4o           # Switch model
/provider ollama        # Switch provider
/theme dark             # Switch theme
/server echo            # Show server details

# Subcommands (when needed)
/model refresh          # Refresh discovery
/tools list             # List tools
/servers ping           # Ping servers
```

This consistent pattern makes the command system intuitive and easy to learn.