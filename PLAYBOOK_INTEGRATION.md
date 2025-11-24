# Playbook Integration in MCP CLI

## Overview

The playbook integration provides mcp-cli with access to a curated repository of procedural knowledge - step-by-step guides for common tasks. When enabled, the LLM can query playbooks to retrieve tested, reliable procedures for "how to" questions.

## Architecture

### Components

1. **PlaybookPreferences** (`src/mcp_cli/utils/preferences.py:154-160`)
   - Manages playbook configuration
   - Stored in `~/.mcp-cli/preferences.json`

2. **PlaybookService** (`src/mcp_cli/services/playbook_service.py`)
   - Clean interface to playbook MCP server
   - Methods: `query_playbook()`, `list_playbooks()`, `get_playbook()`, `submit_playbook()`

3. **Conditional Server Binding** (`src/mcp_cli/config/cli_options.py:103-113`)
   - Playbook server only bound when `playbook.enabled = True`
   - Otherwise completely skipped

4. **System Prompt Enhancement** (`src/mcp_cli/chat/system_prompt.py:15-39`)
   - When enabled, LLM receives guidance to check playbooks first
   - Provides clear instructions on when/how to use playbooks

5. **Slash Command** (`src/mcp_cli/commands/definitions/playbook.py`)
   - `/playbook` command for chat mode
   - Enable/disable, query, list, get playbooks

## Configuration

### Preferences Model

```python
class PlaybookPreferences(BaseModel):
    enabled: bool = True                          # Enable/disable feature
    mcp_server_name: str = "playbook-repo"        # MCP server name
    local_playbook_dir: str | None = None         # Local playbooks directory
    top_k: int = 3                                # Number of results to consider
```

### Location

Stored in `~/.mcp-cli/preferences.json`:

```json
{
  "playbook": {
    "enabled": true,
    "mcp_server_name": "playbook-repo",
    "local_playbook_dir": null,
    "top_k": 3
  }
}
```

### MCP Server Configuration

The playbook server is defined in `src/mcp_cli/server_config.json`:

```json
{
  "mcpServers": {
    "playbook-repo": {
      "command": "uvx",
      "args": ["chuk-mcp-playbook"]
    }
  }
}
```

**Important:** The server is only bound if `playbook.enabled = True` in preferences.

## Usage

### CLI Flags

Enable or disable playbook for a session:

```bash
# Enable playbook
mcp-cli --playbook

# Disable playbook
mcp-cli --no-playbook

# Setting persists in preferences
```

### Slash Commands (Chat Mode)

**Management Commands:**

```bash
# Show current playbook status
/playbook
/playbook status

# Enable/disable (requires restart)
/playbook enable
/playbook disable

# List all available playbooks
/playbook list
```

**Querying Playbooks:**

Don't use slash commands for querying! Just ask the LLM naturally:

```bash
# ✅ Correct - Ask the LLM
> How do I get sunset times for London?

# ❌ Wrong - Don't use slash command
> /playbook query how to get sunset times

# The LLM will automatically use the query_playbook tool when appropriate
```

### PreferenceManager Methods

```python
from mcp_cli.utils.preferences import get_preference_manager

pref_manager = get_preference_manager()

# Check if enabled
enabled = pref_manager.is_playbook_enabled()

# Enable/disable
pref_manager.set_playbook_enabled(True)
pref_manager.set_playbook_enabled(False)

# Get server name
server = pref_manager.get_playbook_server_name()

# Set local playbook directory
pref_manager.set_local_playbook_dir("/path/to/playbooks")

# Adjust top_k
pref_manager.set_playbook_top_k(5)
```

### PlaybookService API

```python
from mcp_cli.services import PlaybookService

# Initialize with tool manager
playbook_service = PlaybookService(tool_manager)

# Query playbooks
result = await playbook_service.query_playbook(
    "How do I get sunset times?",
    top_k=3
)

# List all playbooks
playbooks = await playbook_service.list_playbooks()

# Get specific playbook
playbook = await playbook_service.get_playbook("Get Sunset and Sunrise Times")

# Submit a new playbook
success = await playbook_service.submit_playbook(
    title="My Custom Playbook",
    content="# Steps\n1. First step\n2. Second step",
    description="A custom procedure",
    tags=["custom", "workflow"]
)
```

## How It Works

### 1. Server Binding

When mcp-cli starts:

```python
# In cli_options.py:extract_server_names()
playbook_server_name = pref_manager.get_playbook_server_name()

if server_name == playbook_server_name:
    if pref_manager.is_playbook_enabled():
        enabled_servers.append(server_name)  # ✅ Bind the server
    else:
        # ❌ Skip the server entirely
        pass
```

### 2. System Prompt Enhancement

When playbook is enabled, the LLM receives this guidance:

```markdown
**PLAYBOOK INTEGRATION:**

You have access to a playbook repository that contains step-by-step guides for common tasks.

- **When to use playbooks:**
  - User asks "how to" or "how do I" questions
  - User requests guidance on multi-step procedures
  - User needs help with workflows or processes

- **How to use playbooks:**
  - Use the `query_playbook` tool with the user's question
  - The playbook will return detailed step-by-step instructions
  - Follow the playbook's guidance to execute the task
  - You can also use `list_playbooks` to see available playbooks

- **Remember:**
  - Always check playbooks FIRST for "how to" questions
  - Playbooks provide tested, reliable procedures
  - Follow playbook steps in order for best results
```

### 3. Query Flow

```
User: "How do I get sunset times for London?"
         ↓
LLM sees playbook guidance in system prompt
         ↓
LLM calls: query_playbook(question="get sunset times")
         ↓
PlaybookService → Tool Manager → playbook-repo MCP server
         ↓
Returns: Full markdown playbook with steps
         ↓
LLM follows playbook steps:
  1. Call get_location("London")
  2. Call get_sunrise_sunset(coordinates)
  3. Format and return result
```

## Default Playbooks

The `chuk-mcp-playbook` server comes with 13 default playbooks:

**Weather & Forecast:**
- Get Current Weather
- Get Sunset and Sunrise Times
- Get Weather Forecast
- Compare Historical Weather
- Compare Recurring Date Weather

**Safety & Conditions:**
- Check Air Quality Safety
- Check Beach Swimming Safety
- Check Fishing Conditions
- Check Sailing/Boating Safety
- Check Surfing Conditions
- Check UV/Sun Safety
- Get Tide Predictions

**Planning:**
- Plan Multi-day Trip Weather

## Advanced Features

### Local Playbooks

You can configure a local directory for agent-specific playbooks:

```python
pref_manager.set_local_playbook_dir("/path/to/my/playbooks")
```

This allows you to add custom playbooks without submitting them to the shared repository.

### Submitting Playbooks

Users can submit new playbooks via the service:

```python
await playbook_service.submit_playbook(
    title="Deploy to Production",
    content="""
# Playbook: Deploy to Production

## Steps
1. Run tests: `npm test`
2. Build: `npm run build`
3. Deploy: `git push heroku main`
4. Verify: Check logs
""",
    description="Standard production deployment procedure",
    tags=["deployment", "production"]
)
```

### Adjusting Results

Control how many playbooks are considered:

```bash
# Via slash command
/playbook query <question> --top-k 5

# Via preferences
pref_manager.set_playbook_top_k(5)

# Via API
result = await playbook_service.query_playbook(question, top_k=5)
```

## Benefits

1. **Consistent Procedures**: Playbooks provide tested, reliable workflows
2. **Knowledge Sharing**: Team can share best practices via playbooks
3. **Faster Execution**: LLM doesn't have to figure out multi-step procedures
4. **Reduced Errors**: Following playbook steps reduces trial-and-error
5. **Discoverability**: `/playbook list` shows what procedures are available

## Troubleshooting

### Playbook server not binding

```bash
# Check if playbook is enabled
/playbook status

# Enable it
/playbook enable

# Restart session (required for server binding changes)
exit
mcp-cli
```

### No playbooks found

```bash
# Verify server is configured
cat src/mcp_cli/server_config.json | grep playbook-repo

# Check server is installed
uvx chuk-mcp-playbook --help

# Test server manually
uvx chuk-mcp-playbook
```

### Playbook queries not working

1. Ensure playbook is enabled: `/playbook status`
2. Check server is bound: `/servers` (should show playbook-repo)
3. Verify tools are available: `/tools` (should show query_playbook)

## Implementation Details

### Files Changed

1. `src/mcp_cli/utils/preferences.py` - Added PlaybookPreferences model + methods
2. `src/mcp_cli/config/cli_options.py` - Conditional server binding logic
3. `src/mcp_cli/chat/system_prompt.py` - System prompt enhancement
4. `src/mcp_cli/services/playbook_service.py` - PlaybookService implementation
5. `src/mcp_cli/commands/definitions/playbook.py` - Slash command
6. `src/mcp_cli/commands/actions/playbook.py` - Command actions
7. `src/mcp_cli/commands/models/playbook.py` - Pydantic models
8. `src/mcp_cli/main.py` - CLI flags
9. `src/mcp_cli/server_config.json` - Server configuration

### Design Principles

1. **No Hard-coded Heuristics**: No pattern matching for "how to" questions
2. **Simple Enable/Disable**: Boolean flag, not complex configuration
3. **Conditional Binding**: Server only exists when enabled
4. **System Prompt Driven**: LLM learns to use playbooks via prompt
5. **Clean Service Layer**: Simple API for playbook operations

## Future Enhancements

- [ ] Local playbook directory scanning
- [ ] Playbook versioning
- [ ] Playbook templates
- [ ] Team playbook sharing
- [ ] Playbook execution tracking
- [ ] Playbook success metrics
