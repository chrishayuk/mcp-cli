# Interactive Mode - Tool Execution Session Demo

This demonstrates a real interactive session showing tool execution with both regular and slash commands.

## Starting an Interactive Session

```bash
$ mcp-cli interactive --server echo --server sqlite
```

## Example Session Transcript

```
MCP Interactive Mode
Type 'help' to see available commands.
Type 'exit' or 'quit' to exit.
Type '/' to bring up the slash-menu.

> help
Available commands:
  clear         Clear the terminal screen
  execute       Execute a tool with parameters (aliases: exec, run)
  exit          Exit the application (aliases: quit, q, bye)
  help          Show help information
  servers       List connected MCP servers
  tools         List available tools
  ...

> /execute
Available Tools

Server: echo
  • echo: Echoes back the provided message

Server: sqlite  
  • list_tables: List all database tables
  • query: Execute SQL query
  • describe_table: Get table schema

Use 'execute <tool_name>' to see tool parameters
Use 'execute <tool_name> <params>' to run a tool

> /execute echo
Tool: echo
Description: Echoes back the provided message

Parameters:
  • message* (string): The message to echo back
  * = required parameter

Example Usage:
  execute echo '{"message": "<message>"}'

> /execute echo '{"message": "Testing slash command!"}'
Executing tool: echo
✓ Tool executed successfully
{
  "echoed": "Testing slash command!"
}

> execute echo '{"message": "Testing without slash!"}'
Executing tool: echo
✓ Tool executed successfully
{
  "echoed": "Testing without slash!"
}

> /exec echo '{"message": "Using /exec alias!"}'
Executing tool: echo
✓ Tool executed successfully
{
  "echoed": "Using /exec alias!"
}

> run echo '{"message": "Using run alias without slash!"}'
Executing tool: echo
✓ Tool executed successfully
{
  "echoed": "Using run alias without slash!"
}

> /execute list_tables
Tool: list_tables
Description: List all database tables

Parameters:
  (No parameters required)

Example Usage:
  execute list_tables

> /execute list_tables '{}'
Executing tool: list_tables
✓ Tool executed successfully
{
  "tables": [
    "users",
    "posts",
    "comments"
  ]
}

> /exec query '{"sql": "SELECT COUNT(*) as total FROM users"}'
Executing tool: query
✓ Tool executed successfully
{
  "rows": [
    {"total": 42}
  ],
  "columns": ["total"]
}

> exit
Goodbye!
```

## Command Equivalence Table

| With Slash | Without Slash | Description |
|------------|---------------|-------------|
| `/execute` | `execute` | List all tools |
| `/execute echo` | `execute echo` | Show tool details |
| `/execute echo '{...}'` | `execute echo '{...}'` | Execute tool |
| `/exec echo '{...}'` | `exec echo '{...}'` | Execute with alias |
| `/run query '{...}'` | `run query '{...}'` | Execute with alias |
| `/help` | `help` | Show help |
| `/help execute` | `help execute` | Show command help |
| `/servers` | `servers` | List servers |
| `/tools` | `tools` | List tools |
| `/exit` | `exit` | Exit interactive mode |

## Key Features Demonstrated

### 1. **Slash Command Support**
- All commands work with `/` prefix
- Provides consistency with chat mode
- Familiar for users coming from chat

### 2. **Command Aliases**
- `execute`, `exec`, `run` all work
- Work with or without slash
- `/exec` and `exec` are equivalent

### 3. **Tool Execution Flow**
1. List tools: `/execute` or `execute`
2. Show details: `/execute <tool>` or `execute <tool>`
3. Run tool: `/execute <tool> '<json>'` or `execute <tool> '<json>'`

### 4. **JSON Parameter Handling**
- Use single quotes around JSON: `'{"key": "value"}'`
- Empty params for no-argument tools: `'{}'`
- Complex nested JSON supported

### 5. **Result Display**
- Formatted JSON output
- Clear success/error messages
- Readable structure

## Tips for Users

1. **Slash is Optional**: In interactive mode, `/` prefix is completely optional
2. **Tab Completion**: Type partial commands and press Tab
3. **Quick Testing**: Use `exec` or `run` aliases for faster typing
4. **JSON Tips**: Always use single quotes around JSON to avoid escaping
5. **Help Available**: Use `help <command>` for detailed command info

## Comparison: Chat Mode vs Interactive Mode

### Chat Mode
```
User: Can you echo "Hello World"?
Assistant: I'll echo that message for you.
[Executes echo tool]
Result: "Hello World"
```

### Interactive Mode
```
> /execute echo '{"message": "Hello World"}'
Executing tool: echo
✓ Tool executed successfully
{
  "echoed": "Hello World"
}
```

## Benefits of Interactive Mode Tool Execution

1. **Direct Control**: Execute exactly the tool you want
2. **Parameter Visibility**: See all required/optional parameters
3. **Immediate Feedback**: Results shown instantly
4. **No Ambiguity**: Explicit tool and parameter specification
5. **Scriptable**: Can be automated in scripts
6. **Slash Compatibility**: Familiar syntax from chat mode

## Advanced Examples

### Chaining Commands
```bash
> /execute list_tables '{}'
> /execute describe_table '{"table": "users"}'
> /execute query '{"sql": "SELECT * FROM users LIMIT 1"}'
```

### Using with Scripts
```bash
echo '/execute echo {"message": "Automated!"}' | mcp-cli interactive --server echo
```

### Batch Execution
```bash
cat commands.txt | mcp-cli interactive --server echo
```
Where `commands.txt` contains:
```
/execute
/execute echo '{"message": "First command"}'
/exec echo '{"message": "Second command"}'
/run echo '{"message": "Third command"}'
exit
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Command not found" | Check spelling, use `help` to see available commands |
| "Tool not found" | Use `/execute` to list available tools |
| "Invalid JSON" | Ensure proper quotes: `'{"key": "value"}'` |
| "Missing parameters" | Use `/execute <tool>` to see required params |
| "No tools available" | Check server connection with `/servers` |

## Summary

Interactive mode with slash commands provides:
- **Flexibility**: Use with or without `/` prefix
- **Familiarity**: Same commands as chat mode
- **Power**: Direct tool execution with full control
- **Convenience**: Aliases and shortcuts available
- **Compatibility**: Works seamlessly with existing workflows