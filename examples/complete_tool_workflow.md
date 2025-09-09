# Complete Tool Execution Workflow - Interactive Mode

This guide demonstrates the complete workflow for executing MCP tools in interactive mode, showing both slash and non-slash commands.

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    TOOL EXECUTION COMMANDS                   │
├─────────────────────────────────────────────────────────────┤
│ List Tools:      execute         OR  /execute                │
│ Tool Details:    execute <tool>  OR  /execute <tool>         │
│ Run Tool:        execute <tool> '<json>'                     │
│                  OR /execute <tool> '<json>'                  │
│                                                               │
│ Aliases:         exec, run  (work with or without /)         │
│                  /exec <tool> '<json>'                        │
│                  run <tool> '<json>'                          │
└─────────────────────────────────────────────────────────────┘
```

## Complete Workflow Example

### Step 1: Connect to Servers

```bash
$ mcp-cli interactive --server echo --server sqlite --server filesystem
```

### Step 2: Explore Available Tools

```
> /execute
Available Tools

Server: echo
  • echo: Echoes back the provided message

Server: sqlite
  • list_tables: List all database tables
  • query: Execute SQL query
  • describe_table: Get table schema
  • insert: Insert data into table
  • update: Update table records
  • delete: Delete table records

Server: filesystem
  • read_file: Read file contents
  • write_file: Write content to file
  • list_directory: List directory contents
  • create_directory: Create new directory
  • delete_file: Delete a file

Use 'execute <tool_name>' to see tool parameters
```

### Step 3: Investigate Tool Parameters

```
> /execute query
Tool: query
Description: Execute SQL query

Parameters:
  • sql* (string): SQL query to execute
  • params (array): Query parameters for prepared statements
  • limit (number): Maximum rows to return (default: 100)
  
  * = required parameter

Example Usage:
  execute query '{"sql": "SELECT * FROM users"}'
  execute query '{"sql": "SELECT * FROM users WHERE age > ?", "params": [18]}'
```

### Step 4: Execute Tools

#### Simple Tool Execution
```
> /execute echo '{"message": "Testing the echo server!"}'
Executing tool: echo
✓ Tool executed successfully
{
  "echoed": "Testing the echo server!"
}
```

#### Database Query
```
> /exec list_tables '{}'
Executing tool: list_tables
✓ Tool executed successfully
{
  "tables": ["users", "posts", "comments", "tags"]
}

> /execute query '{"sql": "SELECT COUNT(*) as total FROM users"}'
Executing tool: query
✓ Tool executed successfully
{
  "rows": [{"total": 42}],
  "columns": ["total"]
}
```

#### File Operations
```
> /run read_file '{"path": "./README.md"}'
Executing tool: read_file
✓ Tool executed successfully
{
  "content": "# My Project\n\nThis is the readme content...",
  "size": 1024,
  "encoding": "utf-8"
}
```

### Step 5: Complex Tool Execution

#### With Multiple Parameters
```
> /execute query '{
    "sql": "SELECT * FROM users WHERE age > ? AND city = ?",
    "params": [21, "San Francisco"],
    "limit": 10
  }'
Executing tool: query
✓ Tool executed successfully
{
  "rows": [
    {"id": 1, "name": "Alice", "age": 25, "city": "San Francisco"},
    {"id": 7, "name": "Bob", "age": 30, "city": "San Francisco"}
  ],
  "columns": ["id", "name", "age", "city"]
}
```

#### Chaining Operations
```
> /execute describe_table '{"table": "users"}'
Executing tool: describe_table
✓ Tool executed successfully
{
  "columns": [
    {"name": "id", "type": "INTEGER", "primary_key": true},
    {"name": "name", "type": "TEXT", "nullable": false},
    {"name": "email", "type": "TEXT", "unique": true},
    {"name": "created_at", "type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"}
  ]
}

> /execute insert '{
    "table": "users",
    "data": {
      "name": "Charlie",
      "email": "charlie@example.com"
    }
  }'
Executing tool: insert
✓ Tool executed successfully
{
  "inserted_id": 43,
  "rows_affected": 1
}
```

## Command Variations

All these variations work identically:

### With Slash Prefix
```
/execute                           # List tools
/execute echo                      # Show echo tool
/execute echo '{"message": "Hi"}' # Run echo tool
/exec echo '{"message": "Hi"}'    # Using exec alias
/run echo '{"message": "Hi"}'     # Using run alias
```

### Without Slash Prefix
```
execute                            # List tools
execute echo                       # Show echo tool
execute echo '{"message": "Hi"}'  # Run echo tool
exec echo '{"message": "Hi"}'     # Using exec alias
run echo '{"message": "Hi"}'      # Using run alias
```

## Advanced Tips

### 1. JSON Formatting Tips

#### Single Line (Recommended for simple params)
```
> /execute echo '{"message": "Hello World"}'
```

#### Multi-line (For complex params)
```
> /execute query '{
    "sql": "SELECT * FROM users",
    "limit": 10
  }'
```

### 2. Using Shell Features

#### Command History
```
> /execute echo '{"message": "First"}'
> ↑ (arrow up to repeat/edit)
```

#### Tab Completion
```
> /exe<TAB>  →  /execute
> execute e<TAB>  →  execute echo
```

### 3. Scripting

Create a file `commands.txt`:
```
/execute
/execute echo '{"message": "Automated message 1"}'
/exec echo '{"message": "Automated message 2"}'
run echo '{"message": "Automated message 3"}'
exit
```

Run it:
```bash
$ mcp-cli interactive --server echo < commands.txt
```

### 4. Piping Results

```bash
$ echo '/execute echo {"message": "Piped!"}' | mcp-cli interactive --server echo
```

## Error Handling

### Missing Required Parameters
```
> /execute echo '{}'
✗ Tool execution failed: Missing required parameter: message
```

### Invalid JSON
```
> /execute echo '{"message": invalid}'
✗ Invalid parameters: Expecting value: line 1 column 13
Expected JSON or key=value pairs
```

### Tool Not Found
```
> /execute nonexistent
✗ Tool not found: nonexistent
[Lists available tools]
```

## Best Practices

1. **Use Single Quotes**: Always wrap JSON in single quotes to avoid shell escaping
2. **Check Parameters First**: Use `execute <tool>` to see requirements
3. **Start Simple**: Test with simple tools like echo first
4. **Use Aliases**: `exec` and `run` are faster to type
5. **Slash Optional**: Use slash for muscle memory from chat mode, omit for speed

## Summary

Interactive mode with tool execution provides:

✅ **Direct Control** - Execute exactly what you want  
✅ **Full Visibility** - See all parameters and options  
✅ **Immediate Feedback** - Results shown instantly  
✅ **Flexible Syntax** - Slash commands optional  
✅ **Power User Features** - Aliases, scripting, piping  
✅ **Seamless Integration** - Works with all MCP servers  

Whether you use `/execute`, `execute`, `/exec`, or `run`, the unified command system ensures consistent behavior across all modes of MCP-CLI.