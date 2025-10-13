# Interactive Tool Execution Demo

This guide shows how to use the new `execute` command in MCP-CLI's interactive mode to run MCP tools directly.

## Starting Interactive Mode

```bash
# Connect to a server (e.g., echo server)
mcp-cli interactive --server echo

# Or connect to multiple servers
mcp-cli interactive --server echo --server sqlite
```

## Using the Execute Command

### 1. List Available Tools

```
> execute
```

This shows all tools available from connected servers:
- Tool names and descriptions
- Grouped by server

### 2. Show Tool Details

```
> execute <tool_name>
```

Example:
```
> execute echo
```

This displays:
- Tool description
- Required and optional parameters
- Parameter types and descriptions
- Usage examples

### 3. Execute a Tool

```
> execute <tool_name> '<json_parameters>'
```

Examples:

#### Echo Tool
```
> execute echo '{"message": "Hello, World!"}'
```

#### Database Query (if sqlite server connected)
```
> execute query '{"sql": "SELECT * FROM users LIMIT 5"}'
```

#### List Tables
```
> execute list_tables
```

### 4. Using Command Aliases

The execute command has convenient aliases:

```
> exec echo '{"message": "Using exec alias"}'
> run list_tables
```

## Advanced Examples

### Tool with Multiple Parameters
```
> execute calculate '{"operation": "multiply", "a": 10, "b": 5}'
```

### Tool with Optional Parameters
```
> execute search '{"query": "test", "limit": 10, "offset": 0}'
```

### Tool without Parameters
```
> execute get_status
# or with empty JSON
> execute get_status '{}'
```

## Tips

1. **Tab Completion**: Type `exe<tab>` to complete to `execute`

2. **JSON Formatting**: Use single quotes around JSON to avoid shell escaping issues:
   - Good: `'{"key": "value"}'`
   - Problematic: `"{\"key\": \"value\"}"`

3. **Viewing Results**: Results are displayed as formatted JSON for easy reading

4. **Error Handling**: If parameters are incorrect, you'll see helpful error messages

## Real-World Example Session

```bash
$ mcp-cli interactive --server echo --server sqlite

> execute
# Lists all available tools

> execute echo
# Shows echo tool parameters

> execute echo '{"message": "Testing tool execution"}'
# Executes echo tool and shows result

> exec list_tables
# Lists database tables using alias

> execute query '{"sql": "SELECT COUNT(*) FROM users"}'
# Runs SQL query

> exit
```

## Benefits

- **Direct Tool Access**: No need to go through chat mode
- **Parameter Visibility**: See exactly what parameters each tool needs
- **Quick Testing**: Test MCP tools rapidly during development
- **Scriptable**: Can be used in scripts for automation

## Comparison with Other Modes

| Feature | Interactive Mode | Chat Mode | CLI Mode |
|---------|-----------------|-----------|----------|
| Tool Execution | `execute <tool> <params>` | Natural language | `mcp-cli cmd --tool <tool>` |
| Parameter Display | Yes | No | No |
| Interactive | Yes | Yes | No |
| Scriptable | Yes | No | Yes |
| Best For | Testing & Development | End users | Automation |

## Troubleshooting

- **"Tool not found"**: Check tool name with `execute` command
- **"Invalid parameters"**: View required params with `execute <tool_name>`
- **"No tools available"**: Ensure server is connected properly

## Next Steps

1. Try connecting to different MCP servers
2. Explore available tools with `execute`
3. Test tool execution with various parameters
4. Use in scripts for automation