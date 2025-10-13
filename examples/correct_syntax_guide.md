# Correct Syntax for Tool Execution in Interactive Mode

## The Issue
When you type:
```
/exec echo_text "hello world"
```

The system tries to parse `"hello world"` as JSON, but it's just a plain string, not a JSON object with the required `message` field.

## Correct Syntax Options

### Option 1: Full JSON Format (Recommended)
```bash
/exec echo_text '{"message": "hello world"}'
```

### Option 2: Without Slash
```bash
exec echo_text '{"message": "hello world"}'
```

### Option 3: Using the run alias
```bash
run echo_text '{"message": "hello world"}'
```

## Common Patterns

### Simple Message
```bash
/exec echo_text '{"message": "Hello, World!"}'
```

### With Optional Parameters
```bash
/exec echo_text '{"message": "Test", "prefix": "[INFO] ", "suffix": " [END]"}'
```

### Database Query (SQLite)
```bash
/exec read_query '{"sql": "SELECT * FROM users LIMIT 5"}'
```

### List Tables
```bash
/exec list_tables '{}'
```

## Important Notes

1. **Always use JSON format**: Tools expect parameters as JSON objects
2. **Use single quotes**: Wrap JSON in single quotes to avoid shell escaping issues
3. **Empty object for no params**: Use `'{}'` for tools with no required parameters
4. **Check parameters first**: Use `/execute <tool_name>` to see what parameters are required

## Quick Reference

| What You Want | What to Type |
|--------------|--------------|
| Echo a message | `/exec echo_text '{"message": "your message"}'` |
| List database tables | `/exec list_tables '{}'` |
| Run a SQL query | `/exec read_query '{"sql": "SELECT * FROM table"}'` |
| Get service info | `/exec get_service_info '{}'` |

## Troubleshooting

### Error: "Invalid parameter 'message': expected string, got NoneType"
**Cause**: The JSON doesn't include the required `message` field
**Fix**: Use `'{"message": "your text"}'` not just `"your text"`

### Error: "Expecting property name enclosed in double quotes"
**Cause**: Invalid JSON format
**Fix**: Ensure proper JSON syntax with double quotes inside, single quotes outside

### To see what parameters a tool needs:
```bash
/execute echo_text
```
This will show you the required and optional parameters.

## Examples for Your Session

Since you have both sqlite and echo servers connected, here are some examples:

```bash
# Echo server tools
/exec echo_text '{"message": "Hello from echo server!"}'
/exec echo_uppercase '{"text": "make me uppercase"}'
/exec echo_reverse '{"text": "reverse me"}'
/exec get_service_info '{}'

# SQLite server tools
/exec list_tables '{}'
/exec describe_table '{"table_name": "users"}'
/exec read_query '{"sql": "SELECT COUNT(*) FROM users"}'
/exec write_query '{"sql": "INSERT INTO logs (message) VALUES (?)", "params": ["Test log"]}'
```