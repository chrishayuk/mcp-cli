# Echo MCP Server

## Overview

The Echo MCP server is a simple testing and demonstration server that echoes back the input it receives. It's primarily used for testing MCP integrations and verifying server connectivity.

**Transport:** stdio (local process)

## What it Does

This server enables:
- Testing MCP server connectivity
- Verifying stdio-based server communication
- Demonstrating basic MCP tool usage
- Debugging MCP client implementations

## Tools

The Echo server provides the following tools:

### `echo`
Returns the exact input message that was provided.

**Parameters:**
- `message` (string, required): The message to echo back

**Returns:** The same message that was input

### `echo_json`
Echoes back JSON data in structured format.

**Parameters:**
- `data` (object, required): JSON object to echo back

**Returns:** The same JSON object that was input

### `ping`
Simple health check endpoint that returns "pong".

**Returns:** The string "pong"

## Configuration

### Required Parameters

- `stdio`: Communication mode (required)

### Example Configuration

```json
{
  "mcpServers": {
    "echo": {
      "command": "uvx",
      "args": ["chuk-mcp-echo", "stdio"]
    }
  }
}
```

### Tokens

**Default Token Names:** None

**Authentication Type:** None (Local testing tool)

**OAuth Support:** No

No authentication tokens are required for the Echo server. It's a local testing tool with no external dependencies.

## Usage Notes

- This server is primarily for testing and development
- Runs locally via `uvx` (no network connection required)
- Useful for verifying that your MCP client is working correctly
- Can be used to test tool calling mechanisms before integrating more complex servers
