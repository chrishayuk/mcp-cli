# SQLite MCP Server

## Overview

The SQLite MCP server provides tools for interacting with SQLite databases. It allows you to query, modify, and manage SQLite database files through the Model Context Protocol.

**Transport:** stdio (local process)

## What it Does

This server enables:
- Querying SQLite databases using SQL
- Creating and modifying database schemas
- Reading and writing data to database tables
- Executing arbitrary SQL commands
- Analyzing database structure and contents

## Tools

The SQLite server provides the following tools:

### `query`
Execute a SELECT query against the SQLite database.

**Parameters:**
- `sql` (string, required): The SQL SELECT query to execute

**Returns:** Query results as structured data

### `execute`
Execute INSERT, UPDATE, DELETE, or other SQL statements.

**Parameters:**
- `sql` (string, required): The SQL statement to execute

**Returns:** Execution status and affected row count

### `list_tables`
List all tables in the database.

**Returns:** Array of table names

### `describe_table`
Get the schema information for a specific table.

**Parameters:**
- `table_name` (string, required): Name of the table to describe

**Returns:** Column definitions and schema information

### `create_table`
Create a new table in the database.

**Parameters:**
- `table_name` (string, required): Name of the new table
- `schema` (string, required): SQL column definitions

**Returns:** Confirmation of table creation

## Configuration

### Required Parameters

- `--db-path`: Path to the SQLite database file (required)

### Example Configuration

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "/path/to/your/database.db"]
    }
  }
}
```

### Tokens

**Default Token Names:** None

**Authentication Type:** None (Local file access)

**OAuth Support:** No

No authentication tokens are required for the SQLite server. Access is controlled by file system permissions on the database file.

## Usage Notes

- The database file will be created if it doesn't exist
- Ensure the application has read/write permissions to the database file
- Use absolute paths for better reliability
- The server runs locally via `uvx` (no network connection required)
