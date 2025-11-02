# Notion MCP Server

## Overview

The Notion MCP server provides integration with Notion's workspace, allowing you to read, write, and manage Notion pages, databases, and content through the Model Context Protocol.

**Transport:** Streamable HTTP

## What it Does

This server enables:
- Reading and writing Notion pages
- Querying Notion databases
- Creating and updating database entries
- Searching across your Notion workspace
- Managing page properties and metadata
- Working with blocks and content within pages

## Tools

The Notion server provides the following tools:

### `search`
Search across your Notion workspace for pages and databases.

**Parameters:**
- `query` (string, required): Search query text
- `filter` (string, optional): Filter by "page" or "database"

**Returns:** List of matching pages/databases with metadata

### `get_page`
Retrieve the contents of a specific Notion page.

**Parameters:**
- `page_id` (string, required): The ID of the page to retrieve

**Returns:** Page content including all blocks and properties

### `create_page`
Create a new page in Notion.

**Parameters:**
- `parent_id` (string, required): ID of the parent page or database
- `title` (string, required): Title of the new page
- `content` (array, optional): Blocks to include in the page

**Returns:** The newly created page object

### `update_page`
Update an existing Notion page.

**Parameters:**
- `page_id` (string, required): ID of the page to update
- `properties` (object, optional): Properties to update
- `content` (array, optional): New blocks to append

**Returns:** The updated page object

### `query_database`
Query a Notion database with filters and sorting.

**Parameters:**
- `database_id` (string, required): ID of the database to query
- `filter` (object, optional): Filter criteria
- `sorts` (array, optional): Sort criteria

**Returns:** Database entries matching the query

### `create_database_entry`
Create a new entry in a Notion database.

**Parameters:**
- `database_id` (string, required): ID of the database
- `properties` (object, required): Property values for the entry

**Returns:** The newly created database entry

## Configuration

### Required Tokens

The Notion server requires authentication via Notion's API. You need to:

1. Create a Notion integration at https://www.notion.so/my-integrations
2. Grant the integration access to the workspaces/pages you want to access
3. Copy the integration token (starts with `secret_`)

### Token Configuration

**Default Token Name:** `NOTION_TOKEN`

**Authentication Type:** API Token (Integration Token)

**OAuth Support:** Yes - Notion supports OAuth 2.0 for user authorization. See [OAUTH.md](../OAUTH.md) for OAuth setup details.

Set the token using one of these methods:

**Environment Variable:**
```bash
export NOTION_TOKEN="secret_your_token_here"
```

**Configuration File:**
See the [TOKEN_MANAGEMENT.md](../TOKEN_MANAGEMENT.md) documentation for details on storing tokens securely.

### Example Configuration

```json
{
  "mcpServers": {
    "notion": {
      "url": "https://mcp.notion.com/mcp",
      "env": {
        "NOTION_TOKEN": "${NOTION_TOKEN}"
      }
    }
  }
}
```

## Usage Notes

- The integration must be explicitly granted access to each page or database you want to access
- Page and database IDs can be found in the Notion URL
- This is a remote server (connects via HTTPS)
- Rate limits apply according to Notion's API guidelines
- The server uses Notion's official API v1
