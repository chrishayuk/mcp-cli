# Cloudflare Workers MCP Server

## Overview

The Cloudflare Workers MCP server provides integration with Cloudflare's Workers platform, allowing you to manage, deploy, and interact with Workers, KV storage, Durable Objects, and other Cloudflare developer platform features.

**Transport:** Streamable HTTP

## What it Does

This server enables:
- Deploying and managing Cloudflare Workers
- Reading and writing to KV namespaces
- Managing Durable Objects
- Interacting with R2 storage
- Querying Workers analytics
- Managing Workers bindings and environment variables
- Working with Cloudflare's developer platform programmatically

## Tools

The Cloudflare Workers server provides the following tools:

### `deploy_worker`
Deploy a Worker script to Cloudflare.

**Parameters:**
- `name` (string, required): Name of the Worker
- `script` (string, required): Worker script content
- `bindings` (object, optional): KV, Durable Object, or other bindings

**Returns:** Deployment status and Worker URL

### `get_worker`
Retrieve information about a deployed Worker.

**Parameters:**
- `name` (string, required): Name of the Worker

**Returns:** Worker configuration and metadata

### `list_workers`
List all Workers in your account.

**Returns:** Array of Worker names and metadata

### `delete_worker`
Delete a Worker from your account.

**Parameters:**
- `name` (string, required): Name of the Worker to delete

**Returns:** Deletion confirmation

### `kv_get`
Get a value from a KV namespace.

**Parameters:**
- `namespace_id` (string, required): KV namespace identifier
- `key` (string, required): Key to retrieve

**Returns:** The value stored at the key

### `kv_put`
Store a value in a KV namespace.

**Parameters:**
- `namespace_id` (string, required): KV namespace identifier
- `key` (string, required): Key to store
- `value` (string, required): Value to store
- `expiration_ttl` (number, optional): TTL in seconds

**Returns:** Confirmation of storage

### `kv_list`
List keys in a KV namespace.

**Parameters:**
- `namespace_id` (string, required): KV namespace identifier
- `prefix` (string, optional): Filter by key prefix
- `limit` (number, optional): Maximum number of keys to return

**Returns:** Array of keys and metadata

### `kv_delete`
Delete a key from a KV namespace.

**Parameters:**
- `namespace_id` (string, required): KV namespace identifier
- `key` (string, required): Key to delete

**Returns:** Deletion confirmation

## Configuration

### Required Tokens

The Cloudflare Workers server requires authentication via Cloudflare API credentials. You need:

1. **API Token**: Create a token at https://dash.cloudflare.com/profile/api-tokens
   - Recommended permissions: Workers Scripts:Edit, Workers KV Storage:Edit, Account Settings:Read
   - Or use a token with appropriate scopes for your use case

2. **Account ID**: Find this in your Cloudflare dashboard URL or account settings

### Token Configuration

**Default Token Names:**
- `CLOUDFLARE_API_TOKEN` - Your Cloudflare API token
- `CLOUDFLARE_ACCOUNT_ID` - Your Cloudflare account identifier

**Authentication Type:** API Token

**OAuth Support:** No - Cloudflare uses API tokens for authentication.

Set the tokens using one of these methods:

**Environment Variables:**
```bash
export CLOUDFLARE_API_TOKEN="your_api_token_here"
export CLOUDFLARE_ACCOUNT_ID="your_account_id_here"
```

**Configuration File:**
See the [TOKEN_MANAGEMENT.md](../TOKEN_MANAGEMENT.md) documentation for details on storing tokens securely.

### Example Configuration

```json
{
  "mcpServers": {
    "cloudflare_workers": {
      "url": "https://bindings.mcp.cloudflare.com/mcp",
      "env": {
        "CLOUDFLARE_API_TOKEN": "${CLOUDFLARE_API_TOKEN}",
        "CLOUDFLARE_ACCOUNT_ID": "${CLOUDFLARE_ACCOUNT_ID}"
      }
    }
  }
}
```

## Usage Notes

- This is a remote server (connects via HTTPS to Cloudflare's API)
- API rate limits apply according to your Cloudflare plan
- Workers deployed via this server follow standard Cloudflare Workers pricing
- KV operations are subject to KV storage limits and pricing
- Ensure your API token has appropriate permissions for the operations you need
- Account ID is required for all operations and must match the token's account
