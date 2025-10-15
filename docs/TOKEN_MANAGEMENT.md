# Token Management

MCP CLI provides comprehensive token management for OAuth tokens, bearer tokens, API keys, and other credentials through the `mcp-cli token` command.

## Overview

All tokens are stored securely using OS-native secure storage (Keychain, Credential Manager, Secret Service) or HashiCorp Vault. See [OAUTH.md](./OAUTH.md) for details on storage backends.

## Commands

### List Tokens

View all stored tokens (metadata only, no sensitive values):

```bash
# List all tokens
mcp-cli token list

# Filter by namespace
mcp-cli token list --namespace provider-keys

# Show only specific types
mcp-cli token list --no-oauth --api-keys
mcp-cli token list --bearer --no-api-keys
```

**Output includes:**
- **Type**: Token type (bearer, api_key, etc.)
- **Name**: Token identifier
- **Created**: Date token was registered
- **Expires**: Expiration date (if applicable, shows ⚠️ if expired)
- **Details**: Provider, namespace, and other metadata

### Store Tokens

Manually store tokens for authentication:

```bash
# Store a bearer token (provide value as option)
mcp-cli token set my-server --type bearer --value "abc123..."

# Store an API key
mcp-cli token set openai --type api-key --provider openai --value "sk-..."

# Store in custom namespace
mcp-cli token set my-token --type bearer --namespace custom --value "token123"

# Prompt for value (more secure - recommended)
mcp-cli token set my-server --type bearer
# Enter token value for 'my-server': [hidden input]
```

**Note:** The `--value` option should be used carefully as it may expose tokens in shell history. Using the interactive prompt (omitting `--value`) is more secure.

### Get Token Info

Retrieve information about a stored token (metadata only - actual values never shown):

```bash
# Get metadata
mcp-cli token get my-server

# Get from specific namespace
mcp-cli token get openai --namespace api-key
```

### Delete Tokens

Remove stored tokens:

```bash
# Delete a generic token
mcp-cli token delete my-server

# Delete from specific namespace
mcp-cli token delete openai --namespace api-key

# Delete OAuth token for a server
mcp-cli token delete notion --oauth
```

### Clear All Tokens

Remove multiple tokens at once:

```bash
# Clear tokens in a namespace
mcp-cli token clear --namespace bearer

# Clear ALL tokens (with confirmation)
mcp-cli token clear

# Skip confirmation
mcp-cli token clear --force
```

### List Storage Backends

View available and active storage backends:

```bash
mcp-cli token backends
```

## Token Types

### OAuth Tokens

OAuth tokens are managed automatically during MCP server authentication. They include:
- Access token
- Refresh token (if available)
- Expiration time
- Issued timestamp

**Namespace:** Stored directly by server name (e.g., `notion`, `github`)

**Example:**
```bash
# View OAuth token metadata
mcp-cli token get notion --oauth

# Delete OAuth token
mcp-cli token delete notion --oauth
```

### Bearer Tokens

Simple bearer tokens for HTTP authentication.

**Namespace:** `bearer` (default) or custom

**Example:**
```bash
# Store bearer token
mcp-cli token set my-api --type bearer --value "token_abc123"

# Use in MCP server config
{
  "mcpServers": {
    "my-api": {
      "url": "https://api.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${TOKEN:bearer:my-api}"
      }
    }
  }
}
```

### API Keys

Provider API keys for LLM services (OpenAI, Anthropic, Google, etc.).

**Namespace:** `api-key` or `provider`

**Example:**
```bash
# Store API key
mcp-cli token set openai --type api-key --provider openai --value "sk-..."

# Store with organization
mcp-cli token set anthropic \
  --type api-key \
  --provider anthropic \
  --value "sk-ant-..."

# Use via environment variable substitution
export OPENAI_API_KEY=$(mcp-cli token get openai --namespace api-key --show-value)
```

### Basic Authentication

Username/password credentials (future enhancement).

## Namespaces

Tokens are organized into namespaces for better organization:

| Namespace | Purpose | Examples |
|-----------|---------|----------|
| `oauth` | OAuth tokens (implicit) | Notion, GitHub, Google |
| `bearer` | Bearer tokens | Custom APIs |
| `api-key` | Provider API keys | OpenAI, Anthropic |
| `provider` | Provider credentials | Alternative to api-key |
| `generic` | General purpose | Custom tokens |
| `custom-*` | User-defined | Any custom namespace |

## Configuration Integration

### Using Tokens in Server Config

Reference stored tokens in `server_config.json`:

```json
{
  "mcpServers": {
    "custom-api": {
      "url": "https://api.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${TOKEN:bearer:my-api}",
        "X-API-Key": "${TOKEN:api-key:my-service}"
      }
    }
  }
}
```

### Environment Variable Pattern

Reference tokens directly in configuration using the `${TOKEN:namespace:name}` pattern:

```json
{
  "mcpServers": {
    "my-service": {
      "command": "node",
      "args": ["server.js"],
      "env": {
        "API_TOKEN": "${TOKEN:bearer:my-service}"
      }
    }
  }
}
```

## Security Best Practices

1. **Tokens are never displayed** - The CLI shows only metadata, never actual token values
2. **Use namespaces** - Organize tokens by purpose for better management
3. **Rotate regularly** - Update tokens periodically using `delete` + `set`
4. **Use OAuth when possible** - Auto-refreshing tokens are more secure
5. **Leverage OS keychains** - Better than file-based storage
6. **Consider Vault for teams** - Centralized secret management with audit logs

## Common Workflows

### Migrating API Keys from Environment Variables

```bash
# Store existing API keys
mcp-cli token set openai --type api-key --provider openai --value "$OPENAI_API_KEY"
mcp-cli token set anthropic --type api-key --provider anthropic --value "$ANTHROPIC_API_KEY"

# Remove from environment
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY

# Verify storage
mcp-cli token list --api-keys
```

### Setting Up Bearer Token Authentication

```bash
# Store token
mcp-cli token set my-service --type bearer

# Configure server
cat > server_config.json <<EOF
{
  "mcpServers": {
    "my-service": {
      "url": "https://api.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${TOKEN:bearer:my-service}"
      }
    }
  }
}
EOF
```

### Cleaning Up Test Tokens

```bash
# List tokens in test namespace
mcp-cli token list --namespace test

# Clear all test tokens
mcp-cli token clear --namespace test --force
```

### Team Setup with HashiCorp Vault

```bash
# Configure Vault backend
export VAULT_ADDR=https://vault.company.com
export VAULT_TOKEN=s.your-token

# Verify backend
mcp-cli token backends

# Store shared tokens
mcp-cli token set shared-api --type bearer --namespace team

# Team members with same Vault access can retrieve
mcp-cli token get shared-api --namespace team
```

## Troubleshooting

### Token Not Found

```bash
# Try different namespaces
mcp-cli token get my-token --namespace bearer
mcp-cli token get my-token --namespace generic

# List all to find it
mcp-cli token list
```

### Backend Issues

```bash
# Check available backends
mcp-cli token backends

# Verify current backend in config
cat server_config.json | grep -A5 tokenStorage

# Force specific backend
# Edit server_config.json:
{
  "tokenStorage": {
    "backend": "encrypted"
  }
}
```

### Permission Denied

- **macOS**: Grant terminal access to Keychain in System Preferences
- **Windows**: Run as user with credential access
- **Linux**: Ensure keyring daemon is running (`gnome-keyring-daemon`, `kwalletd`)
- **Vault**: Verify token has read/write permissions to path

## Architecture

Token management system consists of:

- **`token_types.py`**: Token models (OAuth, Bearer, APIKey, BasicAuth)
- **`secure_token_store.py`**: Abstract storage interface
- **`token_store_factory.py`**: Backend selection and creation
- **`stores/`**: Platform-specific storage implementations
- **`token_manager.py`**: High-level token management
- **`commands/token.py`**: CLI command implementation

## Future Enhancements

Planned features:

- [ ] `mcp-cli token export` - Export tokens to encrypted file
- [ ] `mcp-cli token import` - Import tokens from file
- [ ] `mcp-cli token rotate` - Automatic token rotation
- [ ] `mcp-cli token validate` - Test token validity
- [ ] Support for client certificates
- [ ] Token expiration notifications
- [ ] Audit logging for token access
