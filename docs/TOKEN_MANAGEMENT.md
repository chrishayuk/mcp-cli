# Token Management

MCP CLI provides comprehensive token management for OAuth tokens, bearer tokens, API keys, and other credentials through the `mcp-cli token` command and integrated provider/server workflows.

## Overview

All tokens are stored securely using OS-native secure storage (Keychain, Credential Manager, Secret Service) or HashiCorp Vault. Tokens are used throughout MCP CLI for:

- **Provider Authentication**: API keys for LLM providers (OpenAI, Anthropic, etc.)
- **Server Authentication**: OAuth and bearer tokens for MCP servers
- **Generic Credentials**: Any authentication token needed by tools or servers

See [OAUTH.md](./OAUTH.md) for details on OAuth workflows and storage backends.

## Token Types

### OAuth Tokens

OAuth tokens are managed automatically during MCP server authentication. They include:
- Access token
- Refresh token (if available)
- Expiration time
- Issued timestamp
- Auto-refresh capability

**Namespace:** Stored by server name (e.g., `notion`, `github`)

**Typical Usage:**
- Automatically obtained through OAuth flow when connecting to OAuth-enabled MCP servers
- Refreshed automatically when expired
- Used transparently for server authentication

**Example:**
```bash
# View OAuth token metadata for a server
mcp-cli token get notion --oauth

# Delete OAuth token (forces re-authentication)
mcp-cli token delete notion --oauth

# List all OAuth tokens
mcp-cli token list --oauth
```

### Bearer Tokens

Simple bearer tokens for HTTP authentication with APIs and MCP servers.

**Namespace:** `bearer` (default) or custom

**Typical Usage:**
- Static API authentication
- MCP server HTTP/SSE transport authentication
- Custom service authentication

**Example:**
```bash
# Store bearer token interactively (recommended - no shell history)
mcp-cli token set my-api --type bearer
# Enter token value for 'my-api': [hidden input]

# Or provide directly (less secure - visible in shell history)
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

**Typical Usage:**
- LLM provider authentication
- Cloud service API keys
- Third-party service keys

**Example:**
```bash
# Store API key for OpenAI
mcp-cli token set openai --type api-key --provider openai
# Enter token value for 'openai': [hidden input]

# Store with explicit value
mcp-cli token set anthropic --type api-key --provider anthropic --value "sk-ant-..."

# Provider commands use these automatically
mcp-cli provider set openai    # Uses stored api-key if available
mcp-cli --provider openai --server sqlite  # Uses stored key

# Or reference explicitly
export OPENAI_API_KEY=$(mcp-cli token get openai --namespace api-key --show-value)
```

### Basic Authentication

Username/password credentials (planned feature).

## Commands

### List Tokens

View all stored tokens (metadata only, no sensitive values):

```bash
# List all tokens
mcp-cli token list

# Filter by type
mcp-cli token list --oauth          # Only OAuth tokens
mcp-cli token list --api-keys       # Only API keys
mcp-cli token list --bearer         # Only bearer tokens

# Filter by namespace
mcp-cli token list --namespace provider-keys

# Combine filters
mcp-cli token list --no-oauth --api-keys  # API keys but not OAuth
```

**Output includes:**
- **Type**: Token type (oauth, bearer, api_key, etc.)
- **Name**: Token identifier
- **Created**: Date token was registered
- **Expires**: Expiration date (if applicable, shows ⚠️ if expired)
- **Details**: Provider, namespace, and other metadata

### Store Tokens

Manually store tokens for authentication:

```bash
# Interactive mode (recommended - more secure)
mcp-cli token set my-server --type bearer
# Enter token value for 'my-server': [hidden input]

# Store a bearer token with value
mcp-cli token set my-server --type bearer --value "abc123..."

# Store an API key
mcp-cli token set openai --type api-key --provider openai --value "sk-..."

# Store in custom namespace
mcp-cli token set my-token --type bearer --namespace custom --value "token123"

# With expiration
mcp-cli token set temp-token --type bearer --expires "2025-12-31" --value "xyz789"
```

**Security Note:** The `--value` option should be used carefully as it may expose tokens in shell history. Using the interactive prompt (omitting `--value`) is more secure.

### Get Token Info

Retrieve information about a stored token (metadata only - actual values never shown):

```bash
# Get metadata
mcp-cli token get my-server

# Get from specific namespace
mcp-cli token get openai --namespace api-key

# Get OAuth token
mcp-cli token get notion --oauth

# Show value (use with caution)
mcp-cli token get my-server --show-value
```

### Delete Tokens

Remove stored tokens:

```bash
# Delete a bearer token
mcp-cli token delete my-server

# Delete from specific namespace
mcp-cli token delete openai --namespace api-key

# Delete OAuth token for a server
mcp-cli token delete notion --oauth

# Force delete without confirmation
mcp-cli token delete my-server --force
```

### Clear All Tokens

Remove multiple tokens at once:

```bash
# Clear tokens in a namespace (with confirmation)
mcp-cli token clear --namespace bearer

# Clear ALL tokens (with confirmation)
mcp-cli token clear

# Skip confirmation
mcp-cli token clear --force

# Clear only expired tokens
mcp-cli token clear --expired
```

### List Storage Backends

View available and active storage backends:

```bash
mcp-cli token backends

# Output shows:
# - Available backends (keyring, encrypted, vault)
# - Active backend
# - Backend health status
```

## Integration with Providers

Tokens are deeply integrated with the provider system for seamless LLM provider authentication.

### Automatic Provider Token Usage

When you configure or use a provider, MCP CLI automatically checks for stored API keys:

```bash
# Set provider - automatically uses stored token if available
mcp-cli provider set openai

# Or prompts for API key if not stored
# Enter API key for openai: [hidden input]
# Save to secure storage? [Y/n]: y

# Use provider - automatically retrieves stored token
mcp-cli --provider openai --server sqlite
```

### Provider Configuration with Tokens

```bash
# Interactive provider setup (stores token securely)
mcp-cli provider set openai
# API key for openai: [hidden input]
# Save to secure storage? [Y/n]: y

# Set provider configuration including token
mcp-cli provider set openai api_key sk-...

# View provider status (shows if token is stored)
mcp-cli provider
# Provider: openai
# Status: ✅ Ready (API key stored securely)

# List providers with token status
mcp-cli providers
# Provider   | Status    | Token Status
# openai     | ✅ Ready  | 🔑 Stored
# anthropic  | ✅ Ready  | 🔑 Stored
# ollama     | ✅ Ready  | ⚠️  No key needed
```

### Token-based Provider Switching

```bash
# Switch to provider that has stored token
mcp-cli provider openai
# Using stored API key ✅

# Switch to provider without token
mcp-cli provider anthropic
# No API key found. Enter API key: [hidden input]
# Save to secure storage? [Y/n]: y

# Override stored token temporarily
mcp-cli --provider openai --api-key sk-temp-key --server sqlite
# Note: Temporary key not saved
```

## Integration with Servers

Tokens are used for MCP server authentication, especially for OAuth and HTTP/SSE transports.

### OAuth Server Authentication

OAuth tokens are managed automatically for OAuth-enabled MCP servers:

```bash
# Add OAuth-enabled server (triggers OAuth flow)
mcp-cli server add notion --oauth \
  --client-id "your-client-id" \
  --client-secret "your-secret" \
  --auth-url "https://api.notion.com/v1/oauth/authorize" \
  --token-url "https://api.notion.com/v1/oauth/token"

# OAuth flow starts automatically:
# 1. Browser opens to authorization page
# 2. User grants permission
# 3. Token automatically stored securely
# 4. Server connection established

# View OAuth token status
mcp-cli token get notion --oauth

# Server automatically uses stored OAuth token
mcp-cli --server notion

# Token auto-refreshes when expired
# Manual refresh if needed:
mcp-cli token refresh notion --oauth
```

### Bearer Token Server Authentication

For HTTP/SSE MCP servers requiring bearer token authentication:

```bash
# Store bearer token for server
mcp-cli token set github-mcp --type bearer --value "ghp_..."

# Configure server to use token
{
  "mcpServers": {
    "github-mcp": {
      "transport": "http",
      "url": "https://api.github.com/mcp",
      "headers": {
        "Authorization": "Bearer ${TOKEN:bearer:github-mcp}"
      }
    }
  }
}

# Or add dynamically with token reference
mcp-cli server add github-mcp \
  --transport http \
  --header "Authorization: Bearer ${TOKEN:bearer:github-mcp}" \
  -- https://api.github.com/mcp
```

### Server Environment Variables with Tokens

Tokens can be injected as environment variables for STDIO servers:

```bash
# Store API token
mcp-cli token set my-service-key --type api-key --value "key_..."

# Reference in server config
{
  "mcpServers": {
    "my-service": {
      "command": "node",
      "args": ["server.js"],
      "env": {
        "API_TOKEN": "${TOKEN:api-key:my-service-key}",
        "SERVICE_KEY": "${TOKEN:bearer:my-service}"
      }
    }
  }
}
```

## Namespaces

Tokens are organized into namespaces for better organization:

| Namespace | Purpose | Examples |
|-----------|---------|----------|
| `oauth` | OAuth tokens (implicit) | Notion, GitHub, Google servers |
| `bearer` | Bearer tokens | Custom APIs, HTTP MCP servers |
| `api-key` | Provider API keys | OpenAI, Anthropic, Gemini |
| `provider` | Provider credentials | Alternative to api-key |
| `generic` | General purpose | Custom tokens |
| `custom-*` | User-defined | Any custom namespace |

## Configuration Integration

### Token References in Configuration

Use the `${TOKEN:namespace:name}` pattern to reference stored tokens:

```json
{
  "mcpServers": {
    "custom-api": {
      "url": "https://api.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${TOKEN:bearer:my-api}",
        "X-API-Key": "${TOKEN:api-key:my-service}"
      }
    },
    "secure-service": {
      "command": "node",
      "args": ["server.js"],
      "env": {
        "API_TOKEN": "${TOKEN:bearer:my-service}",
        "OAUTH_TOKEN": "${TOKEN:oauth:my-oauth-server}"
      }
    }
  }
}
```

### Provider Configuration File

Provider tokens are stored separately from `~/.chuk_llm/config.yaml` for security:

```yaml
# ~/.chuk_llm/config.yaml - NO tokens stored here
openai:
  api_base: https://api.openai.com/v1
  default_model: gpt-4o

anthropic:
  api_base: https://api.anthropic.com
  default_model: claude-3-5-sonnet-20241022
```

Tokens are stored securely in the OS keyring or encrypted storage, referenced by provider name.

## Security Best Practices

1. **Tokens are never displayed** - The CLI shows only metadata, never actual token values
2. **Use interactive prompts** - Avoid `--value` flag to prevent shell history exposure
3. **Use namespaces** - Organize tokens by purpose for better management
4. **Rotate regularly** - Update tokens periodically using `delete` + `set`
5. **Use OAuth when possible** - Auto-refreshing tokens are more secure than static tokens
6. **Leverage OS keychains** - Better than file-based storage
7. **Consider Vault for teams** - Centralized secret management with audit logs
8. **Principle of least privilege** - Store only necessary tokens
9. **Monitor expiration** - Use `token list` to check for expired tokens
10. **Audit token usage** - Review stored tokens periodically with `token list`

## Common Workflows

### Initial Provider Setup

```bash
# Set up OpenAI with token storage
mcp-cli provider set openai
# Enter API key: [hidden input]
# Save to secure storage? [Y/n]: y

# Set up Anthropic
mcp-cli provider set anthropic
# Enter API key: [hidden input]
# Save to secure storage? [Y/n]: y

# Verify stored tokens
mcp-cli token list --api-keys

# Provider    | Status    | Token
# openai      | ✅ Ready  | 🔑 Stored
# anthropic   | ✅ Ready  | 🔑 Stored
```

### Migrating API Keys from Environment Variables

```bash
# Store existing API keys from environment
mcp-cli token set openai --type api-key --provider openai --value "$OPENAI_API_KEY"
mcp-cli token set anthropic --type api-key --provider anthropic --value "$ANTHROPIC_API_KEY"

# Remove from environment for security
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY

# Verify storage
mcp-cli token list --api-keys

# Providers now use stored tokens automatically
mcp-cli --provider openai --server sqlite  # Uses stored token
```

### Setting Up OAuth MCP Server

```bash
# Add OAuth server (automatic token flow)
mcp-cli server add notion \
  --transport http \
  --oauth \
  --client-id "oauth2_..." \
  --client-secret "secret_..." \
  --auth-url "https://api.notion.com/v1/oauth/authorize" \
  --token-url "https://api.notion.com/v1/oauth/token" \
  -- https://api.notion.com/v1/mcp

# OAuth browser flow starts automatically
# Token stored securely after authorization

# Check token status
mcp-cli token get notion --oauth

# Token used automatically when connecting
mcp-cli --server notion
```

### Setting Up Bearer Token Authentication for MCP Server

```bash
# Store token securely
mcp-cli token set my-service --type bearer
# Enter token value for 'my-service': [hidden input]

# Configure server with token reference
cat > server_config.json <<EOF
{
  "mcpServers": {
    "my-service": {
      "transport": "http",
      "url": "https://api.example.com/mcp",
      "headers": {
        "Authorization": "Bearer \${TOKEN:bearer:my-service}"
      }
    }
  }
}
EOF

# Connect to server (uses stored token)
mcp-cli --server my-service
```

### Team Setup with HashiCorp Vault

```bash
# Configure Vault backend
export VAULT_ADDR=https://vault.company.com
export VAULT_TOKEN=s.your-token

# Verify backend
mcp-cli token backends
# Active: vault
# Status: ✅ Connected

# Store shared team tokens
mcp-cli token set shared-api --type bearer --namespace team
# Enter token value for 'shared-api': [hidden input]

# Team members with same Vault access can retrieve
mcp-cli token get shared-api --namespace team

# Provider tokens shared across team
mcp-cli token set openai --type api-key --provider openai
# All team members can now use: mcp-cli --provider openai
```

### Rotating Tokens

```bash
# List tokens to find ones needing rotation
mcp-cli token list

# Rotate a bearer token
mcp-cli token delete my-api --force
mcp-cli token set my-api --type bearer
# Enter token value: [new token]

# Rotate provider API key
mcp-cli token delete openai --namespace api-key
mcp-cli provider set openai
# Enter API key: [new key]
# Save to secure storage? [Y/n]: y

# Rotate OAuth token (forces re-authentication)
mcp-cli token delete notion --oauth
mcp-cli --server notion  # Triggers new OAuth flow
```

### Cleaning Up Test Tokens

```bash
# List tokens in test namespace
mcp-cli token list --namespace test

# Clear all test tokens
mcp-cli token clear --namespace test --force

# Clear expired tokens across all namespaces
mcp-cli token clear --expired --force
```

## Troubleshooting

### Token Not Found

```bash
# Check all namespaces
mcp-cli token list

# Try specific namespace
mcp-cli token get my-token --namespace bearer
mcp-cli token get my-token --namespace api-key
mcp-cli token get my-token --oauth

# List tokens to find it
mcp-cli token list | grep my-token
```

### Provider Cannot Find Token

```bash
# Check if token is stored
mcp-cli token list --api-keys

# Check provider status
mcp-cli provider
mcp-cli providers

# Re-set provider with new token
mcp-cli provider set openai
# Enter API key: [hidden input]
```

### Server Authentication Failing

```bash
# Check server token
mcp-cli token get my-server --oauth
mcp-cli token get my-server --namespace bearer

# Check server config references
cat server_config.json | grep TOKEN

# Verify token is not expired
mcp-cli token list | grep my-server

# Refresh OAuth token
mcp-cli token refresh my-server --oauth

# Re-store bearer token
mcp-cli token delete my-server
mcp-cli token set my-server --type bearer
```

### Backend Issues

```bash
# Check available backends
mcp-cli token backends

# Verify current backend
cat ~/.mcp-cli/config.yaml | grep -A5 tokenStorage

# Switch to different backend (edit config)
{
  "tokenStorage": {
    "backend": "encrypted"  # or "keyring", "vault"
  }
}

# Test backend
mcp-cli token set test-token --type bearer --value "test123"
mcp-cli token get test-token
mcp-cli token delete test-token
```

### Permission Denied

Platform-specific solutions:

- **macOS**: Grant terminal access to Keychain in System Preferences > Privacy & Security
- **Windows**: Run as user with credential access, check Credential Manager permissions
- **Linux**: Ensure keyring daemon is running:
  ```bash
  # Check daemon status
  ps aux | grep -E 'gnome-keyring|kwallet|keepass'

  # Start gnome-keyring
  gnome-keyring-daemon --start

  # Or use encrypted fallback
  export MCP_TOKEN_BACKEND=encrypted
  ```
- **Vault**: Verify token has read/write permissions to secret path

### Token Value Not Showing

By design, token values are never displayed for security. If you need to retrieve a value:

```bash
# Use --show-value flag (use carefully)
mcp-cli token get my-token --show-value

# Or export to environment variable
export MY_TOKEN=$(mcp-cli token get my-token --show-value)
```

## Architecture

Token management system consists of:

- **Token Models** (`token_types.py`): OAuth, Bearer, APIKey, BasicAuth models
- **Storage Interface** (`secure_token_store.py`): Abstract storage interface
- **Storage Factory** (`token_store_factory.py`): Backend selection and creation
- **Storage Backends** (`stores/`): Platform-specific implementations
  - `keyring_store.py`: OS keychain integration
  - `encrypted_store.py`: Encrypted file storage
  - `vault_store.py`: HashiCorp Vault integration
- **Token Manager** (`token_manager.py`): High-level token operations
- **CLI Commands** (`commands/token.py`): Command-line interface
- **Provider Integration**: Automatic token lookup for providers
- **Server Integration**: OAuth flow and bearer token injection

## Future Enhancements

Planned features:

- [ ] `mcp-cli token export` - Export tokens to encrypted file
- [ ] `mcp-cli token import` - Import tokens from file
- [ ] `mcp-cli token rotate` - Automatic token rotation
- [ ] `mcp-cli token validate` - Test token validity
- [ ] `mcp-cli token refresh --all` - Refresh all OAuth tokens
- [ ] Support for client certificates
- [ ] Token expiration notifications
- [ ] Audit logging for token access
- [ ] Token permission scopes
- [ ] Multi-tenancy support
- [ ] Token templates for common providers
- [ ] Automatic token rotation policies
- [ ] Integration with external secret managers (AWS Secrets Manager, Azure Key Vault)
