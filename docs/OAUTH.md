# OAuth Support for MCP Servers

MCP CLI now supports OAuth 2.0 authentication for HTTP-based MCP servers like Notion.

## Overview

The OAuth implementation provides:

- **Authorization Code Flow** with PKCE support
- **Automatic token management** (storage, refresh, expiration)
- **Browser-based authentication** with local callback server
- **Secure token storage** in `~/.mcp_cli/tokens/`

## Configuration

Add OAuth configuration to your server in `server_config.json`:

```json
{
  "mcpServers": {
    "notion": {
      "url": "https://mcp.notion.com/mcp",
      "oauth": {
        "authorization_url": "https://api.notion.com/v1/oauth/authorize",
        "token_url": "https://api.notion.com/v1/oauth/token",
        "client_id": "YOUR_CLIENT_ID",
        "client_secret": "YOUR_CLIENT_SECRET",
        "scopes": [],
        "redirect_uri": "http://localhost:8080/callback",
        "use_pkce": false
      }
    }
  }
}
```

### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `authorization_url` | Yes | OAuth authorization endpoint |
| `token_url` | Yes | OAuth token exchange endpoint |
| `client_id` | Yes | OAuth client ID from the provider |
| `client_secret` | No | OAuth client secret (not required for public clients) |
| `scopes` | No | List of OAuth scopes to request |
| `redirect_uri` | No | Callback URL (default: `http://localhost:8080/callback`) |
| `use_pkce` | No | Enable PKCE for additional security (default: `true`) |
| `extra_auth_params` | No | Additional parameters for authorization request |

## Setting Up Notion OAuth

1. **Create a Notion Integration**:
   - Go to https://www.notion.so/my-integrations
   - Click "New integration"
   - Set the integration type to "Public"
   - Note your Client ID and Client Secret

2. **Configure Redirect URI**:
   - In your Notion integration settings
   - Add redirect URI: `http://localhost:8080/callback`

3. **Update Configuration**:
   ```json
   {
     "mcpServers": {
       "notion": {
         "url": "https://mcp.notion.com/mcp",
         "oauth": {
           "authorization_url": "https://api.notion.com/v1/oauth/authorize",
           "token_url": "https://api.notion.com/v1/oauth/token",
           "client_id": "YOUR_NOTION_CLIENT_ID",
           "client_secret": "YOUR_NOTION_CLIENT_SECRET",
           "redirect_uri": "http://localhost:8080/callback"
         }
       }
     }
   }
   ```

## Usage

When you connect to a server with OAuth configured:

1. **First Time**: The CLI will open your browser for authorization
   ```bash
   mcp-cli --server notion

   🔐 Authentication required for notion
   ============================================================
   Opening browser for authorization...
   ```

2. **Authorize in Browser**: Grant permissions to the integration

3. **Automatic Token Management**: Tokens are stored securely and refreshed automatically

4. **Subsequent Uses**: Already authenticated - no browser needed

## Token Management

### Secure Token Storage

MCP CLI uses secure, OS-native token storage to protect your OAuth credentials. Tokens are **never** stored in plain text files.

#### Available Storage Backends

1. **macOS Keychain** (default on macOS)
   - Leverages macOS Keychain for secure credential storage
   - Integrates with system security features
   - Requires `keyring` library

2. **Windows Credential Manager** (default on Windows)
   - Uses Windows DPAPI for encrypted storage
   - Integrates with Windows security infrastructure
   - Requires `keyring` library

3. **Linux Secret Service** (default on Linux)
   - Supports GNOME Keyring, KWallet, and other Secret Service providers
   - Uses D-Bus Secret Service API
   - Requires `keyring` library and a keyring daemon

4. **HashiCorp Vault** (enterprise option)
   - Remote, centralized secret storage
   - Multi-tenant support with namespaces
   - Audit logging and access policies
   - Configure via environment variables or config file

5. **Encrypted File Storage** (fallback)
   - AES-256 encrypted files using PBKDF2 key derivation
   - Protected by user password
   - Used when OS keyring is unavailable

#### Configuration

By default, MCP CLI auto-detects the best storage backend for your platform. You can configure this in `server_config.json`:

```json
{
  "tokenStorage": {
    "backend": "auto",
    "password": "${MCP_CLI_ENCRYPTION_KEY}"
  }
}
```

**Backend Options:**
- `auto` - Auto-detect best available backend (default)
- `keychain` - macOS Keychain
- `windows` - Windows Credential Manager
- `secretservice` - Linux Secret Service
- `vault` - HashiCorp Vault
- `encrypted` - Encrypted file storage

**For HashiCorp Vault:**
```json
{
  "tokenStorage": {
    "backend": "vault",
    "vaultUrl": "https://vault.example.com:8200",
    "vaultToken": "${VAULT_TOKEN}",
    "vaultMountPoint": "secret",
    "vaultPathPrefix": "mcp-cli/oauth",
    "vaultNamespace": "my-namespace"
  }
}
```

Or use environment variables:
```bash
export VAULT_ADDR=https://vault.example.com:8200
export VAULT_TOKEN=s.your-token-here
```

**For Encrypted File Storage:**
```bash
export MCP_CLI_ENCRYPTION_KEY=your-secure-password
```

### Token Refresh

The system automatically:
- Detects expired tokens (with 5-minute buffer)
- Refreshes using refresh token if available
- Falls back to full OAuth flow if refresh fails

### Manual Token Reset

To re-authenticate a server, you'll need to delete the token from your secure storage backend:

**Using the CLI (recommended):**
```bash
# Future: mcp-cli auth reset {server_name}
```

**Manual removal:**
- **macOS Keychain**: Use Keychain Access app and search for "mcp-cli-oauth"
- **Windows**: Use Credential Manager and remove "mcp-cli-oauth" entries
- **Linux**: Use your keyring manager (seahorse, kwalletmanager, etc.)
- **Vault**: Delete from `{vault_path_prefix}/{server_name}`
- **Encrypted File**: Remove `~/.mcp_cli/tokens/{server_name}.enc`

After removal, next connection will trigger re-authentication:
```bash
mcp-cli --server {server_name}
```

## Security Considerations

1. **Secure Token Storage**:
   - Tokens are stored using OS-native secure storage (Keychain, Credential Manager, Secret Service)
   - Never stored in plain text files
   - Protected by OS-level encryption and access controls
   - Vault option provides enterprise-grade secret management with audit logging

2. **Client Secrets**: Store client secrets securely using environment variables:
   ```json
   {
     "oauth": {
       "client_id": "${NOTION_CLIENT_ID}",
       "client_secret": "${NOTION_CLIENT_SECRET}"
     }
   }
   ```

3. **PKCE**: Enabled by default for additional security. Disable only if the provider doesn't support it.

4. **Localhost Callback**: The callback server runs on localhost only and shuts down after receiving the callback

5. **Encrypted File Fallback**: When OS keyring is unavailable:
   - Uses AES-256 encryption with PBKDF2 key derivation
   - 480,000 iterations for key derivation (OWASP recommended)
   - Requires secure password (set via environment variable or prompt)

6. **HashiCorp Vault Integration**:
   - Supports both KV v1 and v2 secret engines
   - Namespace support for enterprise multi-tenancy
   - Token-based authentication (avoid storing Vault tokens in config)
   - Use short-lived tokens and token renewal policies

## Troubleshooting

### Storage Backend Issues

**Keyring not available:**
```bash
# Install keyring support
pip install keyring

# macOS: No additional setup needed
# Windows: No additional setup needed
# Linux: Install a keyring daemon (gnome-keyring, kwallet, or secretstorage)
```

**Vault connection issues:**
- Verify `VAULT_ADDR` and `VAULT_TOKEN` environment variables
- Check Vault token has permissions to read/write at the configured path
- Ensure network connectivity to Vault server
- Verify TLS certificates if using HTTPS

**Encrypted file password issues:**
- Set `MCP_CLI_ENCRYPTION_KEY` environment variable to avoid prompts
- Password is used to derive encryption key (cannot recover if lost)
- Delete `.salt` and `.enc` files in `~/.mcp_cli/tokens/` to start fresh

### Browser Doesn't Open

If the browser doesn't open automatically, copy the URL from the terminal and paste it into your browser.

### Port Already in Use

If port 8080 is in use, change the `redirect_uri` in your config and update it in your OAuth provider settings:

```json
{
  "oauth": {
    "redirect_uri": "http://localhost:9090/callback"
  }
}
```

### Token Refresh Fails

If token refresh continuously fails:
1. Delete the token from secure storage (see "Manual Token Reset" above)
2. Verify your OAuth configuration is correct
3. Check that your client credentials are valid

### Authorization Errors

Common issues:
- **Redirect URI mismatch**: Ensure the redirect URI in config matches your OAuth provider settings
- **Invalid client credentials**: Verify client ID and secret are correct
- **Missing scopes**: Add required scopes to the configuration

## Supported OAuth Flows

- ✅ **Authorization Code Flow** (with and without PKCE)
- ✅ **Token Refresh Flow**
- ❌ Client Credentials Flow (not needed for browser-based auth)
- ❌ Implicit Flow (deprecated, use Authorization Code + PKCE)

## Example: Complete Notion Setup

```json
{
  "mcpServers": {
    "notion": {
      "url": "https://mcp.notion.com/mcp",
      "oauth": {
        "authorization_url": "https://api.notion.com/v1/oauth/authorize",
        "token_url": "https://api.notion.com/v1/oauth/token",
        "client_id": "abcd1234-5678-90ef-ghij-klmnopqrstuv",
        "client_secret": "secret_abc123def456ghi789jkl012mno345pqr678stu",
        "scopes": [],
        "redirect_uri": "http://localhost:8080/callback",
        "use_pkce": false
      }
    }
  }
}
```

Running the CLI:
```bash
# First time - opens browser
mcp-cli --server notion

# After authentication
You: list my pages

# Token automatically included in all requests
```

## Architecture

The OAuth implementation consists of:

- **`oauth_config.py`**: Configuration models (OAuthConfig, OAuthTokens)
- **`oauth_flow.py`**: Authorization flow with local callback server
- **`token_manager.py`**: Token management with secure storage backends
- **`oauth_handler.py`**: High-level OAuth orchestration
- **`secure_token_store.py`**: Abstract interface for token storage
- **`token_store_factory.py`**: Factory for creating storage backends
- **`stores/`**: Platform-specific storage implementations
  - `keychain_store.py` - macOS Keychain
  - `windows_store.py` - Windows Credential Manager
  - `linux_store.py` - Linux Secret Service
  - `vault_store.py` - HashiCorp Vault
  - `encrypted_file_store.py` - Encrypted file fallback

Integration happens in `ToolManager` which:
1. Detects servers with OAuth configuration
2. Ensures valid tokens exist before connecting
3. Injects Authorization headers into HTTP requests
4. Handles token refresh transparently

Token storage backend is selected automatically based on platform and configuration, with the following priority:
1. Vault (if `VAULT_ADDR` and `VAULT_TOKEN` are set)
2. Platform-specific keyring (Keychain/Credential Manager/Secret Service)
3. Encrypted file storage (fallback)
