#!/bin/bash
# Complete example: Add a server, use its tools, then remove it
# This demonstrates the full lifecycle with actual tool execution

set -e

echo "=============================================="
echo "🚀 Complete Server Management & Tool Usage Demo"
echo "=============================================="
echo ""
echo "This demo will:"
echo "1. Add a new MCP server (filesystem server)"
echo "2. Connect and list its tools"
echo "3. Execute a tool from the server"
echo "4. Remove the server"
echo ""

# Step 1: Add filesystem MCP server
echo "Step 1: Adding filesystem MCP server"
echo "-------------------------------------"
cat > /tmp/add_fs_server.py << 'EOF'
import asyncio
from mcp_cli.context import initialize_context
from mcp_cli.commands.actions.servers import servers_action_async

async def main():
    initialize_context()
    # Add filesystem server with access to /tmp directory
    await servers_action_async(args=[
        "add", 
        "demo-fs",
        "stdio",  # transport type
        "npx",    # command
        "@modelcontextprotocol/server-filesystem"  # package
    ])
    print("\nServer added. Configuration saved to server_config.json")

asyncio.run(main())
EOF

uv run python /tmp/add_fs_server.py
echo ""

# Step 2: Show the server was added
echo "Step 2: Verifying server was added"
echo "-----------------------------------"
uv run mcp-cli servers | grep -A 1 -B 1 "demo-fs" || echo "Server listing shown above"
echo ""

# Step 3: Start a new session to connect to the server and list tools
echo "Step 3: Listing tools from the filesystem server"
echo "------------------------------------------------"
echo "Note: This requires restarting to connect to the new server"
echo ""

# We'll use the command mode to list tools
uv run mcp-cli tools --server demo-fs 2>/dev/null | head -20 || {
    echo "New servers need a session restart to connect."
    echo "In a real scenario, you would restart your chat session."
    echo ""
    echo "Alternative: Use existing 'echo' server to demonstrate tool execution"
    uv run mcp-cli tools --server echo | head -10
}
echo ""

# Step 4: Execute a tool
echo "Step 4: Executing a tool from echo server"
echo "-----------------------------------------"
echo "Executing 'echo_message' tool with message 'Hello from MCP!'"
echo ""

# Execute echo tool
uv run mcp-cli cmd --server echo --tool echo_message <<< '{"message": "Hello from MCP CLI!"}' | head -20
echo ""

# Step 5: Demonstrate adding a Playwright server with real-world syntax
echo "Step 5: Real-world example - Adding Playwright server"
echo "-----------------------------------------------------"
cat > /tmp/add_playwright.py << 'EOF'
import asyncio
from mcp_cli.context import initialize_context
from mcp_cli.commands.actions.servers import servers_action_async

async def main():
    initialize_context()
    await servers_action_async(args=[
        "add",
        "playwright",
        "stdio",
        "npx",
        "@playwright/mcp@latest"
    ])
    print("Playwright server added successfully!")

asyncio.run(main())
EOF

uv run python /tmp/add_playwright.py
echo ""

# Step 6: Show all servers
echo "Step 6: All configured servers"
echo "-------------------------------"
uv run mcp-cli servers
echo ""

# Step 7: Clean up - remove demo servers
echo "Step 7: Cleaning up - removing demo servers"
echo "-------------------------------------------"
cat > /tmp/cleanup_servers.py << 'EOF'
import asyncio
from mcp_cli.context import initialize_context
from mcp_cli.commands.actions.servers import servers_action_async

async def main():
    initialize_context()
    # Remove demo servers
    for server in ["demo-fs", "playwright"]:
        print(f"Removing {server}...")
        await servers_action_async(args=["remove", server])

asyncio.run(main())
EOF

uv run python /tmp/cleanup_servers.py
echo ""

# Final server list
echo "Step 8: Final server configuration"
echo "----------------------------------"
uv run mcp-cli servers
echo ""

# Clean up temp files
rm -f /tmp/add_fs_server.py /tmp/add_playwright.py /tmp/cleanup_servers.py

echo "=============================================="
echo "✅ Demo Complete!"
echo "=============================================="
echo ""
echo "Key takeaways:"
echo "• Servers can be added at runtime with various transports (stdio, http, sse)"
echo "• Environment variables and headers can be configured"
echo "• New servers require session restart to fully connect"
echo "• Configuration persists in server_config.json"
echo "• Tools from connected servers can be executed immediately"
echo ""
echo "Try it yourself in chat mode:"
echo "  /server add playwright stdio npx @playwright/mcp@latest"
echo "  /server add github --env GITHUB_TOKEN=your-token -- stdio npx github-mcp-server"
echo "  /server add api --transport http --header \"Authorization: Bearer token\" -- https://api.example.com/mcp"
echo ""