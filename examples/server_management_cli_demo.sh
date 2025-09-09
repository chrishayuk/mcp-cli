#!/bin/bash
# End-to-end demo of MCP server management via CLI
# This script demonstrates adding, using, and removing MCP servers at runtime

set -e  # Exit on error

echo "================================================"
echo "🚀 MCP Server Runtime Management Demo"
echo "================================================"
echo ""

# Function to pause between steps
pause() {
    echo ""
    echo "Press Enter to continue..."
    read -r
    echo ""
}

# Step 1: Show initial servers
echo "Step 1: Listing current servers"
echo "--------------------------------"
uv run mcp-cli servers
pause

# Step 2: Add a new server (using the time server as it's simple)
echo "Step 2: Adding a new MCP time server"
echo "------------------------------------"
echo "Command: mcp-cli server add demo-time stdio uvx mcp-server-time"
echo ""

# Create a simple Python script to add the server
cat > /tmp/add_server.py << 'EOF'
import asyncio
from mcp_cli.context import initialize_context
from mcp_cli.commands.actions.servers import servers_action_async

async def main():
    initialize_context()
    await servers_action_async(args=["add", "demo-time", "stdio", "uvx", "mcp-server-time"])

asyncio.run(main())
EOF

uv run python /tmp/add_server.py
pause

# Step 3: List servers to show the new one
echo "Step 3: Listing servers (showing new server)"
echo "--------------------------------------------"
uv run mcp-cli servers
pause

# Step 4: Show details of the new server
echo "Step 4: Showing details of demo-time server"
echo "-------------------------------------------"
cat > /tmp/show_server.py << 'EOF'
import asyncio
from mcp_cli.context import initialize_context
from mcp_cli.commands.actions.servers import servers_action_async

async def main():
    initialize_context()
    await servers_action_async(args=["demo-time"])

asyncio.run(main())
EOF

uv run python /tmp/show_server.py
pause

# Step 5: Test the server by listing tools (requires reconnection)
echo "Step 5: Testing server by listing available tools"
echo "-------------------------------------------------"
echo "Note: New servers require session restart to connect"
echo ""

# This will show tools from all connected servers
uv run mcp-cli tools --limit 10
pause

# Step 6: Execute a tool from the server (if connected)
echo "Step 6: Attempting to execute a tool"
echo "------------------------------------"
echo "Using the 'echo_message' tool from echo server as example"
echo ""

# Try to execute a simple tool
uv run mcp-cli cmd --server echo --tool echo_message --raw <<< '{"message": "Hello from CLI!"}'
pause

# Step 7: Disable the server
echo "Step 7: Disabling the demo-time server"
echo "--------------------------------------"
cat > /tmp/disable_server.py << 'EOF'
import asyncio
from mcp_cli.context import initialize_context
from mcp_cli.commands.actions.servers import servers_action_async

async def main():
    initialize_context()
    await servers_action_async(args=["disable", "demo-time"])

asyncio.run(main())
EOF

uv run python /tmp/disable_server.py
pause

# Step 8: Show servers with disabled status
echo "Step 8: Listing servers (showing disabled status)"
echo "-------------------------------------------------"
cat > /tmp/list_all_servers.py << 'EOF'
import asyncio
from mcp_cli.context import initialize_context
from mcp_cli.commands.actions.servers import servers_action_async

async def main():
    initialize_context()
    await servers_action_async(args=["list", "all"])

asyncio.run(main())
EOF

uv run python /tmp/list_all_servers.py
pause

# Step 9: Re-enable the server
echo "Step 9: Re-enabling the demo-time server"
echo "----------------------------------------"
cat > /tmp/enable_server.py << 'EOF'
import asyncio
from mcp_cli.context import initialize_context
from mcp_cli.commands.actions.servers import servers_action_async

async def main():
    initialize_context()
    await servers_action_async(args=["enable", "demo-time"])

asyncio.run(main())
EOF

uv run python /tmp/enable_server.py
pause

# Step 10: Remove the server
echo "Step 10: Removing the demo-time server"
echo "--------------------------------------"
cat > /tmp/remove_server.py << 'EOF'
import asyncio
from mcp_cli.context import initialize_context
from mcp_cli.commands.actions.servers import servers_action_async

async def main():
    initialize_context()
    await servers_action_async(args=["remove", "demo-time"])

asyncio.run(main())
EOF

uv run python /tmp/remove_server.py
pause

# Step 11: Final server list
echo "Step 11: Final server list (demo-time removed)"
echo "----------------------------------------------"
uv run mcp-cli servers

# Cleanup
rm -f /tmp/add_server.py /tmp/show_server.py /tmp/disable_server.py 
rm -f /tmp/enable_server.py /tmp/remove_server.py /tmp/list_all_servers.py

echo ""
echo "================================================"
echo "✅ Demo Complete!"
echo "================================================"
echo ""
echo "This demo showed:"
echo "  • Adding a new MCP server (demo-time)"
echo "  • Listing servers to see the addition"
echo "  • Showing server details"
echo "  • Disabling and re-enabling the server"
echo "  • Removing the server"
echo "  • Tool execution from existing servers"
echo ""
echo "💡 Tips:"
echo "  • New servers need session restart to fully connect"
echo "  • Configuration is saved in server_config.json"
echo "  • Use '/server' commands in chat mode for easier management"
echo ""