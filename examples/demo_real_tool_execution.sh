#!/bin/bash
# Demo script showing real tool execution with MCP servers

echo "=== MCP-CLI Tool Execution Demo with Real Servers ==="
echo ""
echo "This demo shows tool execution with actual MCP servers."
echo ""

# Demo with echo server
echo "1. Starting interactive mode with echo server..."
echo "   Command: mcp-cli interactive --server echo"
echo ""

# Create a script with commands to run
cat << 'EOF' > /tmp/interactive_commands.txt
# List available tools
execute

# Show echo tool details
execute echo

# Execute echo tool
execute echo '{"message": "Hello from MCP-CLI!"}'

# Execute using alias
exec echo '{"message": "Using the exec alias!"}'

# Exit
exit
EOF

echo "2. Running interactive commands:"
echo "   - List tools"
echo "   - Show echo tool details" 
echo "   - Execute echo with message"
echo "   - Use exec alias"
echo ""

# Run the commands
uv run mcp-cli interactive --server echo < /tmp/interactive_commands.txt

echo ""
echo "=== Demo Complete ==="
echo ""
echo "Key features demonstrated:"
echo "✓ Listed available tools from MCP server"
echo "✓ Showed tool parameter details"
echo "✓ Executed tool with JSON parameters"
echo "✓ Used command aliases (exec)"
echo ""
echo "Try it yourself:"
echo "  mcp-cli interactive --server echo"
echo "  > execute"
echo "  > execute echo '{\"message\": \"Your message here\"}'"
echo ""

# Clean up
rm -f /tmp/interactive_commands.txt