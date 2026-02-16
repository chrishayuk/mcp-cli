#!/bin/bash
# MCP CLI Command Mode Demo
# Demonstrates Unix-friendly automation capabilities

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "MCP CLI - Command Mode Demo"
echo "Unix-friendly automation and scripting capabilities"
echo "════════════════════════════════════════════════════════════════"
echo

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

demo_section() {
    echo
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
}

run_command() {
    echo -e "${YELLOW}$ $1${NC}"
    eval "$1"
    echo
}

# ═══════════════════════════════════════════════════════════════════
# Section 1: Direct Tool Execution (No LLM)
# ═══════════════════════════════════════════════════════════════════
demo_section "1. Direct Tool Execution (No LLM Required)"

echo "Execute tools directly for scripting without LLM interaction:"
echo

run_command "uv run mcp-cli cmd --server sqlite --tool list_tables --raw"

echo "With tool arguments:"
run_command 'uv run mcp-cli cmd --server sqlite --tool read_query --tool-args '"'"'{"query": "SELECT name FROM sqlite_master WHERE type='"'"'"'"'"'table'"'"'"'"'"' LIMIT 3"}'"'"' --raw'

# ═══════════════════════════════════════════════════════════════════
# Section 2: Pipeline Processing
# ═══════════════════════════════════════════════════════════════════
demo_section "2. Unix Pipeline Integration"

echo "Command mode works seamlessly with Unix pipes:"
echo

# Create temp file
echo "Sample data for analysis" > /tmp/mcp_demo_input.txt

run_command "cat /tmp/mcp_demo_input.txt | uv run mcp-cli cmd --input - --prompt 'Echo this text back' --single-turn --raw 2>/dev/null || echo 'Note: Requires API key for LLM mode'"

# ═══════════════════════════════════════════════════════════════════
# Section 3: File I/O
# ═══════════════════════════════════════════════════════════════════
demo_section "3. File Input/Output"

echo "Process files and save results:"
echo

# Create input file
cat > /tmp/mcp_sample.txt << 'EOF'
Database Analysis Request:
- List all tables
- Show structure of first table
- Count total records
EOF

echo -e "${GREEN}Input file created: /tmp/mcp_sample.txt${NC}"
cat /tmp/mcp_sample.txt
echo

run_command "uv run mcp-cli cmd --server sqlite --tool list_tables --output /tmp/mcp_output.json --raw"

if [ -f /tmp/mcp_output.json ]; then
    echo -e "${GREEN}Output saved to: /tmp/mcp_output.json${NC}"
    echo "Contents:"
    cat /tmp/mcp_output.json
    echo
fi

# ═══════════════════════════════════════════════════════════════════
# Section 4: Batch Processing
# ═══════════════════════════════════════════════════════════════════
demo_section "4. Batch Processing Example"

echo "Process multiple files in a loop:"
echo

# Create sample files
mkdir -p /tmp/mcp_batch_demo
for i in 1 2 3; do
    echo "Data file $i: Important information" > "/tmp/mcp_batch_demo/file$i.txt"
done

echo -e "${GREEN}Created sample files:${NC}"
ls -1 /tmp/mcp_batch_demo/
echo

echo -e "${YELLOW}# Batch processing script:${NC}"
cat << 'SCRIPT'
for file in /tmp/mcp_batch_demo/*.txt; do
    echo "Processing: $file"
    uv run mcp-cli cmd --server sqlite --tool list_tables --raw > "${file}.result" 2>/dev/null || echo "Note: Tool execution depends on server availability"
done
SCRIPT
echo

# Actually run it
for file in /tmp/mcp_batch_demo/*.txt; do
    echo "Processing: $file"
    uv run mcp-cli cmd --server sqlite --tool list_tables --raw > "${file}.result" 2>/dev/null || echo "  Note: Tool execution depends on server availability"
done

echo -e "${GREEN}Results created:${NC}"
ls -1 /tmp/mcp_batch_demo/*.result 2>/dev/null || echo "No results generated"
echo

# ═══════════════════════════════════════════════════════════════════
# Section 5: Raw vs Formatted Output
# ═══════════════════════════════════════════════════════════════════
demo_section "5. Output Formatting"

echo "Raw output (--raw) for scripting:"
run_command "uv run mcp-cli cmd --server sqlite --tool list_tables --raw 2>&1 | head -5"

echo "Formatted output (default) for humans:"
run_command "uv run mcp-cli cmd --server sqlite --tool list_tables 2>&1 | head -10"

# ═══════════════════════════════════════════════════════════════════
# Section 6: Error Handling
# ═══════════════════════════════════════════════════════════════════
demo_section "6. Error Handling in Scripts"

echo "Command mode provides clear error messages for scripting:"
echo

run_command "uv run mcp-cli cmd --tool nonexistent_tool --raw 2>&1 | grep -i error || true"

# ═══════════════════════════════════════════════════════════════════
# Section 7: Common Patterns
# ═══════════════════════════════════════════════════════════════════
demo_section "7. Common Automation Patterns"

echo "Pattern 1: Extract, Transform, Output"
echo -e "${YELLOW}$ uv run mcp-cli cmd --tool read_query --tool-args '{...}' --raw | jq '.results'${NC}"
echo

echo "Pattern 2: Conditional Processing"
cat << 'SCRIPT'
if uv run mcp-cli cmd --tool list_tables --raw | grep -q "users"; then
    echo "Users table exists"
    uv run mcp-cli cmd --tool read_query --tool-args '{"query": "SELECT COUNT(*) FROM users"}'
fi
SCRIPT
echo

echo "Pattern 3: Parallel Processing with GNU Parallel"
echo -e "${YELLOW}$ ls *.txt | parallel uv run mcp-cli cmd --input {} --output {.}.summary${NC}"
echo

echo "Pattern 4: Integration with other tools"
echo -e "${YELLOW}$ uv run mcp-cli cmd --tool get_data --raw | jq '.' | sqlite3 output.db${NC}"
echo

# ═══════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════
demo_section "Cleanup"

echo "Cleaning up demo files..."
rm -f /tmp/mcp_demo_input.txt
rm -f /tmp/mcp_sample.txt
rm -f /tmp/mcp_output.json
rm -rf /tmp/mcp_batch_demo
echo -e "${GREEN}Demo files cleaned up${NC}"
echo

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
demo_section "Summary"

cat << 'SUMMARY'
Command Mode Features:
  ✓ Direct tool execution (no LLM required)
  ✓ Unix pipeline integration (stdin/stdout)
  ✓ File I/O with --input and --output
  ✓ Raw output mode for scripting
  ✓ Batch processing support
  ✓ Error handling for automation
  ✓ Theme support for formatted output

Key Options:
  --tool TOOL          Execute tool directly
  --tool-args JSON     Tool arguments
  --input FILE         Input file (- for stdin)
  --output FILE        Output file (- for stdout)
  --prompt TEXT        Prompt for LLM mode
  --raw                Raw output (no formatting)
  --single-turn        Disable multi-turn conversation
  --server SERVER      Server to connect to
  --provider PROVIDER  LLM provider (for prompt mode)
  --model MODEL        LLM model (for prompt mode)
  --theme THEME        UI theme for formatted output

Example Scripts:
  1. Direct tool:     mcp-cli cmd --tool list_tables --raw
  2. With arguments:  mcp-cli cmd --tool read_query --tool-args '{"query": "..."}'
  3. Pipeline:        cat data.txt | mcp-cli cmd --input - --output -
  4. File processing: mcp-cli cmd --input data.txt --output result.json
  5. Batch:           for f in *.txt; do mcp-cli cmd --input "$f" --output "$f.result"; done

For more examples, see the README.md
SUMMARY

echo
echo "════════════════════════════════════════════════════════════════"
echo "Demo Complete!"
echo "════════════════════════════════════════════════════════════════"
