#!/bin/bash
# MCP CLI Command Mode - LLM Integration Demo
# Demonstrates prompt-based processing with various LLM providers

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "MCP CLI - Command Mode LLM Demo"
echo "Prompt-based processing and LLM integration"
echo "════════════════════════════════════════════════════════════════"
echo

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

demo_section() {
    echo
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
}

show_command() {
    echo -e "${YELLOW}$ $1${NC}"
}

run_if_configured() {
    local provider=$1
    local env_var=$2
    shift 2
    local cmd="$@"

    if [ -n "${!env_var}" ]; then
        echo -e "${GREEN}✓ $provider configured${NC}"
        show_command "$cmd"
        eval "$cmd"
        echo
    else
        echo -e "${RED}✗ $provider not configured (set $env_var)${NC}"
        show_command "$cmd"
        echo "   [Skipped - requires API key]"
        echo
    fi
}

# ═══════════════════════════════════════════════════════════════════
# Prerequisites Check
# ═══════════════════════════════════════════════════════════════════
demo_section "Prerequisites Check"

echo "Checking for available LLM providers..."
echo

providers=(
    "Ollama:LOCAL:ollama"
    "OpenAI:OPENAI_API_KEY:openai"
    "Anthropic:ANTHROPIC_API_KEY:anthropic"
    "Groq:GROQ_API_KEY:groq"
)

available_providers=()

for provider_info in "${providers[@]}"; do
    IFS=':' read -r name env_var provider_id <<< "$provider_info"

    if [ "$env_var" = "LOCAL" ]; then
        # Check if Ollama is running
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo -e "${GREEN}✓ $name - Running locally${NC}"
            available_providers+=("$provider_id")
        else
            echo -e "${YELLOW}○ $name - Not running (start with: ollama serve)${NC}"
        fi
    else
        if [ -n "${!env_var}" ]; then
            echo -e "${GREEN}✓ $name - Configured${NC}"
            available_providers+=("$provider_id")
        else
            echo -e "${RED}✗ $name - Not configured (set $env_var)${NC}"
        fi
    fi
done

if [ ${#available_providers[@]} -eq 0 ]; then
    echo
    echo -e "${RED}No LLM providers configured!${NC}"
    echo
    echo "To run this demo, configure at least one provider:"
    echo "  • Ollama (local):  ollama serve"
    echo "  • OpenAI:          export OPENAI_API_KEY=sk-..."
    echo "  • Anthropic:       export ANTHROPIC_API_KEY=sk-ant-..."
    echo "  • Groq:            export GROQ_API_KEY=gsk_..."
    echo
    exit 1
fi

echo
echo -e "${GREEN}Found ${#available_providers[@]} available provider(s)${NC}"

# ═══════════════════════════════════════════════════════════════════
# Section 1: Simple Text Processing
# ═══════════════════════════════════════════════════════════════════
demo_section "1. Simple Text Processing with --prompt"

echo "Process text directly with LLM using --prompt option:"
echo

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    show_command "uv run mcp-cli cmd --prompt 'What is 2 + 2? Reply with just the number.' --single-turn --raw --provider ollama --model llama3.2"
    echo "Expected output: 4"
    echo -n "Result: "
    uv run mcp-cli cmd --prompt 'What is 2 + 2? Reply with just the number.' --single-turn --raw --provider ollama --model llama3.2 2>/dev/null | tail -1 || echo "[Demo skipped]"
    echo
fi

run_if_configured "OpenAI" "OPENAI_API_KEY" \
    "echo 'Calculate: 15 * 7' | uv run mcp-cli cmd --input - --single-turn --raw --provider openai --model gpt-4o-mini 2>/dev/null | head -3"

# ═══════════════════════════════════════════════════════════════════
# Section 2: File-based Processing
# ═══════════════════════════════════════════════════════════════════
demo_section "2. File-based Processing with --input"

echo "Process files and generate outputs:"
echo

# Create sample data file
cat > /tmp/mcp_sample_data.txt << 'EOF'
Product Sales Data:
- Widget A: $1200
- Widget B: $850
- Widget C: $2100
- Widget D: $450
Total Items: 4
EOF

echo -e "${GREEN}Created sample file: /tmp/mcp_sample_data.txt${NC}"
cat /tmp/mcp_sample_data.txt
echo

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    show_command "uv run mcp-cli cmd --input /tmp/mcp_sample_data.txt --prompt 'Calculate the total sales. Reply with just the dollar amount.' --single-turn --raw --provider ollama --model llama3.2"
    echo -n "Result: "
    uv run mcp-cli cmd --input /tmp/mcp_sample_data.txt --prompt 'Calculate the total sales. Reply with just the dollar amount.' --single-turn --raw --provider ollama --model llama3.2 2>/dev/null | tail -1 || echo "[Demo skipped]"
    echo
fi

# ═══════════════════════════════════════════════════════════════════
# Section 3: Input/Output Files
# ═══════════════════════════════════════════════════════════════════
demo_section "3. File Input/Output with --output"

echo "Save LLM output directly to files:"
echo

cat > /tmp/mcp_report_data.txt << 'EOF'
Q4 2024 Summary:
- Revenue increased 23%
- New customers: 1,234
- Customer satisfaction: 4.5/5
- Market share: 18%
EOF

echo -e "${GREEN}Input file: /tmp/mcp_report_data.txt${NC}"
cat /tmp/mcp_report_data.txt
echo

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    show_command "uv run mcp-cli cmd --input /tmp/mcp_report_data.txt --output /tmp/mcp_summary.txt --prompt 'Write a 2-sentence executive summary' --single-turn --provider ollama --model llama3.2"
    uv run mcp-cli cmd --input /tmp/mcp_report_data.txt --output /tmp/mcp_summary.txt --prompt 'Write a 2-sentence executive summary' --single-turn --provider ollama --model llama3.2 2>/dev/null || echo "[Demo skipped]"

    if [ -f /tmp/mcp_summary.txt ]; then
        echo -e "${GREEN}Output saved to: /tmp/mcp_summary.txt${NC}"
        cat /tmp/mcp_summary.txt
        echo
    fi
fi

# ═══════════════════════════════════════════════════════════════════
# Section 4: Pipeline Processing (stdin/stdout)
# ═══════════════════════════════════════════════════════════════════
demo_section "4. Unix Pipeline Integration"

echo "Use - for stdin/stdout in Unix pipelines:"
echo

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    show_command "echo 'The quick brown fox' | uv run mcp-cli cmd --input - --output - --prompt 'Count the words. Reply with just the number.' --single-turn --raw --provider ollama --model llama3.2"
    echo -n "Result: "
    echo 'The quick brown fox' | uv run mcp-cli cmd --input - --output - --prompt 'Count the words. Reply with just the number.' --single-turn --raw --provider ollama --model llama3.2 2>/dev/null | tail -1 || echo "[Demo skipped]"
    echo
fi

# ═══════════════════════════════════════════════════════════════════
# Section 5: Multi-Provider Comparison
# ═══════════════════════════════════════════════════════════════════
demo_section "5. Multi-Provider Comparison"

echo "Compare responses across different providers:"
echo

QUESTION="What is the capital of France? Reply with just the city name."

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${YELLOW}Ollama (llama3.2):${NC}"
    show_command "uv run mcp-cli cmd --prompt '$QUESTION' --single-turn --raw --provider ollama --model llama3.2"
    echo -n "Result: "
    uv run mcp-cli cmd --prompt "$QUESTION" --single-turn --raw --provider ollama --model llama3.2 2>/dev/null | tail -1 || echo "[Skipped]"
    echo
fi

run_if_configured "OpenAI" "OPENAI_API_KEY" \
    "uv run mcp-cli cmd --prompt '$QUESTION' --single-turn --raw --provider openai --model gpt-4o-mini 2>/dev/null | head -1"

run_if_configured "Anthropic" "ANTHROPIC_API_KEY" \
    "uv run mcp-cli cmd --prompt '$QUESTION' --single-turn --raw --provider anthropic --model claude-3-5-sonnet-20241022 2>/dev/null | head -1"

run_if_configured "Groq" "GROQ_API_KEY" \
    "uv run mcp-cli cmd --prompt '$QUESTION' --single-turn --raw --provider groq --model llama-3.1-8b-instant 2>/dev/null | head -1"

# ═══════════════════════════════════════════════════════════════════
# Section 6: Batch Processing with LLM
# ═══════════════════════════════════════════════════════════════════
demo_section "6. Batch Processing Multiple Files"

echo "Process multiple files with LLM in a loop:"
echo

# Create sample files
mkdir -p /tmp/mcp_llm_batch
for i in 1 2 3; do
    cat > "/tmp/mcp_llm_batch/review$i.txt" << EOF
Review $i:
Product quality: Excellent
Delivery: On time
Rating: ${i}/5 stars
EOF
done

echo -e "${GREEN}Created 3 sample review files${NC}"
ls -1 /tmp/mcp_llm_batch/
echo

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Processing each file..."
    for file in /tmp/mcp_llm_batch/*.txt; do
        basename=$(basename "$file" .txt)
        echo "  • $basename"
        show_command "    uv run mcp-cli cmd --input '$file' --prompt 'Summarize in 5 words' --single-turn --raw --provider ollama --model llama3.2"
        echo -n "    Result: "
        result=$(uv run mcp-cli cmd --input "$file" --prompt 'Summarize in 5 words' --single-turn --raw --provider ollama --model llama3.2 2>/dev/null | tail -1 || echo "[Skipped]")
        echo "$result"
    done
    echo
fi

# ═══════════════════════════════════════════════════════════════════
# Section 7: Different Output Modes
# ═══════════════════════════════════════════════════════════════════
demo_section "7. Output Modes: Raw vs Formatted"

echo "Compare raw (--raw) vs formatted output:"
echo

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${YELLOW}Raw output (--raw) for scripting:${NC}"
    show_command "uv run mcp-cli cmd --prompt 'Say hello' --single-turn --raw --provider ollama --model llama3.2"
    echo -n "Result: "
    uv run mcp-cli cmd --prompt 'Say hello' --single-turn --raw --provider ollama --model llama3.2 2>/dev/null | tail -1 || echo "[Skipped]"
    echo

    echo -e "${YELLOW}Formatted output (default) for humans:${NC}"
    show_command "uv run mcp-cli cmd --prompt 'Say hello' --single-turn --provider ollama --model llama3.2"
    uv run mcp-cli cmd --prompt 'Say hello' --single-turn --provider ollama --model llama3.2 2>/dev/null | tail -2 || echo "[Skipped]"
    echo
fi

# ═══════════════════════════════════════════════════════════════════
# Section 8: Combined Tool + LLM Mode
# ═══════════════════════════════════════════════════════════════════
demo_section "8. Combined: Direct Tool vs LLM Processing"

echo "Compare direct tool execution vs LLM-assisted processing:"
echo

echo -e "${YELLOW}Direct tool execution (no LLM):${NC}"
show_command "uv run mcp-cli cmd --server sqlite --tool list_tables --raw"
echo "  [Fast, deterministic, no API key needed]"
echo

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${YELLOW}LLM-assisted processing (with prompt):${NC}"
    show_command "uv run mcp-cli cmd --prompt 'Analyze data and provide insights' --input data.txt --provider ollama"
    echo "  [Intelligent analysis, requires LLM]"
    echo
fi

# ═══════════════════════════════════════════════════════════════════
# Section 9: System Prompts
# ═══════════════════════════════════════════════════════════════════
demo_section "9. Custom System Prompts"

echo "Use --system-prompt to set behavior:"
echo

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    show_command "uv run mcp-cli cmd --system-prompt 'You are a pirate. Always respond like a pirate.' --prompt 'What is the weather like?' --single-turn --raw --provider ollama --model llama3.2"
    echo "Result:"
    uv run mcp-cli cmd --system-prompt 'You are a pirate. Always respond like a pirate.' --prompt 'What is the weather like?' --single-turn --raw --provider ollama --model llama3.2 2>/dev/null | tail -1 || echo "[Skipped]"
    echo
fi

# ═══════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════
demo_section "Cleanup"

echo "Cleaning up demo files..."
rm -f /tmp/mcp_sample_data.txt
rm -f /tmp/mcp_report_data.txt
rm -f /tmp/mcp_summary.txt
rm -rf /tmp/mcp_llm_batch
echo -e "${GREEN}Demo files cleaned up${NC}"
echo

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
demo_section "Summary"

cat << 'SUMMARY'
LLM Integration Features:

1. Simple Text Processing
   • --prompt: Direct text processing
   • --single-turn: Disable tool calls
   • --raw: Machine-readable output

2. File Processing
   • --input FILE: Process file content
   • --output FILE: Save LLM response
   • Supports any text format

3. Pipeline Integration
   • --input -: Read from stdin
   • --output -: Write to stdout
   • Unix-friendly automation

4. Multi-Provider Support
   • Ollama (local, free)
   • OpenAI (GPT-4, GPT-5)
   • Anthropic (Claude)
   • Groq (fast inference)

5. Batch Processing
   • Loop over files
   • Collect results
   • Parallel processing ready

6. Output Control
   • --raw: For scripts
   • Default: For humans
   • --theme: UI customization

7. System Prompts
   • --system-prompt: Set behavior
   • Custom personas
   • Role-based processing

Two Modes Compared:

┌─────────────────┬────────────────────┬─────────────────────┐
│ Feature         │ Direct Tool        │ LLM Prompt Mode     │
├─────────────────┼────────────────────┼─────────────────────┤
│ Speed           │ Fast               │ Slower (API calls)  │
│ Cost            │ Free               │ May have costs      │
│ Intelligence    │ Deterministic      │ Reasoning/Analysis  │
│ Use Case        │ Data extraction    │ Text processing     │
│ API Key         │ Not needed         │ Required (or local) │
│ Example         │ --tool list_tables │ --prompt "Analyze"  │
└─────────────────┴────────────────────┴─────────────────────┘

Quick Reference:

# Direct tool execution
mcp-cli cmd --tool TOOL [--tool-args JSON] --raw

# LLM text processing
mcp-cli cmd --prompt "TEXT" [--input FILE] [--output FILE]

# Pipeline processing
cat data.txt | mcp-cli cmd --input - --prompt "Analyze" --output -

# Batch processing
for f in *.txt; do
  mcp-cli cmd --input "$f" --prompt "Summarize" --output "$f.summary"
done

# Multi-provider
mcp-cli cmd --prompt "TEXT" --provider ollama --model llama3.2
mcp-cli cmd --prompt "TEXT" --provider openai --model gpt-4o-mini
mcp-cli cmd --prompt "TEXT" --provider anthropic --model claude-3-5-sonnet

Setup Required:
  • Ollama: ollama serve && ollama pull llama3.2
  • OpenAI: export OPENAI_API_KEY=sk-...
  • Anthropic: export ANTHROPIC_API_KEY=sk-ant-...
  • Groq: export GROQ_API_KEY=gsk_...
SUMMARY

echo
echo "════════════════════════════════════════════════════════════════"
echo "LLM Demo Complete!"
echo "════════════════════════════════════════════════════════════════"
