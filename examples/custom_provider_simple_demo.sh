#!/bin/bash
# Simple working demo of custom provider management.
# Shows the complete workflow with actual working commands.

echo "=========================================="
echo "🚀 Custom Provider Working Demo"
echo "=========================================="
echo ""

# Load environment if .env exists
if [ -f .env ]; then
    source .env
fi

# Check for OpenAI key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not found"
    echo "Please set OPENAI_API_KEY in your .env file"
    exit 1
fi

echo "✅ Found OPENAI_API_KEY"
echo ""

# Step 1: Add custom provider
echo "Step 1: Adding custom OpenAI provider"
echo "--------------------------------------"
uv run mcp-cli provider add my-ai "https://api.openai.com/v1" gpt-4o-mini
echo ""

# Step 2: Set environment variable
echo "Step 2: Setting API key"
echo "------------------------"
export MY_AI_API_KEY=$OPENAI_API_KEY
echo "✅ Set MY_AI_API_KEY from OPENAI_API_KEY"
echo ""

# Step 3: Show configuration
echo "Step 3: Verify configuration"
echo "-----------------------------"
uv run mcp-cli provider custom
echo ""

# Step 4: Show in provider list
echo "Step 4: Check provider status"
echo "------------------------------"
uv run mcp-cli providers | grep -A1 -B1 my-ai
echo ""

# Step 5: Test with Python script
echo "Step 5: Test actual inference"
echo "------------------------------"
cat > /tmp/test_custom_provider.py << 'EOF'
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.model_manager import ModelManager

# Create model manager (will load custom providers from preferences)
model_manager = ModelManager()

# Get client for custom provider
try:
    client = model_manager._get_custom_provider_client("my-ai", "gpt-4o-mini")
    
    # Make API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        max_tokens=10
    )
    
    print(f"✅ Inference successful!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ Error: {e}")
EOF

cd "$(dirname "$0")/.."
python3 /tmp/test_custom_provider.py
rm /tmp/test_custom_provider.py
echo ""

# Step 6: Clean up
echo "Step 6: Remove custom provider"
echo "-------------------------------"
uv run mcp-cli provider remove my-ai
echo ""

echo "Step 7: Verify removal"
echo "-----------------------"
uv run mcp-cli provider custom
echo ""

echo "=========================================="
echo "✨ Demo Complete!"
echo "=========================================="
echo ""
echo "Key points demonstrated:"
echo "  ✅ Added custom OpenAI-compatible provider"
echo "  ✅ Set API key via environment variable"
echo "  ✅ Verified configuration"
echo "  ✅ Performed actual inference"
echo "  ✅ Removed provider"
echo ""
echo "The pattern works for any OpenAI-compatible API:"
echo "  • LocalAI (http://localhost:8080/v1)"
echo "  • LM Studio (http://localhost:1234/v1)"
echo "  • Corporate proxies"
echo "  • Any OpenAI-compatible endpoint"