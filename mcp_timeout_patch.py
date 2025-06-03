
# Runtime patch for MCP CLI timeout
import os

# Set environment variables that might be checked
os.environ["MCP_TOOL_TIMEOUT"] = "300"
os.environ["CHUK_TOOL_TIMEOUT"] = "300"
os.environ["DEFAULT_TIMEOUT"] = "300"

# Try to monkey patch common timeout locations
try:
    from chuk_tool_processor.execution.strategies.inprocess_strategy import InProcessStrategy
    
    # Store original __init__
    _original_init = InProcessStrategy.__init__
    
    def patched_init(self, registry, max_concurrency=None, default_timeout=300.0, **kwargs):
        print(f"🔧 InProcessStrategy patched: timeout={default_timeout}")
        return _original_init(self, registry, max_concurrency=max_concurrency, 
                             default_timeout=default_timeout, **kwargs)
    
    InProcessStrategy.__init__ = patched_init
    print("✅ Successfully patched InProcessStrategy")
    
except Exception as e:
    print(f"❌ Failed to patch InProcessStrategy: {e}")

# Save this as patch.py and import it before running MCP CLI
