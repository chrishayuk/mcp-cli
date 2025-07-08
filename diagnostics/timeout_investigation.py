#!/usr/bin/env python3
"""
Debug script to find the actual source of the 10-second timeout
in MCP CLI by examining the runtime environment.
"""

import sys
import os
import inspect
import importlib

def trace_mcp_cli_imports():
    """Trace where MCP CLI components are loaded from."""
    print("🔍 Tracing MCP CLI imports and timeout sources...")
    
    try:
        # Try to import the actual MCP CLI components
        components_to_check = [
            'mcp_cli',
            'mcp_cli.tools.manager',
            'mcp_cli.run_command', 
            'chuk_tool_processor',
            'chuk_tool_processor.execution.strategies.inprocess_strategy'
        ]
        
        for component in components_to_check:
            try:
                module = importlib.import_module(component)
                file_path = getattr(module, '__file__', 'Unknown')
                print(f"✅ {component}: {file_path}")
                
                # Check for timeout-related attributes
                for attr in dir(module):
                    if 'timeout' in attr.lower():
                        value = getattr(module, attr, None)
                        print(f"   - {attr}: {value}")
                        
            except ImportError as e:
                print(f"❌ {component}: Not found ({e})")
    
    except Exception as e:
        print(f"Error during import tracing: {e}")

def check_environment_variables():
    """Check for timeout-related environment variables."""
    print("\n🌍 Environment variables:")
    timeout_vars = [
        'MCP_TOOL_TIMEOUT',
        'CHUK_TOOL_TIMEOUT', 
        'ASYNCIO_TIMEOUT',
        'TOOL_EXECUTION_TIMEOUT',
        'TIMEOUT',
        'DEFAULT_TIMEOUT'
    ]
    
    for var in timeout_vars:
        value = os.environ.get(var)
        if value:
            print(f"   {var}={value}")
        else:
            print(f"   {var}=not set")

def inspect_inprocess_strategy():
    """Inspect the InProcessStrategy class for timeout configuration."""
    print("\n🔧 Inspecting InProcessStrategy...")
    
    try:
        from chuk_tool_processor.execution.strategies.inprocess_strategy import InProcessStrategy
        
        # Get the signature of __init__
        sig = inspect.signature(InProcessStrategy.__init__)
        print(f"   InProcessStrategy.__init__ signature: {sig}")
        
        # Check default values
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                print(f"   - {param_name} default: {param.default}")
                
        # Create an instance to see actual values
        try:
            # We need a mock registry, so let's just examine the class
            print(f"   InProcessStrategy source: {inspect.getfile(InProcessStrategy)}")
            
        except Exception as e:
            print(f"   Could not create instance: {e}")
            
    except ImportError as e:
        print(f"   Could not import InProcessStrategy: {e}")

def inspect_tool_executor():
    """Inspect ToolExecutor for timeout settings."""
    print("\n⚡ Inspecting ToolExecutor...")
    
    try:
        from chuk_tool_processor.execution.tool_executor import ToolExecutor
        
        sig = inspect.signature(ToolExecutor.__init__)
        print(f"   ToolExecutor.__init__ signature: {sig}")
        
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                print(f"   - {param_name} default: {param.default}")
                
        print(f"   ToolExecutor source: {inspect.getfile(ToolExecutor)}")
        
    except ImportError as e:
        print(f"   Could not import ToolExecutor: {e}")

def find_timeout_in_source():
    """Find the actual timeout value in source code."""
    print("\n📋 Source code analysis:")
    
    try:
        from chuk_tool_processor.execution.strategies.inprocess_strategy import InProcessStrategy
        import inspect
        
        # Get the source code
        source = inspect.getsource(InProcessStrategy)
        
        # Look for timeout references
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'timeout' in line.lower() and ('10' in line or '30' in line):
                print(f"   Line {i+1}: {line.strip()}")
                
    except Exception as e:
        print(f"   Could not analyze source: {e}")

def create_runtime_patch():
    """Create a runtime patch to override the timeout."""
    print("\n🔧 Creating runtime patch...")
    
    patch_code = '''
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
'''
    
    with open('mcp_timeout_patch.py', 'w') as f:
        f.write(patch_code)
    
    print("✅ Created mcp_timeout_patch.py")
    print("   Usage: python -c 'import mcp_timeout_patch' && mcp-cli chat")

if __name__ == "__main__":
    print("🕵️ MCP CLI Timeout Detective")
    print("=" * 50)
    
    trace_mcp_cli_imports()
    check_environment_variables() 
    inspect_inprocess_strategy()
    inspect_tool_executor()
    find_timeout_in_source()
    create_runtime_patch()
    
    print("\n🎯 Next steps:")
    print("1. Run the runtime patch: python -c 'import mcp_timeout_patch' && mcp-cli chat")
    print("2. Or set environment variable: export CHUK_TOOL_TIMEOUT=300 && mcp-cli chat")
    print("3. If those don't work, the timeout might be in the MCP server itself")