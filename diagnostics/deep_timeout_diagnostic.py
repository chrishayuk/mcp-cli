#!/usr/bin/env python3
"""
Deep diagnostic script to trace timeout and retry configuration through all layers.

This script will test actual tool execution and trace through:
1. CLI configuration (server_config.json)
2. ToolManager initialization
3. HTTP transport setup
4. chuk-tool-processor ToolProcessor
5. chuk-mcp StreamManager
6. Actual tool call execution with timeout

Goal: Identify where the 120s timeout and "4 attempts" are coming from.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Configure logging to see everything
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

# Suppress some noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("chuk_mcp.transports.http.transport").setLevel(logging.INFO)
logging.getLogger("root").setLevel(logging.INFO)  # MCP client logging


class TimeoutDiagnostic:
    def __init__(self):
        self.findings = []
        self.config_path = Path("server_config.json")

    def log_finding(self, layer: str, key: str, value: any):
        """Log a finding at a specific layer."""
        finding = f"[{layer}] {key}: {value}"
        self.findings.append(finding)
        print(f"  ✓ {finding}")

    async def run_full_diagnostic(self):
        """Run complete diagnostic through all layers."""
        print("=" * 80)
        print("DEEP TIMEOUT & RETRY DIAGNOSTIC")
        print("=" * 80)
        print()

        # Layer 1: Configuration file
        await self.diagnose_config_layer()

        # Layer 2: ToolManager initialization
        await self.diagnose_tool_manager_layer()

        # Layer 3: HTTP Transport parameters
        await self.diagnose_http_transport_layer()

        # Layer 4: chuk-tool-processor setup
        await self.diagnose_tool_processor_layer()

        # Layer 5: StreamManager
        await self.diagnose_stream_manager_layer()

        # Layer 6: Actual tool execution with instrumentation
        await self.diagnose_actual_execution_layer()

        # Layer 7: Check for retry wrappers
        await self.diagnose_retry_wrappers()

        # Summary
        self.print_summary()

    async def diagnose_config_layer(self):
        """Layer 1: Check server_config.json"""
        print("\n" + "─" * 80)
        print("LAYER 1: server_config.json Configuration")
        print("─" * 80)

        if not self.config_path.exists():
            print("  ⚠️  server_config.json not found!")
            return

        with open(self.config_path) as f:
            config = json.load(f)

        monday_config = config.get("mcpServers", {}).get("monday", {})

        if not monday_config:
            print("  ⚠️  monday server not found in config!")
            return

        self.log_finding("CONFIG", "url", monday_config.get("url"))
        self.log_finding("CONFIG", "timeout", monday_config.get("timeout", "NOT SET"))
        self.log_finding(
            "CONFIG", "max_retries", monday_config.get("max_retries", "NOT SET")
        )
        self.log_finding(
            "CONFIG", "transport", monday_config.get("transport", "NOT SET")
        )

    async def diagnose_tool_manager_layer(self):
        """Layer 2: ToolManager initialization"""
        print("\n" + "─" * 80)
        print("LAYER 2: ToolManager Initialization")
        print("─" * 80)

        # Import and inspect ToolManager defaults
        from mcp_cli.tools.manager import ToolManager
        import inspect

        sig = inspect.signature(ToolManager.__init__)
        for param_name, param in sig.parameters.items():
            if "timeout" in param_name.lower() or "retry" in param_name.lower():
                default = (
                    param.default
                    if param.default != inspect.Parameter.empty
                    else "REQUIRED"
                )
                self.log_finding("ToolManager.__init__", param_name, default)

        # Check _determine_timeout method
        print("\n  Checking _determine_timeout logic:")
        tm = ToolManager(
            config_file="server_config.json",
            servers=["monday"],
            tool_timeout=None,  # Let it determine from env/defaults
        )
        self.log_finding("ToolManager", "determined timeout", tm.tool_timeout)

    async def diagnose_http_transport_layer(self):
        """Layer 3: HTTP Transport Configuration"""
        print("\n" + "─" * 80)
        print("LAYER 3: HTTP Transport Parameters")
        print("─" * 80)

        try:
            from chuk_mcp.transports.http.parameters import StreamableHTTPParameters
            import inspect

            sig = inspect.signature(StreamableHTTPParameters.__init__)
            for param_name, param in sig.parameters.items():
                if "timeout" in param_name.lower() or "retry" in param_name.lower():
                    default = (
                        param.default
                        if param.default != inspect.Parameter.empty
                        else "REQUIRED"
                    )
                    self.log_finding("StreamableHTTPParameters", param_name, default)

        except ImportError as e:
            print(f"  ⚠️  Could not import HTTP parameters: {e}")

    async def diagnose_tool_processor_layer(self):
        """Layer 4: chuk-tool-processor ToolProcessor"""
        print("\n" + "─" * 80)
        print("LAYER 4: chuk-tool-processor ToolProcessor")
        print("─" * 80)

        try:
            from chuk_tool_processor.core.processor import ToolProcessor
            import inspect

            sig = inspect.signature(ToolProcessor.__init__)
            for param_name, param in sig.parameters.items():
                if "timeout" in param_name.lower() or "retry" in param_name.lower():
                    default = (
                        param.default
                        if param.default != inspect.Parameter.empty
                        else "REQUIRED"
                    )
                    self.log_finding("ToolProcessor.__init__", param_name, default)

        except ImportError as e:
            print(f"  ⚠️  Could not import ToolProcessor: {e}")

    async def diagnose_stream_manager_layer(self):
        """Layer 5: StreamManager"""
        print("\n" + "─" * 80)
        print("LAYER 5: chuk-tool-processor StreamManager")
        print("─" * 80)

        try:
            from chuk_tool_processor.mcp.stream_manager import StreamManager
            import inspect

            # Check call_tool signature
            sig = inspect.signature(StreamManager.call_tool)
            self.log_finding("StreamManager.call_tool", "signature", str(sig))

            # Check if there are timeout parameters
            for param_name, param in sig.parameters.items():
                if "timeout" in param_name.lower():
                    default = (
                        param.default
                        if param.default != inspect.Parameter.empty
                        else "REQUIRED"
                    )
                    self.log_finding("StreamManager.call_tool", param_name, default)

        except ImportError as e:
            print(f"  ⚠️  Could not import StreamManager: {e}")

    async def diagnose_actual_execution_layer(self):
        """Layer 6: Actual tool execution with instrumentation"""
        print("\n" + "─" * 80)
        print("LAYER 6: Actual Tool Execution (Instrumented)")
        print("─" * 80)

        print("\n  Setting up ToolManager for real execution test...")

        from mcp_cli.tools.manager import ToolManager

        # Create tool manager with monday server
        tm = ToolManager(
            config_file="server_config.json",
            servers=["monday"],
            tool_timeout=None,  # Let it use config value
        )

        self.log_finding("EXEC", "ToolManager.tool_timeout", tm.tool_timeout)

        # Initialize
        print("\n  Initializing ToolManager...")
        success = await tm.initialize()

        if not success:
            print("  ❌ ToolManager initialization failed!")
            return

        self.log_finding("EXEC", "initialization", "SUCCESS")
        self.log_finding("EXEC", "_effective_timeout", tm._effective_timeout)
        self.log_finding("EXEC", "_effective_max_retries", tm._effective_max_retries)

        # Check processor configuration
        if tm.processor:
            self.log_finding(
                "EXEC",
                "processor.default_timeout",
                getattr(tm.processor, "default_timeout", "NOT SET"),
            )
            self.log_finding(
                "EXEC",
                "processor.max_retries",
                getattr(tm.processor, "max_retries", "NOT SET"),
            )

        # Check executor configuration
        if tm._executor:
            self.log_finding(
                "EXEC",
                "executor.default_timeout",
                getattr(tm._executor, "default_timeout", "NOT SET"),
            )
            if hasattr(tm._executor, "strategy"):
                self.log_finding(
                    "EXEC",
                    "executor.strategy.default_timeout",
                    getattr(tm._executor.strategy, "default_timeout", "NOT SET"),
                )

        # Check stream_manager configuration
        if tm.stream_manager:
            self.log_finding(
                "EXEC", "stream_manager", f"{type(tm.stream_manager).__name__}"
            )

        # Now try to execute a simple tool with detailed timing
        print("\n  Attempting actual tool execution...")
        print("  Tool: list_workspaces")
        print("  Arguments: {limit: 100}")
        print("  NOTE: Execution may take up to 120s to timeout...")
        print()

        import time

        start_time = time.time()

        try:
            # Execute WITHOUT explicit timeout to see what happens naturally
            # This should respect the configured timeout (300s) but we expect it might use 120s
            result = await tm.execute_tool("list_workspaces", {"limit": 100})

            elapsed = time.time() - start_time

            self.log_finding("EXEC_RESULT", "elapsed_time", f"{elapsed:.2f}s")
            self.log_finding("EXEC_RESULT", "success", result.success)

            if not result.success:
                self.log_finding("EXEC_RESULT", "error", result.error)

                # Parse the error to find timeout and retry info
                error_str = str(result.error)
                if "timed out after" in error_str:
                    import re

                    timeout_match = re.search(r"timed out after ([\d.]+)s", error_str)
                    if timeout_match:
                        self.log_finding(
                            "EXEC_ERROR", "timeout_value", f"{timeout_match.group(1)}s"
                        )

                if "failed after" in error_str:
                    import re

                    retry_match = re.search(r"failed after (\d+) attempts", error_str)
                    if retry_match:
                        self.log_finding(
                            "EXEC_ERROR", "retry_attempts", retry_match.group(1)
                        )

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"  ⚠️  Tool execution timed out after {elapsed:.2f}s")
            self.log_finding("EXEC_TIMEOUT", "elapsed", f"{elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ❌ Exception after {elapsed:.2f}s: {e}")
            self.log_finding("EXEC_EXCEPTION", "error", str(e))
            self.log_finding("EXEC_EXCEPTION", "elapsed", f"{elapsed:.2f}s")

        finally:
            await tm.close()

    async def diagnose_retry_wrappers(self):
        """Layer 7: Check for retry wrappers in the execution chain"""
        print("\n" + "─" * 80)
        print("LAYER 7: Retry Wrapper Detection")
        print("─" * 80)

        try:
            from chuk_tool_processor.execution.wrappers.retry import RetryWrapper
            import inspect

            sig = inspect.signature(RetryWrapper.__init__)
            for param_name, param in sig.parameters.items():
                if "retry" in param_name.lower():
                    default = (
                        param.default
                        if param.default != inspect.Parameter.empty
                        else "REQUIRED"
                    )
                    self.log_finding("RetryWrapper.__init__", param_name, default)

        except ImportError:
            print("  ℹ️  No RetryWrapper found in chuk-tool-processor")

        # Check if there's retry logic in the HTTP transport
        try:
            from chuk_mcp.transports.http.transport import StreamableHTTPTransport

            # Check if there's a retry method
            if hasattr(StreamableHTTPTransport, "_retry_request"):
                print("  ✓ Found _retry_request method in StreamableHTTPTransport")
            else:
                print("  ℹ️  No _retry_request method in StreamableHTTPTransport")

        except ImportError:
            print("  ⚠️  Could not import StreamableHTTPTransport")

    def print_summary(self):
        """Print diagnostic summary"""
        print("\n" + "=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)
        print("\nAll findings:")
        for finding in self.findings:
            print(f"  {finding}")

        print("\n" + "=" * 80)
        print("\nKey Issues to Investigate:")
        print("  1. Where does the 120s timeout come from?")
        print("  2. Where does the '4 attempts' retry logic come from?")
        print("  3. Is the server config timeout=300 being ignored?")
        print("  4. Is max_retries=0 being ignored?")
        print("=" * 80)


async def main():
    diagnostic = TimeoutDiagnostic()
    await diagnostic.run_full_diagnostic()


if __name__ == "__main__":
    asyncio.run(main())
