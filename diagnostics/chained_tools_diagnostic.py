#!/usr/bin/env python3
"""
Chained Tool Call Diagnostic - Reproduces the exact CLI scenario

This script simulates the exact tool call chain that happens when a user asks:
"select top 10 products from the database"

The LLM typically makes these chained calls:
1. list_tables (to discover table structure)
2. read_query (to execute the actual SELECT query)

This will help us diagnose where the "Missing required parameter 'query'" error
occurs in the streaming tool call processing chain.
"""

import asyncio
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    rprint = print


@dataclass
class ChainedToolCallTrace:
    """Trace for a chained tool call scenario."""
    step: int
    tool_name: str
    arguments: Dict[str, Any]
    depends_on: Optional[str] = None
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    streaming_artifacts: Optional[Dict[str, Any]] = None


class ChainedToolCallDiagnostic:
    """Diagnostic tool for chained tool call scenarios."""
    
    def __init__(self):
        self.console = Console() if HAS_RICH else None
        self.traces: List[ChainedToolCallTrace] = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('chained_tool_diagnostic.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def diagnose_chained_scenario(self, config_file: str, servers: List[str]) -> bool:
        """
        Diagnose the exact chained tool call scenario from the CLI.
        
        This reproduces what happens when user asks: "select top 10 products from the database"
        """
        self.log_info("🔍 Starting chained tool call diagnostic")
        self.log_info("Simulating: 'select top 10 products from the database'")
        
        # Import MCP components
        try:
            from mcp_cli.tools.manager import ToolManager
            from mcp_cli.chat.chat_context import ChatContext
            from mcp_cli.model_manager import ModelManager
        except ImportError as e:
            self.log_error(f"Failed to import MCP CLI components: {e}")
            return False
        
        # Initialize ToolManager
        tool_manager = None
        try:
            tool_manager = ToolManager(config_file, servers)
            success = await tool_manager.initialize()
            if not success:
                self.log_error("ToolManager initialization failed")
                return False
            
            self.log_success("ToolManager initialized successfully")
            
            # Test the exact chained scenario
            await self._test_chained_tool_calls(tool_manager)
            
            # Test streaming simulation
            await self._test_streaming_chain_simulation(tool_manager)
            
            # Analyze results
            self._analyze_chained_results()
            
            return True
            
        except Exception as e:
            self.log_error(f"Diagnostic error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            if tool_manager:
                await tool_manager.close()
    
    async def _test_chained_tool_calls(self, tool_manager):
        """Test the actual chained tool call scenario."""
        self.log_info("🔗 Testing chained tool calls (direct execution)")
        
        # Step 1: list_tables (what LLM does first)
        step1_trace = ChainedToolCallTrace(
            step=1,
            tool_name="list_tables",
            arguments={},
            depends_on=None
        )
        
        start_time = time.time()
        try:
            result = await tool_manager.execute_tool("list_tables", {})
            step1_trace.execution_time = time.time() - start_time
            step1_trace.success = result.success
            step1_trace.result = result.result
            step1_trace.error = result.error
            
            if result.success:
                self.log_success(f"Step 1 SUCCESS: {result.result}")
            else:
                self.log_error(f"Step 1 FAILED: {result.error}")
                
        except Exception as e:
            step1_trace.error = str(e)
            self.log_error(f"Step 1 EXCEPTION: {e}")
        
        self.traces.append(step1_trace)
        
        # Step 2: read_query (what LLM does next with table info)
        # This is where the error typically occurs
        query = "SELECT * FROM products LIMIT 10"
        step2_trace = ChainedToolCallTrace(
            step=2,
            tool_name="read_query",
            arguments={"query": query},
            depends_on="list_tables"
        )
        
        start_time = time.time()
        try:
            result = await tool_manager.execute_tool("read_query", {"query": query})
            step2_trace.execution_time = time.time() - start_time
            step2_trace.success = result.success
            step2_trace.result = result.result
            step2_trace.error = result.error
            
            if result.success:
                self.log_success(f"Step 2 SUCCESS: Query executed")
            else:
                self.log_error(f"Step 2 FAILED: {result.error}")
                
        except Exception as e:
            step2_trace.error = str(e)
            self.log_error(f"Step 2 EXCEPTION: {e}")
        
        self.traces.append(step2_trace)
    
    async def _test_streaming_chain_simulation(self, tool_manager):
        """
        Simulate how the streaming system processes chained tool calls.
        
        This tests the exact path that fails in the streaming chat interface.
        """
        self.log_info("🌊 Testing streaming chain simulation")
        
        # Simulate the streaming tool call format that would come from LLM
        streaming_tool_calls = [
            {
                "id": "call_list_tables_123",
                "type": "function",
                "function": {
                    "name": "stdio_list_tables",  # Note: sanitized name
                    "arguments": "{}"
                }
            },
            {
                "id": "call_read_query_456", 
                "type": "function",
                "function": {
                    "name": "stdio_read_query",  # Note: sanitized name
                    "arguments": '{"query": "SELECT * FROM products LIMIT 10"}'
                }
            }
        ]
        
        # Test name mapping (universal tool compatibility)
        name_mapping = {
            "stdio_list_tables": "stdio.list_tables",
            "stdio_read_query": "stdio.read_query"
        }
        
        # Process tools calls like the streaming system does
        for i, tool_call in enumerate(streaming_tool_calls):
            step = i + 3  # Continue from direct test steps
            
            # Extract info like streaming system does
            llm_tool_name = tool_call["function"]["name"]
            execution_tool_name = name_mapping.get(llm_tool_name, llm_tool_name)
            raw_arguments = tool_call["function"]["arguments"]
            
            self.log_debug(f"Processing streaming tool call:")
            self.log_debug(f"  LLM name: {llm_tool_name}")
            self.log_debug(f"  Execution name: {execution_tool_name}")
            self.log_debug(f"  Raw arguments: {raw_arguments}")
            
            # Parse arguments like streaming system does
            try:
                if isinstance(raw_arguments, str):
                    arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
                else:
                    arguments = raw_arguments or {}
            except json.JSONDecodeError as e:
                self.log_error(f"JSON decode error: {e}")
                arguments = {}
            
            # Create trace
            trace = ChainedToolCallTrace(
                step=step,
                tool_name=execution_tool_name,
                arguments=arguments,
                depends_on="streaming_simulation",
                streaming_artifacts={
                    "llm_name": llm_tool_name,
                    "execution_name": execution_tool_name,
                    "raw_arguments": raw_arguments,
                    "tool_call_id": tool_call["id"]
                }
            )
            
            # Execute like streaming system does
            start_time = time.time()
            try:
                result = await tool_manager.execute_tool(execution_tool_name, arguments)
                trace.execution_time = time.time() - start_time
                trace.success = result.success
                trace.result = result.result
                trace.error = result.error
                
                if result.success:
                    self.log_success(f"Streaming Step {step} SUCCESS")
                else:
                    self.log_error(f"Streaming Step {step} FAILED: {result.error}")
                    # This is likely where our error occurs!
                    
            except Exception as e:
                trace.error = str(e)
                self.log_error(f"Streaming Step {step} EXCEPTION: {e}")
            
            self.traces.append(trace)
    
    async def _test_malformed_streaming_arguments(self, tool_manager):
        """Test what happens with malformed streaming arguments."""
        self.log_info("🔧 Testing malformed streaming arguments")
        
        # These simulate the kinds of malformed JSON that streaming might produce
        malformed_cases = [
            {
                "name": "Empty String",
                "arguments": "",
                "expected_error": "Missing required parameter"
            },
            {
                "name": "Incomplete JSON",
                "arguments": '{"query": "SELECT * FROM',
                "expected_error": "JSON decode error"
            },
            {
                "name": "Concatenated JSON",
                "arguments": '{"query": "SELECT"}{"table": "products"}',
                "expected_error": "JSON decode error"
            },
            {
                "name": "Null Arguments",
                "arguments": None,
                "expected_error": "Missing required parameter"
            }
        ]
        
        for case in malformed_cases:
            step = len(self.traces) + 1
            
            self.log_info(f"Testing malformed case: {case['name']}")
            
            trace = ChainedToolCallTrace(
                step=step,
                tool_name="read_query",
                arguments={},  # Will be set based on parsing
                depends_on="malformed_test",
                streaming_artifacts=case
            )
            
            # Try to parse arguments like streaming system
            try:
                raw_args = case["arguments"]
                if isinstance(raw_args, str):
                    arguments = json.loads(raw_args) if raw_args and raw_args.strip() else {}
                else:
                    arguments = raw_args or {}
                    
                trace.arguments = arguments
                
            except json.JSONDecodeError as e:
                trace.error = f"JSON parsing failed: {e}"
                arguments = {}
                trace.arguments = arguments
            
            # Try to execute
            start_time = time.time()
            try:
                result = await tool_manager.execute_tool("read_query", arguments)
                trace.execution_time = time.time() - start_time
                trace.success = result.success
                trace.result = result.result
                if not result.success:
                    trace.error = result.error
                    
            except Exception as e:
                trace.error = str(e)
            
            self.traces.append(trace)
            
            # Log the result
            if trace.error:
                if case["expected_error"] in trace.error:
                    self.log_success(f"✅ Expected error for {case['name']}: {trace.error}")
                else:
                    self.log_warning(f"⚠️ Unexpected error for {case['name']}: {trace.error}")
            else:
                self.log_warning(f"⚠️ No error for {case['name']} (expected: {case['expected_error']})")
    
    def _analyze_chained_results(self):
        """Analyze the chained tool call results."""
        if not self.console:
            self._print_text_analysis()
            return
            
        self.log_info("📊 Analyzing chained tool call results")
        
        # Summary
        total_steps = len(self.traces)
        successful_steps = len([t for t in self.traces if t.success])
        failed_steps = total_steps - successful_steps
        
        # Create summary table
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Steps", str(total_steps))
        summary_table.add_row("Successful Steps", str(successful_steps))
        summary_table.add_row("Failed Steps", str(failed_steps))
        
        self.console.print(Panel(summary_table, title="Chained Tool Call Analysis", border_style="blue"))
        
        # Detailed trace table
        trace_table = Table(show_header=True, header_style="bold yellow")
        trace_table.add_column("Step", style="cyan")
        trace_table.add_column("Tool", style="cyan")
        trace_table.add_column("Arguments", style="dim")
        trace_table.add_column("Status", style="bold")
        trace_table.add_column("Error/Time", style="red")
        trace_table.add_column("Type", style="magenta")
        
        for trace in self.traces:
            status = "✅ SUCCESS" if trace.success else "❌ FAILED"
            error_or_time = f"{trace.execution_time:.3f}s" if trace.success else (trace.error or "Unknown error")
            test_type = "Streaming" if trace.streaming_artifacts else "Direct"
            
            args_str = str(trace.arguments)
            if len(args_str) > 30:
                args_str = args_str[:27] + "..."
            
            trace_table.add_row(
                str(trace.step),
                trace.tool_name,
                args_str,
                status,
                error_or_time,
                test_type
            )
        
        self.console.print(Panel(trace_table, title="Detailed Execution Traces", border_style="yellow"))
        
        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            rec_text = Text()
            for i, rec in enumerate(recommendations, 1):
                rec_text.append(f"{i}. {rec}\n", style="white")
            
            self.console.print(Panel(rec_text, title="Recommendations", border_style="green"))
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on trace analysis."""
        recommendations = []
        
        # Check for pattern: direct calls work but streaming fails
        direct_calls = [t for t in self.traces if not t.streaming_artifacts]
        streaming_calls = [t for t in self.traces if t.streaming_artifacts]
        
        direct_success = all(t.success for t in direct_calls)
        streaming_failures = [t for t in streaming_calls if not t.success]
        
        if direct_success and streaming_failures:
            recommendations.append("CRITICAL: Direct tool calls work but streaming fails - issue is in streaming processing")
            recommendations.append("Focus debugging on streaming_handler.py tool call accumulation")
            recommendations.append("Check tool call argument parsing in streaming system")
        
        # Check for specific error patterns
        query_param_errors = [t for t in self.traces if t.error and "query" in t.error and "required" in t.error]
        if query_param_errors:
            recommendations.append("Query parameter validation errors detected - check argument preprocessing")
            recommendations.append("Investigate tool call JSON parsing in streaming chunks")
        
        # Check for JSON parsing errors
        json_errors = [t for t in self.traces if t.error and "JSON" in t.error]
        if json_errors:
            recommendations.append("JSON parsing errors in streaming - implement robust argument accumulation")
            recommendations.append("Add argument validation before tool execution")
        
        return recommendations
    
    def _print_text_analysis(self):
        """Print analysis in plain text format."""
        print("\n" + "="*80)
        print("CHAINED TOOL CALL ANALYSIS")
        print("="*80)
        
        total_steps = len(self.traces)
        successful_steps = len([t for t in self.traces if t.success])
        failed_steps = total_steps - successful_steps
        
        print(f"Total Steps: {total_steps}")
        print(f"Successful Steps: {successful_steps}")
        print(f"Failed Steps: {failed_steps}")
        print()
        
        print("DETAILED TRACES:")
        print("-" * 40)
        for trace in self.traces:
            status = "SUCCESS" if trace.success else "FAILED"
            test_type = "Streaming" if trace.streaming_artifacts else "Direct"
            
            print(f"Step {trace.step}: {trace.tool_name} ({test_type})")
            print(f"  Arguments: {trace.arguments}")
            print(f"  Status: {status}")
            if trace.error:
                print(f"  Error: {trace.error}")
            elif trace.success:
                print(f"  Time: {trace.execution_time:.3f}s")
            print()
    
    def log_info(self, message: str):
        """Log info message."""
        if self.console:
            self.console.print(f"[blue]ℹ️  {message}[/blue]")
        else:
            print(f"INFO: {message}")
        self.logger.info(message)
    
    def log_success(self, message: str):
        """Log success message."""
        if self.console:
            self.console.print(f"[green]✅ {message}[/green]")
        else:
            print(f"SUCCESS: {message}")
        self.logger.info(message)
    
    def log_error(self, message: str):
        """Log error message."""
        if self.console:
            self.console.print(f"[red]❌ {message}[/red]")
        else:
            print(f"ERROR: {message}")
        self.logger.error(message)
    
    def log_warning(self, message: str):
        """Log warning message."""
        if self.console:
            self.console.print(f"[yellow]⚠️  {message}[/yellow]")
        else:
            print(f"WARNING: {message}")
        self.logger.warning(message)
    
    def log_debug(self, message: str):
        """Log debug message."""
        if self.console:
            self.console.print(f"[dim]🔍 {message}[/dim]")
        self.logger.debug(message)


async def main():
    """Main entry point for chained tool call diagnostic."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chained Tool Call Diagnostic")
    parser.add_argument("--config", default="server_config.json", help="Server config file")
    parser.add_argument("--server", action="append", default=[], help="Server names to test")
    
    args = parser.parse_args()
    
    if not args.server:
        args.server = ["sqlite"]
    
    diagnostic = ChainedToolCallDiagnostic()
    
    try:
        success = await diagnostic.diagnose_chained_scenario(args.config, args.server)
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nDiagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())