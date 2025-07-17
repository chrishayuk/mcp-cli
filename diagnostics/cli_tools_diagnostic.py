#!/usr/bin/env python3
"""
MCP CLI Tool Parameter Diagnosis Script

This script helps diagnose the "Missing required parameter 'query'" error
that occurs in MCP CLI tool execution. It traces the complete flow from
LLM tool calls through parameter validation to actual execution.

Usage:
    python diagnose_mcp_tools.py --config server_config.json --server sqlite
    python diagnose_mcp_tools.py --debug --trace-validation
    python diagnose_mcp_tools.py --simulate-error --tool read_query
"""

import argparse
import asyncio
import json
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Rich for output formatting
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
class ToolCallTrace:
    """Trace information for a tool call execution."""
    timestamp: str
    tool_name: str
    original_args: Dict[str, Any]
    validation_result: Dict[str, Any]
    execution_result: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None
    call_stack: Optional[List[str]] = None


@dataclass
class DiagnosisResult:
    """Results from the diagnosis process."""
    total_tools_tested: int
    successful_calls: int
    failed_calls: int
    validation_errors: int
    execution_errors: int
    traces: List[ToolCallTrace]
    recommendations: List[str]


class MCPToolDiagnosticTool:
    """Comprehensive diagnostic tool for MCP CLI tool parameter issues."""
    
    def __init__(self, console: Optional['Console'] = None):
        self.console = console or (Console() if HAS_RICH else None)
        self.traces: List[ToolCallTrace] = []
        self.debug_mode = False
        self.trace_validation = False
        
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('mcp_diagnosis.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def enable_debug(self, trace_validation: bool = False):
        """Enable debug mode with optional validation tracing."""
        self.debug_mode = True
        self.trace_validation = trace_validation
        self.logger.setLevel(logging.DEBUG)
        
    def log_debug(self, message: str, **kwargs):
        """Debug logging with rich formatting if available."""
        if self.debug_mode:
            if self.console:
                self.console.print(f"[dim]DEBUG: {message}[/dim]", **kwargs)
            self.logger.debug(message)
    
    def log_error(self, message: str, **kwargs):
        """Error logging with rich formatting if available."""
        if self.console:
            self.console.print(f"[red]ERROR: {message}[/red]", **kwargs)
        self.logger.error(message)
    
    def log_warning(self, message: str, **kwargs):
        """Warning logging with rich formatting if available."""
        if self.console:
            self.console.print(f"[yellow]WARNING: {message}[/yellow]", **kwargs)
        self.logger.warning(message)
    
    def log_success(self, message: str, **kwargs):
        """Success logging with rich formatting if available."""
        if self.console:
            self.console.print(f"[green]SUCCESS: {message}[/green]", **kwargs)
        self.logger.info(message)

    async def diagnose_tool_manager(self, config_file: str, servers: List[str]) -> DiagnosisResult:
        """
        Comprehensive diagnosis of ToolManager and tool execution.
        
        Args:
            config_file: Path to MCP server configuration
            servers: List of server names to test
            
        Returns:
            DiagnosisResult with detailed findings
        """
        self.log_debug("Starting comprehensive ToolManager diagnosis")
        
        # Import MCP CLI components
        try:
            from mcp_cli.tools.manager import ToolManager
            from mcp_cli.tools.models import ToolCallResult
        except ImportError as e:
            self.log_error(f"Failed to import MCP CLI components: {e}")
            return DiagnosisResult(0, 0, 0, 0, 0, [], ["Install MCP CLI components"])
        
        # Initialize ToolManager
        tool_manager = None
        try:
            self.log_debug("Initializing ToolManager...")
            tool_manager = ToolManager(config_file, servers)
            
            success = await tool_manager.initialize()
            if not success:
                self.log_error("ToolManager initialization failed")
                return DiagnosisResult(0, 0, 0, 0, 0, [], ["Check server configuration"])
            
            self.log_success("ToolManager initialized successfully")
            
        except Exception as e:
            self.log_error(f"ToolManager initialization error: {e}")
            traceback.print_exc()
            return DiagnosisResult(0, 0, 0, 0, 0, [], ["Fix ToolManager initialization"])
        
        try:
            # Get available tools
            tools = await tool_manager.get_all_tools()
            self.log_debug(f"Found {len(tools)} available tools")
            
            # Test critical tools
            critical_tools = [
                ("read_query", {"query": "SELECT 1"}),
                ("write_query", {"query": "SELECT COUNT(*) FROM sqlite_master"}),
                ("list_tables", {}),
                ("describe_table", {"table_name": "products"}),
            ]
            
            results = []
            for tool_name, test_args in critical_tools:
                result = await self._test_tool_execution(tool_manager, tool_name, test_args)
                results.append(result)
                
            # Analyze results
            return self._analyze_results(results)
            
        finally:
            if tool_manager:
                await tool_manager.close()

    async def _test_tool_execution(
        self, 
        tool_manager: Any, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> ToolCallTrace:
        """
        Test a single tool execution with comprehensive tracing.
        
        Args:
            tool_manager: ToolManager instance
            tool_name: Name of tool to test
            arguments: Arguments to pass to tool
            
        Returns:
            ToolCallTrace with detailed execution information
        """
        timestamp = datetime.now().isoformat()
        trace = ToolCallTrace(
            timestamp=timestamp,
            tool_name=tool_name,
            original_args=arguments.copy(),
            validation_result={},
            call_stack=[]
        )
        
        self.log_debug(f"Testing tool: {tool_name} with args: {arguments}")
        
        try:
            # Capture call stack
            import inspect
            trace.call_stack = [
                f"{frame.filename}:{frame.lineno} in {frame.function}"
                for frame in inspect.stack()[:5]
            ]
            
            # Test validation manually if trace_validation is enabled
            if self.trace_validation:
                validation_result = self._manual_validate_tool_arguments(tool_name, arguments)
                trace.validation_result = validation_result
                
                if not validation_result.get("valid", True):
                    trace.error_details = validation_result.get("error", "Validation failed")
                    self.log_error(f"Validation failed for {tool_name}: {trace.error_details}")
                    return trace
            
            # Execute tool
            self.log_debug(f"Executing tool {tool_name}...")
            result = await tool_manager.execute_tool(tool_name, arguments)
            
            # Record results
            if hasattr(result, 'success'):
                trace.execution_result = {
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                    "execution_time": getattr(result, 'execution_time', None)
                }
                
                if result.success:
                    self.log_success(f"Tool {tool_name} executed successfully")
                else:
                    self.log_error(f"Tool {tool_name} failed: {result.error}")
                    trace.error_details = result.error
            else:
                # Handle unexpected result format
                trace.execution_result = {"raw_result": str(result)}
                self.log_warning(f"Unexpected result format from {tool_name}: {type(result)}")
                
        except Exception as e:
            trace.error_details = str(e)
            trace.execution_result = {"exception": str(e), "traceback": traceback.format_exc()}
            self.log_error(f"Exception executing {tool_name}: {e}")
            
        self.traces.append(trace)
        return trace
    
    def _manual_validate_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manually validate tool arguments using the same logic as ToolManager.
        
        This helps diagnose validation issues by replicating the validation
        logic outside of the normal execution flow.
        """
        # Known tool validation rules (from ToolManager._validate_tool_arguments)
        validation_rules = {
            "read_query": {"required": ["query"], "types": {"query": str}},
            "write_query": {"required": ["query"], "types": {"query": str}},
            "describe_table": {"required": ["table_name"], "types": {"table_name": str}},
            "create_table": {"required": ["query"], "types": {"query": str}},
        }
        
        self.log_debug(f"Manual validation for {tool_name} with rules: {validation_rules.get(tool_name, 'none')}")
        
        if tool_name not in validation_rules:
            return {"valid": True, "message": "No validation rules for this tool"}
        
        rules = validation_rules[tool_name]
        
        # Check required parameters
        for required_param in rules.get("required", []):
            if required_param not in arguments:
                return {
                    "valid": False,
                    "error": f"Missing required parameter '{required_param}' for tool '{tool_name}'"
                }
            
            if not arguments[required_param]:
                return {
                    "valid": False,
                    "error": f"Parameter '{required_param}' cannot be empty for tool '{tool_name}'"
                }
        
        # Check parameter types
        for param, expected_type in rules.get("types", {}).items():
            if param in arguments and not isinstance(arguments[param], expected_type):
                return {
                    "valid": False,
                    "error": f"Parameter '{param}' must be of type {expected_type.__name__} for tool '{tool_name}'"
                }
        
        return {"valid": True, "message": "Validation passed"}
    
    def _analyze_results(self, traces: List[ToolCallTrace]) -> DiagnosisResult:
        """Analyze execution traces and generate recommendations."""
        total_tools = len(traces)
        successful = len([t for t in traces if t.execution_result and t.execution_result.get("success")])
        failed = total_tools - successful
        validation_errors = len([t for t in traces if not t.validation_result.get("valid", True)])
        execution_errors = len([t for t in traces if t.error_details and "validation" not in t.error_details.lower()])
        
        recommendations = []
        
        # Generate recommendations based on patterns
        if validation_errors > 0:
            recommendations.append("Review tool argument validation logic in ToolManager._validate_tool_arguments")
            recommendations.append("Check if tool names are being properly mapped between LLM and MCP servers")
        
        if execution_errors > 0:
            recommendations.append("Check MCP server connectivity and configuration")
            recommendations.append("Verify tool schemas match server implementations")
        
        if failed > successful:
            recommendations.append("Consider increasing tool timeout values")
            recommendations.append("Review server logs for additional error details")
        
        # Specific patterns - fix None handling
        query_failures = [t for t in traces if t.error_details and ("query" in t.error_details and "required" in t.error_details)]
        if query_failures:
            recommendations.append("CRITICAL: Query parameter validation failing - check argument parsing in streaming")
            recommendations.append("Review tool call extraction from LLM responses")
        
        # Check for validation vs execution split
        validation_only_failures = [t for t in traces if not t.validation_result.get("valid", True) and not t.execution_result]
        if validation_only_failures:
            recommendations.append("Tools failing validation before execution - check argument preprocessing")
        
        # Check for streaming-related issues
        streaming_indicators = [t for t in traces if t.error_details and ("streaming" in t.error_details.lower() or "chunk" in t.error_details.lower())]
        if streaming_indicators:
            recommendations.append("Streaming-related failures detected - check tool call extraction from streaming")
        
        return DiagnosisResult(
            total_tools_tested=total_tools,
            successful_calls=successful,
            failed_calls=failed,
            validation_errors=validation_errors,
            execution_errors=execution_errors,
            traces=traces,
            recommendations=recommendations
        )
    
    def simulate_common_errors(self):
        """Simulate common error scenarios for testing diagnosis logic."""
        self.log_debug("Simulating common error scenarios...")
        
        # Simulate empty arguments error
        empty_args_trace = ToolCallTrace(
            timestamp=datetime.now().isoformat(),
            tool_name="read_query",
            original_args={},
            validation_result={"valid": False, "error": "Missing required parameter 'query' for tool 'read_query'"},
            error_details="Missing required parameter 'query' for tool 'read_query'"
        )
        
        # Simulate null arguments error
        null_args_trace = ToolCallTrace(
            timestamp=datetime.now().isoformat(),
            tool_name="read_query",
            original_args={"query": None},
            validation_result={"valid": False, "error": "Parameter 'query' cannot be empty for tool 'read_query'"},
            error_details="Parameter 'query' cannot be empty for tool 'read_query'"
        )
        
        # Simulate type mismatch error
        type_error_trace = ToolCallTrace(
            timestamp=datetime.now().isoformat(),
            tool_name="read_query",
            original_args={"query": 123},
            validation_result={"valid": False, "error": "Parameter 'query' must be of type str for tool 'read_query'"},
            error_details="Parameter 'query' must be of type str for tool 'read_query'"
        )
        
        self.traces.extend([empty_args_trace, null_args_trace, type_error_trace])
        
        return self._analyze_results(self.traces)
    
    def generate_report(self, result: DiagnosisResult) -> str:
        """Generate a comprehensive diagnosis report."""
        if self.console:
            return self._generate_rich_report(result)
        else:
            return self._generate_text_report(result)
    
    def _generate_rich_report(self, result: DiagnosisResult) -> str:
        """Generate a rich-formatted report."""
        # Summary panel
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Tools Tested", str(result.total_tools_tested))
        summary_table.add_row("Successful Calls", str(result.successful_calls))
        summary_table.add_row("Failed Calls", str(result.failed_calls))
        summary_table.add_row("Validation Errors", str(result.validation_errors))
        summary_table.add_row("Execution Errors", str(result.execution_errors))
        
        self.console.print(Panel(summary_table, title="Diagnosis Summary", border_style="blue"))
        
        # Detailed traces
        if result.traces:
            trace_table = Table(show_header=True, header_style="bold yellow")
            trace_table.add_column("Tool", style="cyan")
            trace_table.add_column("Arguments", style="dim")
            trace_table.add_column("Status", style="bold")
            trace_table.add_column("Error/Result", style="red")
            
            for trace in result.traces:
                status = "✅ SUCCESS" if not trace.error_details else "❌ FAILED"
                error_or_result = trace.error_details or str(trace.execution_result)
                
                trace_table.add_row(
                    trace.tool_name,
                    str(trace.original_args),
                    status,
                    error_or_result[:50] + "..." if len(error_or_result) > 50 else error_or_result
                )
            
            self.console.print(Panel(trace_table, title="Execution Traces", border_style="yellow"))
        
        # Recommendations
        if result.recommendations:
            rec_text = Text()
            for i, rec in enumerate(result.recommendations, 1):
                rec_text.append(f"{i}. {rec}\n", style="white")
            
            self.console.print(Panel(rec_text, title="Recommendations", border_style="green"))
        
        return "Rich report displayed above"
    
    def _generate_text_report(self, result: DiagnosisResult) -> str:
        """Generate a plain text report."""
        lines = [
            "=" * 80,
            "MCP CLI Tool Diagnosis Report",
            "=" * 80,
            "",
            f"Total Tools Tested: {result.total_tools_tested}",
            f"Successful Calls: {result.successful_calls}",
            f"Failed Calls: {result.failed_calls}",
            f"Validation Errors: {result.validation_errors}",
            f"Execution Errors: {result.execution_errors}",
            "",
            "Detailed Traces:",
            "-" * 40,
        ]
        
        for trace in result.traces:
            lines.extend([
                f"Tool: {trace.tool_name}",
                f"Arguments: {trace.original_args}",
                f"Status: {'SUCCESS' if not trace.error_details else 'FAILED'}",
                f"Error/Result: {trace.error_details or trace.execution_result}",
                ""
            ])
        
        if result.recommendations:
            lines.extend([
                "Recommendations:",
                "-" * 40,
            ])
            for i, rec in enumerate(result.recommendations, 1):
                lines.append(f"{i}. {rec}")
        
        lines.append("=" * 80)
        return "\n".join(lines)


async def main():
    """Main diagnosis entry point."""
    parser = argparse.ArgumentParser(description="MCP CLI Tool Parameter Diagnosis")
    parser.add_argument("--config", default="server_config.json", help="Server config file")
    parser.add_argument("--server", action="append", default=[], help="Server names to test")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--trace-validation", action="store_true", help="Trace validation logic")
    parser.add_argument("--simulate-error", action="store_true", help="Simulate common errors")
    parser.add_argument("--tool", help="Test specific tool")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    # Default servers if none specified
    if not args.server:
        args.server = ["sqlite"]
    
    # Initialize diagnostic tool
    console = Console() if HAS_RICH else None
    diagnostic = MCPToolDiagnosticTool(console)
    
    if args.debug:
        diagnostic.enable_debug(args.trace_validation)
    
    try:
        if args.simulate_error:
            result = diagnostic.simulate_common_errors()
        else:
            result = await diagnostic.diagnose_tool_manager(args.config, args.server)
        
        # Generate and display report
        report = diagnostic.generate_report(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(diagnostic._generate_text_report(result))
            print(f"Report saved to {args.output}")
        
        # Exit with appropriate code
        if result.failed_calls > 0 or result.validation_errors > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nDiagnosis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Diagnosis failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())