"""
Centralized Chat Display Manager for MCP-CLI.

This module consolidates ALL UI display logic for chat mode into a single
coherent system that prevents conflicts and ensures consistent behavior.

Replaces scattered UI logic from:
- ui_manager.py (partial)
- tool_processor.py (display parts)
- streaming_handler.py (display parts)  
- formatting.py (tool formatting)
- unified_display.py (abandoned approach)
"""

import time
import json
from typing import Optional, Dict, List, Any
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

from chuk_term.ui import output


class ChatDisplayManager:
    """Centralized display manager for all chat UI operations."""
    
    def __init__(self, console: Console):
        self.console = console
        self.live_display: Optional[Live] = None
        
        # Display state
        self.is_streaming = False
        self.streaming_content = ""
        self.streaming_start_time = 0.0
        
        self.is_tool_executing = False
        self.current_tool: Optional[Dict[str, Any]] = None
        self.tool_start_time = 0.0
        
        # Spinner animation
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0
    
    # ==================== STREAMING METHODS ====================
    
    def start_streaming(self):
        """Start streaming response display."""
        self.is_streaming = True
        self.streaming_content = ""
        self.streaming_start_time = time.time()
        self._ensure_live_display()
    
    def update_streaming(self, content: str):
        """Update streaming content."""
        if self.is_streaming:
            self.streaming_content += content
            self._refresh_display()
    
    def finish_streaming(self):
        """Finish streaming and show final response."""
        if not self.is_streaming:
            return
            
        self.is_streaming = False
        self._stop_live_display()
        
        # Show final response panel
        if self.streaming_content:
            elapsed = time.time() - self.streaming_start_time
            self._show_final_response(self.streaming_content, elapsed)
    
    # ==================== TOOL EXECUTION METHODS ====================
    
    def start_tool_execution(self, tool_name: str, arguments: Dict[str, Any]):
        """Start animated tool execution display."""
        self.is_tool_executing = True
        self.current_tool = {
            "name": tool_name,
            "arguments": arguments,
            "start_time": time.time()
        }
        
        # Start animated tool execution panel
        self._ensure_live_display()
    
    def finish_tool_execution(self, result: str, success: bool = True):
        """Finish tool execution and show final result."""
        if not self.is_tool_executing or not self.current_tool:
            return
            
        # Store result for final display
        elapsed = time.time() - self.current_tool["start_time"]
        self.current_tool.update({
            "result": result,
            "success": success,
            "elapsed": elapsed,
            "completed": True
        })
        
        self.is_tool_executing = False
        self._stop_live_display()
        
        # Show final tool result panel
        self._show_final_tool_result()
        self.current_tool = None
    
    # ==================== USER MESSAGE METHODS ====================
    
    def show_user_message(self, message: str):
        """Show user message."""
        output.user_message(message)
    
    def show_assistant_message(self, content: str, elapsed: float):
        """Show assistant message (non-streaming)."""
        output.assistant_message(content, elapsed=elapsed)
    
    # ==================== PRIVATE METHODS ====================
    
    def _ensure_live_display(self):
        """Ensure live display is active."""
        if not self.live_display:
            self.live_display = Live(
                self._create_live_content(),
                console=self.console,
                refresh_per_second=8,
                transient=True
            )
            self.live_display.__enter__()
    
    def _stop_live_display(self):
        """Stop live display."""
        if self.live_display:
            self.live_display.__exit__(None, None, None)
            self.live_display = None
    
    def _refresh_display(self):
        """Refresh live display content."""
        if self.live_display:
            self.live_display.update(self._create_live_content())
    
    def _create_live_content(self) -> Panel:
        """Create live display content."""
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        spinner = self.spinner_frames[self.spinner_index]
        
        parts = []
        title = ""
        border_style = "yellow"
        
        # Assistant streaming section
        if self.is_streaming:
            elapsed = time.time() - self.streaming_start_time
            title = "🤖 Assistant"
            
            # Status line
            status = f"{spinner} Generating response..."
            parts.append(Text(status, style="yellow"))
            
            # Stats line
            stats = f"⎿  {len(self.streaming_content):,} chars • {elapsed:.1f}s"
            parts.append(Text(stats, style="dim"))
            parts.append(Text(""))
            
            # Content preview
            if self.streaming_content:
                preview_lines = self.streaming_content.split('\n')[:3]
                for line in preview_lines:
                    if line.strip():
                        preview = line[:50] + "..." if len(line) > 50 else line
                        parts.append(Text(f"   {preview}", style="dim cyan"))
                
                if len(self.streaming_content) > 200:
                    parts.append(Text("   ... generating content", style="dim italic"))
            else:
                parts.append(Text("   ▌", style="yellow blink"))
        
        # Tool execution section
        elif self.is_tool_executing and self.current_tool:
            elapsed = time.time() - self.current_tool["start_time"]
            title = "🔧 Tool Execution"
            
            # Animated status
            status = f"{spinner} Executing {self.current_tool['name']}..."
            parts.append(Text(status, style="yellow"))
            
            # Stats
            stats = f"⎿  {elapsed:.1f}s"
            parts.append(Text(stats, style="dim"))
            parts.append(Text(""))
            
            # Arguments
            args = self.current_tool.get("arguments", {})
            if args and any(str(v).strip() for v in args.values() if v is not None):
                parts.append(Text("Arguments:", style="dim"))
                filtered_args = {k: v for k, v in args.items() 
                               if v is not None and str(v).strip()}
                for key, value in filtered_args.items():
                    parts.append(Text(f"  {key}: {value}", style="cyan"))
                parts.append(Text(""))
            
            # Animated dots for execution
            dots = "." * (int(elapsed * 2) % 4)
            parts.append(Text(f"Processing{dots}", style="yellow italic"))
        
        return Panel(
            Group(*parts),
            title=title,
            border_style=border_style,
            expand=False
        )
    
    def _show_final_response(self, content: str, elapsed: float):
        """Show final response panel."""
        try:
            # Try to render as markdown for better formatting
            rendered_content = Markdown(content)
        except Exception:
            # Fallback to plain text
            rendered_content = Text(content, overflow="fold")
        
        final_panel = Panel(
            rendered_content,
            title="🤖 Assistant",
            subtitle=f"Response time: {elapsed:.2f}s",
            subtitle_align="right",
            border_style="green",
            expand=True
        )
        
        self.console.print(final_panel)
    
    def _show_final_tool_result(self):
        """Show final tool execution result panel."""
        if not self.current_tool:
            return
            
        tool_info = self.current_tool
        parts = []
        
        # Status header
        status_style = "green" if tool_info["success"] else "red"
        status_text = "✓ Completed" if tool_info["success"] else "✗ Failed"
        parts.append(Text(f"{status_text}: {tool_info['name']}", style=f"bold {status_style}"))
        
        # Execution time
        parts.append(Text(f"Execution time: {tool_info['elapsed']:.2f}s", style="dim"))
        
        # Arguments (compact)
        args = tool_info.get("arguments", {})
        if args and any(str(v).strip() for v in args.values() if v is not None):
            parts.append(Text(""))
            parts.append(Text("Arguments:", style="dim"))
            filtered_args = {k: v for k, v in args.items() 
                           if v is not None and str(v).strip()}
            for key, value in filtered_args.items():
                parts.append(Text(f"  {key}: {value}", style="cyan"))
        
        # Result
        result = tool_info.get("result", "")
        if result:
            parts.append(Text(""))
            parts.append(Text("Result:", style="dim"))
            
            # Try to format result nicely
            try:
                # Try to parse as JSON for better formatting
                parsed = json.loads(result)
                formatted_result = json.dumps(parsed, indent=2)
                parts.append(Text(formatted_result, style=status_style))
            except (json.JSONDecodeError, TypeError):
                # Use as plain text
                parts.append(Text(str(result), style=status_style))
        
        result_panel = Panel(
            Group(*parts),
            title="🔧 Tool Execution",
            border_style=status_style,
            expand=False
        )
        
        self.console.print(result_panel)
    
    def _show_tool_invocation(self, tool_name: str, arguments: Dict[str, Any]):
        """Show tool invocation panel."""
        parts = [Text(f"Tool: {tool_name}", style="bold yellow")]
        
        # Show arguments if available
        if arguments and any(str(v).strip() for v in arguments.values() if v is not None):
            parts.append(Text(""))
            parts.append(Text("Arguments:", style="dim"))
            
            # Filter and format arguments
            filtered_args = {k: v for k, v in arguments.items() 
                           if v is not None and str(v).strip()}
            
            if filtered_args:
                for key, value in filtered_args.items():
                    parts.append(Text(f"  {key}: {value}", style="cyan"))
        
        invocation_panel = Panel(
            Group(*parts),
            title="🔧 Tool Invocation",
            border_style="yellow",
            expand=False
        )
        
        self.console.print(invocation_panel)
    
    def _show_tool_result(self, tool_info: Dict[str, Any], result: str, 
                         elapsed: float, success: bool):
        """Show tool execution result."""
        parts = []
        
        # Tool name and status
        status_style = "green" if success else "red"
        status_text = "✓ Completed" if success else "✗ Failed"
        parts.append(Text(f"{status_text}: {tool_info['name']}", style=f"bold {status_style}"))
        
        # Execution time
        parts.append(Text(f"Execution time: {elapsed:.2f}s", style="dim"))
        
        # Result
        if result:
            parts.append(Text(""))
            parts.append(Text("Result:", style="dim"))
            
            # Try to format result nicely
            try:
                # Try to parse as JSON for better formatting
                parsed = json.loads(result)
                formatted_result = json.dumps(parsed, indent=2)
                parts.append(Text(formatted_result, style="green" if success else "red"))
            except (json.JSONDecodeError, TypeError):
                # Use as plain text
                parts.append(Text(str(result), style="green" if success else "red"))
        
        result_panel = Panel(
            Group(*parts),
            title="🔧 Tool Result",
            border_style="green" if success else "red",
            expand=False
        )
        
        self.console.print(result_panel)