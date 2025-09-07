"""
Unified UI Display Manager for MCP-CLI.

Handles all display coordination including streaming responses, tool execution,
and other UI elements through a single Rich Live display to avoid conflicts.
"""

import time
from typing import Optional, List
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.table import Table
from rich.columns import Columns


class UnifiedDisplayManager:
    """Unified display manager that coordinates all UI elements."""
    
    def __init__(self, console: Console):
        self.console = console
        self.live: Optional[Live] = None
        self.start_time = time.time()
        
        # Display state
        self.assistant_content = ""
        self.assistant_streaming = False
        self.tool_calls: List[dict] = []
        self.current_tool_call: Optional[dict] = None
        self.tool_executing = False
        
        # Spinner state
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0
        
    def start_display(self):
        """Start the unified live display."""
        if not self.live:
            self.live = Live(
                self._create_display(),
                console=self.console,
                refresh_per_second=8,
                transient=True
            )
            self.live.__enter__()
        
    def stop_display(self):
        """Stop the live display and show final panels."""
        if self.live:
            self.live.__exit__(None, None, None)
            self.live = None
            
            # Show final panels
            self._show_final_panels()
    
    def start_assistant_streaming(self, title: str = "🤖 Assistant"):
        """Start assistant response streaming."""
        self.assistant_content = ""
        self.assistant_streaming = True
        self.start_time = time.time()
        self.start_display()
    
    def update_assistant_content(self, chunk: str):
        """Update assistant streaming content."""
        if self.assistant_streaming:
            self.assistant_content += chunk
            self._update_display()
    
    def finish_assistant_streaming(self):
        """Finish assistant streaming."""
        self.assistant_streaming = False
        self._update_display()
    
    def start_tool_execution(self, tool_name: str, args: dict = None):
        """Start tool execution display."""
        self.current_tool_call = {
            "name": tool_name,
            "args": args or {},
            "status": "executing",
            "result": None,
            "start_time": time.time()
        }
        self.tool_executing = True
        
        if not self.live:
            self.start_display()
        else:
            self._update_display()
    
    def finish_tool_execution(self, result: str = None, success: bool = True):
        """Finish tool execution."""
        if self.current_tool_call:
            self.current_tool_call["status"] = "completed" if success else "failed"
            self.current_tool_call["result"] = result
            self.tool_calls.append(self.current_tool_call.copy())
        
        self.current_tool_call = None
        self.tool_executing = False
        self._update_display()
    
    def _update_display(self):
        """Update the live display."""
        if self.live:
            self.live.update(self._create_display())
    
    def _create_display(self) -> Group:
        """Create the unified display content without wrapping in a panel."""
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        spinner = self.spinner_frames[self.spinner_index]
        
        display_parts = []
        
        # Assistant streaming section
        if self.assistant_streaming:
            status = f"{spinner} Generating response..."
            elapsed = time.time() - self.start_time
            stats = f"⎿  {len(self.assistant_content):,} chars • {elapsed:.1f}s"
            
            assistant_parts = []
            assistant_parts.append(Text(status, style="yellow"))
            assistant_parts.append(Text(stats, style="dim"))
            assistant_parts.append(Text(""))
            
            # Show preview of content
            if self.assistant_content:
                preview_lines = self.assistant_content.split('\n')[:3]
                for line in preview_lines:
                    if line.strip():
                        preview = line[:50] + "..." if len(line) > 50 else line
                        assistant_parts.append(Text(f"   {preview}", style="dim cyan"))
                
                if len(self.assistant_content) > 200:
                    assistant_parts.append(Text("   ... generating content", style="dim italic"))
            else:
                assistant_parts.append(Text("   ▌", style="yellow blink"))
            
            # Wrap assistant content in its own panel
            assistant_panel = Panel(
                Group(*assistant_parts),
                title="🤖 Assistant",
                border_style="yellow",
                expand=False
            )
            display_parts.append(assistant_panel)
        
        # Tool execution section
        if self.tool_executing and self.current_tool_call:
            tool_name = self.current_tool_call["name"]
            tool_status = f"{spinner} Executing {tool_name}..."
            tool_elapsed = time.time() - self.current_tool_call["start_time"]
            tool_stats = f"⎿  {tool_elapsed:.1f}s"
            
            tool_parts = []
            tool_parts.append(Text(tool_status, style="yellow"))
            tool_parts.append(Text(tool_stats, style="dim"))
            
            # Show tool arguments if available
            args = self.current_tool_call.get("args", {})
            if args and any(str(v).strip() for v in args.values() if v is not None):
                tool_parts.append(Text(""))
                tool_parts.append(Text("Arguments:", style="dim"))
                for key, value in args.items():
                    if value is not None and str(value).strip():
                        tool_parts.append(Text(f"  • {key}: {value}", style="dim cyan"))
            
            # Wrap tool content in its own panel
            tool_panel = Panel(
                Group(*tool_parts),
                title="🔧 Tool Execution",
                border_style="yellow",
                expand=False
            )
            display_parts.append(tool_panel)
        
        # Return a group of panels instead of a single panel
        return Group(*display_parts)
    
    def _show_final_panels(self):
        """Show final panels for completed operations."""
        panels = []
        
        # Final assistant response
        if self.assistant_content:
            elapsed = time.time() - self.start_time
            try:
                # Try to render as markdown
                content = Markdown(self.assistant_content)
            except Exception:
                # Fallback to text
                content = Text(self.assistant_content, overflow="fold")
            
            panels.append(Panel(
                content,
                title="🤖 Assistant",
                subtitle=f"Response time: {elapsed:.2f}s",
                subtitle_align="right",
                border_style="green",
                expand=True
            ))
        
        # Tool execution results
        for tool_call in self.tool_calls:
            tool_content = []
            tool_content.append(Text(f"Tool: {tool_call['name']}", style="bold"))
            
            # Show arguments
            args = tool_call.get("args", {})
            if args and any(str(v).strip() for v in args.values() if v is not None):
                tool_content.append(Text(""))
                tool_content.append(Text("Arguments:", style="dim"))
                for key, value in args.items():
                    if value is not None and str(value).strip():
                        tool_content.append(Text(f"  {key}: {value}", style="cyan"))
            
            # Show result
            result = tool_call.get("result")
            if result:
                tool_content.append(Text(""))
                tool_content.append(Text("Result:", style="dim"))
                tool_content.append(Text(str(result), style="green"))
            
            panels.append(Panel(
                Group(*tool_content),
                title="🔧 Tool Execution",
                border_style="green" if tool_call["status"] == "completed" else "red",
                expand=True
            ))
        
        # Display all panels
        for panel in panels:
            self.console.print(panel)