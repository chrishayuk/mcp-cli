#!/usr/bin/env python
"""
Complete integration demo using all chuk libraries together.
This serves as a blueprint for MCP-CLI refactoring.

Libraries used:
- chuk-llm: Conversation management, streaming, tool handling
- chuk-term: Terminal UI, themes, output management
- chuk-tool-processor: MCP server integration (simulated here)

Run with: uv run examples/chuk_complete_integration.py
"""

import asyncio
import json
import signal
from contextlib import suppress
from typing import Dict, Any, List, Optional
from datetime import datetime

# Chuk libraries
from chuk_llm import conversation, ask, tools_from_functions
from chuk_term.ui import output, theme, prompts
from chuk_term.ui.terminal import clear_screen, reset_terminal


class ChukIntegratedChat:
    """
    Complete chat implementation using chuk libraries.
    This is how MCP-CLI could be structured after refactoring.
    """
    
    def __init__(self):
        # Terminal and theme setup
        self.setup_terminal()
        
        # State
        self.current_conversation = None
        self.running = True
        self.current_task = None
        
        # Tool functions (simulating MCP tools)
        self.setup_tools()
        
    def setup_terminal(self):
        """Setup terminal with chuk-term."""
        # Clear and setup
        clear_screen()
        
        # Set theme
        theme.set_theme("monokai")  # or "dracula", "solarized", etc.
        
        # Show header
        output.rule("MCP-CLI Chat (Chuk Integration)")
        output.info(f"Theme: {theme.get_current_theme()}")
        output.rule()
    
    def setup_tools(self):
        """Setup available tools."""
        # Define tool functions (simulating MCP tools)
        def list_files(directory: str = ".") -> dict:
            """List files in a directory."""
            return {
                "directory": directory,
                "files": ["README.md", "main.py", "config.json"],
                "count": 3
            }
        
        def read_file(filename: str) -> dict:
            """Read a file's contents."""
            return {
                "filename": filename,
                "content": f"Mock content of {filename}",
                "lines": 10
            }
        
        def run_command(command: str) -> dict:
            """Run a shell command."""
            return {
                "command": command,
                "output": f"Mock output from: {command}",
                "exit_code": 0
            }
        
        # Create toolkit
        self.toolkit = tools_from_functions(list_files, read_file, run_command)
        self.tools = self.toolkit.to_openai_format()
    
    async def start_conversation(self):
        """Start a new conversation with chuk-llm."""
        output.print_section("Starting Conversation")
        
        # Show available tools
        output.print_info("Available tools:")
        for tool in self.tools:
            func = tool["function"]
            output.print_item(f"{func['name']}: {func['description']}")
        
        output.print_separator()
        
        # Start conversation context
        self.current_conversation = conversation("ollama", "gpt-oss")
        return await self.current_conversation.__aenter__()
    
    async def handle_user_input(self, conv, user_input: str) -> bool:
        """
        Handle user input and generate response.
        Returns False if user wants to exit.
        """
        # Check for exit commands
        if user_input.lower() in ["exit", "quit", "/exit", "/quit"]:
            return False
        
        # Check for special commands
        if user_input.startswith("/"):
            return await self.handle_command(user_input)
        
        # Regular conversation with potential tool use
        output.print_separator(style="dim")
        
        # Determine if we should stream or not
        if "list" in user_input.lower() or "show" in user_input.lower():
            # Non-streaming for tool-heavy operations
            await self.handle_tool_response(conv, user_input)
        else:
            # Streaming for regular conversation
            await self.handle_streaming_response(conv, user_input)
        
        return True
    
    async def handle_command(self, command: str) -> bool:
        """Handle special commands."""
        cmd = command.lower().strip()
        
        if cmd == "/help":
            output.print_panel(
                title="Available Commands",
                content="""
/help     - Show this help
/clear    - Clear the screen
/theme    - Change theme
/tools    - List available tools
/stats    - Show conversation stats
/exit     - Exit the chat
                """.strip(),
                style="cyan"
            )
        elif cmd == "/clear":
            clear_screen()
            self.setup_terminal()
        elif cmd == "/theme":
            await self.change_theme()
        elif cmd == "/tools":
            self.show_tools()
        elif cmd == "/stats":
            await self.show_stats()
        else:
            output.print_warning(f"Unknown command: {command}")
        
        return True
    
    async def change_theme(self):
        """Change the terminal theme."""
        themes = ["default", "monokai", "dracula", "solarized", "minimal"]
        
        # Use chuk-term's select prompt
        selected = prompts.select_from_list(
            "Select a theme:",
            themes,
            current_value=theme.get_current_theme()
        )
        
        if selected:
            theme.set_theme(selected)
            output.print_success(f"Theme changed to: {selected}")
    
    def show_tools(self):
        """Display available tools."""
        output.print_table(
            title="Available Tools",
            headers=["Tool", "Description", "Parameters"],
            rows=[
                [
                    func["function"]["name"],
                    func["function"]["description"][:40] + "...",
                    str(len(func["function"]["parameters"].get("properties", {})))
                ]
                for func in self.tools
            ]
        )
    
    async def show_stats(self):
        """Show conversation statistics."""
        if self.current_conversation:
            conv = self.current_conversation
            stats = {
                "Messages": len(conv.messages) if hasattr(conv, 'messages') else 0,
                "Session ID": conv.session_id if hasattr(conv, 'session_id') else "N/A",
                "Has Session": conv.has_session if hasattr(conv, 'has_session') else False,
                "Started": datetime.now().strftime("%I:%M %p")
            }
            
            output.print_dict(stats, title="Conversation Stats", style="cyan")
    
    async def handle_streaming_response(self, conv, user_input: str):
        """Handle streaming response with chuk-term UI."""
        # Start streaming with loading indicator
        with output.loading("Thinking..."):
            await asyncio.sleep(0.5)  # Simulate thinking
        
        # Stream response with proper formatting
        output.print_label("Assistant", style="green")
        
        response_text = ""
        char_count = 0
        
        # Create a live output area
        output.print("", end="")  # Start on same line
        
        try:
            async for chunk in conv.stream(user_input):
                if isinstance(chunk, dict) and "response" in chunk:
                    content = chunk["response"]
                elif isinstance(chunk, str):
                    content = chunk
                else:
                    continue
                
                if content:
                    # Use chuk-term's output for proper terminal handling
                    print(content, end="", flush=True)
                    response_text += content
                    char_count += len(content)
        
        except asyncio.CancelledError:
            output.print_warning("\nStream cancelled")
            return
        
        print()  # New line after streaming
        
        # Show metrics
        if char_count > 0:
            output.print_dim(f"[{char_count} characters streamed]")
    
    async def handle_tool_response(self, conv, user_input: str):
        """Handle response that might use tools."""
        # Show progress while thinking
        with output.loading("Processing request..."):
            result = await conv.ask(user_input, tools=self.tools)
        
        # Check if tools were called
        if isinstance(result, dict) and "tool_calls" in result:
            output.print_label("Assistant", style="green")
            output.print("I'll help you with that. Let me use some tools...")
            
            # Process tool calls
            tool_calls = result["tool_calls"]
            
            for i, tc in enumerate(tool_calls):
                func = tc.get("function", {})
                tool_name = func.get("name")
                args = json.loads(func.get("arguments", "{}"))
                
                # Show tool execution
                output.print_info(f"→ Calling {tool_name}")
                
                # Execute tool (would be real MCP tool in production)
                with output.loading(f"Executing {tool_name}..."):
                    tool_result = self.execute_tool(tool_name, args)
                    await asyncio.sleep(0.3)  # Simulate work
                
                # Display result
                output.print_success(f"  ✓ {tool_name} completed")
                output.print_json(tool_result, indent=2)
        else:
            # Regular response without tools
            output.print_label("Assistant", style="green")
            response_text = result if isinstance(result, str) else result.get('response', '')
            output.print(response_text)
    
    def execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute a tool (simulated)."""
        # In real implementation, this would call MCP tools via chuk-tool-processor
        tool_map = {
            "list_files": lambda: {"files": ["file1.py", "file2.md"], "count": 2},
            "read_file": lambda: {"content": f"Content of {args.get('filename', 'unknown')}", "size": 1024},
            "run_command": lambda: {"output": f"Executed: {args.get('command', 'unknown')}", "status": "success"}
        }
        
        handler = tool_map.get(tool_name, lambda: {"error": "Unknown tool"})
        return handler()
    
    async def run(self):
        """Main chat loop."""
        try:
            # Setup signal handler for clean cancellation
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(signal.SIGINT, self.handle_interrupt)
            
            # Start conversation
            conv = await self.start_conversation()
            
            # Welcome message
            output.print_success("Chat ready! Type /help for commands.")
            output.print_separator()
            
            # Main loop
            while self.running:
                # Get user input with chuk-term prompt
                user_input = prompts.ask("You", style="cyan")
                
                if not user_input:
                    continue
                
                # Process input
                self.current_task = asyncio.current_task()
                should_continue = await self.handle_user_input(conv, user_input)
                
                if not should_continue:
                    break
            
            # Cleanup
            await self.cleanup(conv)
            
        except KeyboardInterrupt:
            output.print_warning("\nInterrupted by user")
        except Exception as e:
            output.print_error(f"Error: {e}")
            import traceback
            if output.is_verbose():
                traceback.print_exc()
        finally:
            # Restore terminal
            reset_terminal()
    
    def handle_interrupt(self):
        """Handle Ctrl+C gracefully."""
        output.print_warning("\n\nInterrupt received - exiting gracefully...")
        self.running = False
        if self.current_task:
            self.current_task.cancel()
    
    async def cleanup(self, conv):
        """Clean up resources."""
        output.print_separator()
        output.print_info("Closing conversation...")
        
        # Close conversation context
        if conv:
            await conv.__aexit__(None, None, None)
        
        # Show farewell
        output.print_success("Thank you for using MCP-CLI Chat!")
        output.print_dim("Session ended")


async def main():
    """Run the integrated chat demo."""
    # Show intro
    clear_screen()
    output.print_banner("""
╔══════════════════════════════════════════════════════════╗
║     MCP-CLI Chat - Chuk Libraries Integration Demo      ║
║                                                          ║
║  Demonstrates clean architecture using:                 ║
║  • chuk-llm for conversation management                 ║
║  • chuk-term for terminal UI                           ║
║  • chuk-tool-processor for MCP tools (simulated)       ║
╚══════════════════════════════════════════════════════════╝
    """.strip())
    
    await asyncio.sleep(1)
    
    # Run chat
    chat = ChukIntegratedChat()
    await chat.run()


if __name__ == "__main__":
    # Run with proper async handling
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")