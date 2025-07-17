#!/usr/bin/env python3
"""
Actual Streaming Bug Reproducer

Since the previous diagnostics show that everything works when we simulate
the tool calls, this reproducer focuses on the ACTUAL streaming processing
that happens when the LLM is called in the CLI.

It uses a REAL LLM call with streaming to reproduce the exact error.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List

try:
    from rich.console import Console
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    rprint = print


class ActualStreamingBugReproducer:
    """Reproduces the bug using actual LLM streaming calls."""
    
    def __init__(self):
        self.console = Console() if HAS_RICH else None
        self.setup_logging()
        
        # Components
        self.tool_manager = None
        self.chat_context = None
        
    def setup_logging(self):
        """Setup logging to capture all streaming details."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('actual_streaming_bug.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def reproduce_actual_streaming_bug(self, config_file: str, servers: List[str]) -> bool:
        """
        Reproduce the bug using actual LLM streaming calls.
        
        This makes a REAL streaming LLM call that should trigger the error.
        """
        self.log_info("🎯 Reproducing ACTUAL streaming bug with real LLM call")
        
        try:
            # Initialize components
            if not await self._initialize_components(config_file, servers):
                return False
            
            # Make a REAL streaming LLM call that should trigger the error
            await self._make_real_streaming_call()
            
            return True
            
        except Exception as e:
            self.log_error(f"Actual streaming bug reproduction failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            await self._cleanup()
    
    async def _initialize_components(self, config_file: str, servers: List[str]) -> bool:
        """Initialize components like the CLI does."""
        try:
            # Import CLI components
            from mcp_cli.tools.manager import ToolManager
            from mcp_cli.chat.chat_context import ChatContext
            
            # Initialize ToolManager
            self.tool_manager = ToolManager(config_file, servers)
            success = await self.tool_manager.initialize()
            if not success:
                self.log_error("ToolManager initialization failed")
                return False
            
            # Create ChatContext with OpenAI (to get streaming)
            self.chat_context = ChatContext.create(
                tool_manager=self.tool_manager,
                provider="openai",
                model="gpt-4o-mini"
            )
            
            # Initialize ChatContext
            if not await self.chat_context.initialize():
                self.log_error("ChatContext initialization failed") 
                return False
            
            self.log_success("Components initialized successfully")
            
            # Log tool information
            self.log_debug(f"Available tools: {len(self.chat_context.openai_tools)}")
            self.log_debug(f"Tool name mapping: {self.chat_context.tool_name_mapping}")
            
            return True
            
        except Exception as e:
            self.log_error(f"Component initialization failed: {e}")
            return False
    
    async def _make_real_streaming_call(self):
        """Make a real streaming LLM call that should trigger the error."""
        self.log_info("📞 Making REAL streaming LLM call")
        
        # Set up conversation history
        self.chat_context.conversation_history = [
            {"role": "system", "content": "You are a helpful assistant with access to database tools."},
            {"role": "user", "content": "select top 10 products from the database"}
        ]
        
        # Log the conversation
        self.log_debug("Conversation history:")
        for i, msg in enumerate(self.chat_context.conversation_history):
            self.log_debug(f"  {i}: {msg['role']} - {msg['content'][:50]}...")
        
        # Get the LLM client
        client = self.chat_context.client
        self.log_debug(f"Using client: {type(client)}")
        
        # Make the streaming call with tools
        try:
            # Import streaming handler
            from mcp_cli.chat.streaming_handler import StreamingResponseHandler
            
            # Create streaming handler
            streaming_handler = StreamingResponseHandler(self.console)
            
            self.log_info("🌊 Starting streaming LLM call...")
            
            # This should reproduce the actual streaming process that causes the error
            completion = await streaming_handler.stream_response(
                client=client,
                messages=self.chat_context.conversation_history,
                tools=self.chat_context.openai_tools
            )
            
            self.log_info(f"✅ Streaming call completed: {completion.get('streaming', False)}")
            
            # Extract and process tool calls like the real system does
            tool_calls = completion.get("tool_calls", [])
            
            if tool_calls:
                self.log_info(f"🔧 Processing {len(tool_calls)} tool calls from streaming")
                await self._process_streaming_tool_calls(tool_calls)
            else:
                self.log_info("ℹ️ No tool calls returned from streaming")
                
        except Exception as e:
            self.log_error(f"❌ Streaming call failed: {e}")
            import traceback
            traceback.print_exc()
            
            # This might be where we see the actual error!
    
    async def _process_streaming_tool_calls(self, tool_calls: List[Dict[str, Any]]):
        """Process tool calls from streaming like the real system does."""
        self.log_info("⚡ Processing streaming tool calls with ToolProcessor")
        
        try:
            # Import and use the actual ToolProcessor
            from mcp_cli.chat.tool_processor import ToolProcessor
            from mcp_cli.chat.ui_manager import ChatUIManager
            
            # Create UI manager
            ui_manager = ChatUIManager(self.chat_context)
            
            # Create tool processor
            tool_processor = ToolProcessor(self.chat_context, ui_manager)
            
            # Get name mapping
            name_mapping = getattr(self.chat_context, "tool_name_mapping", {})
            
            self.log_debug("Tool calls to process:")
            for i, tc in enumerate(tool_calls):
                self.log_debug(f"  {i}: {tc}")
            
            self.log_debug(f"Name mapping: {name_mapping}")
            
            # Process tool calls - this is where the error should occur
            self.log_info("🚨 Calling tool_processor.process_tool_calls() - ERROR EXPECTED")
            await tool_processor.process_tool_calls(tool_calls, name_mapping)
            
            self.log_success("✅ Tool calls processed successfully!")
            
        except Exception as e:
            self.log_error(f"❌ Tool call processing failed: {e}")
            self.log_error("🎯 This is likely the source of the 'Missing required parameter' error!")
            
            # Log detailed error information
            import traceback
            error_details = traceback.format_exc()
            self.log_error(f"Full error traceback:\n{error_details}")
            
            # Try to extract specific error details
            if "Missing required parameter" in str(e):
                self.log_error("🔍 FOUND THE BUG: Missing required parameter error!")
                
                # Log the tool calls that caused the error
                self.log_error("Tool calls that caused the error:")
                for i, tc in enumerate(tool_calls):
                    self.log_error(f"  Tool call {i}:")
                    self.log_error(f"    Function: {tc.get('function', {})}")
                    self.log_error(f"    Arguments: {tc.get('function', {}).get('arguments', 'N/A')}")
    
    async def _cleanup(self):
        """Cleanup resources."""
        if self.tool_manager:
            try:
                await self.tool_manager.close()
            except Exception as e:
                self.log_warning(f"Cleanup failed: {e}")
    
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
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Actual Streaming Bug Reproducer")
    parser.add_argument("--config", default="server_config.json", help="Server config file")
    parser.add_argument("--server", action="append", default=[], help="Server names")
    
    args = parser.parse_args()
    
    if not args.server:
        args.server = ["sqlite"]
    
    reproducer = ActualStreamingBugReproducer()
    
    try:
        success = await reproducer.reproduce_actual_streaming_bug(args.config, args.server)
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nReproduction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Reproduction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())