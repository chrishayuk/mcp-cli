#!/usr/bin/env python3
"""
CLI Conversation Flow Diagnostic

This diagnostic reproduces the EXACT conversation flow that happens in the CLI:
1. User input: "select top 10 products from the database"
2. ChatContext initialization with tools
3. LLM client creation and streaming
4. StreamingResponseHandler processing
5. ToolProcessor execution
6. ConversationProcessor coordination

This should reproduce the "Missing required parameter 'query'" error.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    rprint = print


class ConversationFlowDiagnostic:
    """Diagnostic that reproduces the exact CLI conversation flow."""
    
    def __init__(self):
        self.console = Console() if HAS_RICH else None
        self.setup_logging()
        
        # Components that will be initialized
        self.tool_manager = None
        self.chat_context = None
        self.conversation_processor = None
        self.streaming_handler = None
        
    def setup_logging(self):
        """Setup comprehensive logging to track the entire flow."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('conversation_flow_diagnostic.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def diagnose_conversation_flow(self, config_file: str, servers: List[str]) -> bool:
        """
        Diagnose the complete conversation flow that causes the error.
        
        This reproduces the exact sequence that happens when a user types:
        "select top 10 products from the database" in the CLI.
        """
        self.log_info("🎯 Starting CLI conversation flow diagnostic")
        self.log_info("Reproducing: User types 'select top 10 products from the database'")
        
        try:
            # Step 1: Initialize like the CLI does
            if not await self._initialize_cli_components(config_file, servers):
                return False
                
            # Step 2: Simulate the exact user interaction
            user_input = "select top 10 products from the database"
            await self._simulate_user_input(user_input)
            
            # Step 3: Test the conversation processing pipeline
            await self._test_conversation_processing()
            
            return True
            
        except Exception as e:
            self.log_error(f"Conversation flow diagnostic failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            await self._cleanup()
    
    async def _initialize_cli_components(self, config_file: str, servers: List[str]) -> bool:
        """Initialize components exactly like the CLI does."""
        self.log_info("🔧 Initializing CLI components")
        
        try:
            # Import CLI components
            from mcp_cli.tools.manager import ToolManager
            from mcp_cli.chat.chat_context import ChatContext
            from mcp_cli.chat.conversation import ConversationProcessor
            from mcp_cli.chat.ui_manager import ChatUIManager
            from mcp_cli.chat.streaming_handler import StreamingResponseHandler
            
            # Step 1: Initialize ToolManager (like handle_chat_mode does)
            self.log_debug("Initializing ToolManager...")
            self.tool_manager = ToolManager(config_file, servers)
            success = await self.tool_manager.initialize()
            if not success:
                self.log_error("ToolManager initialization failed")
                return False
            
            self.log_success("ToolManager initialized")
            
            # Step 2: Create ChatContext (like ChatContext.create does)
            self.log_debug("Creating ChatContext...")
            self.chat_context = ChatContext.create(
                tool_manager=self.tool_manager,
                provider="openai",  # Default provider
                model="gpt-4o-mini"  # Default model
            )
            
            # Step 3: Initialize ChatContext (discovers tools, sets up LLM client)
            if not await self.chat_context.initialize():
                self.log_error("ChatContext initialization failed")
                return False
                
            self.log_success("ChatContext initialized")
            self.log_debug(f"Available tools: {len(self.chat_context.tools)}")
            self.log_debug(f"OpenAI tools: {len(self.chat_context.openai_tools)}")
            
            # Step 4: Create UI manager and conversation processor
            ui_manager = ChatUIManager(self.chat_context)
            self.conversation_processor = ConversationProcessor(self.chat_context, ui_manager)
            
            # Step 5: Create streaming handler
            self.streaming_handler = StreamingResponseHandler()
            
            self.log_success("All CLI components initialized successfully")
            return True
            
        except Exception as e:
            self.log_error(f"Component initialization failed: {e}")
            return False
    
    async def _simulate_user_input(self, user_input: str):
        """Simulate user input exactly like the CLI does."""
        self.log_info(f"👤 Simulating user input: '{user_input}'")
        
        # Add user message to conversation history (like UI does)
        self.chat_context.add_user_message(user_input)
        
        # Log the current conversation state
        self.log_debug(f"Conversation history length: {len(self.chat_context.conversation_history)}")
        for i, msg in enumerate(self.chat_context.conversation_history):
            self.log_debug(f"  Message {i}: {msg['role']} - {msg['content'][:50]}...")
    
    async def _test_conversation_processing(self):
        """
        Test the conversation processing that triggers the error.
        
        This goes through the exact same flow as ConversationProcessor.process_conversation()
        """
        self.log_info("💬 Testing conversation processing pipeline")
        
        try:
            # This is the exact flow from ConversationProcessor.process_conversation()
            
            # Step 1: Check if client supports streaming (like conversation.py does)
            client = self.chat_context.client
            supports_streaming = hasattr(client, 'create_completion')
            
            self.log_debug(f"Client supports streaming: {supports_streaming}")
            self.log_debug(f"Client type: {type(client)}")
            
            # Step 2: Load tools if not already loaded (like conversation.py does)
            if not getattr(self.chat_context, "openai_tools", None):
                await self._load_tools_like_conversation()
            
            # Step 3: Test both streaming and non-streaming paths
            if supports_streaming:
                self.log_info("🌊 Testing streaming completion path")
                await self._test_streaming_completion()
            else:
                self.log_info("📝 Testing regular completion path")
                await self._test_regular_completion()
                
        except Exception as e:
            self.log_error(f"Conversation processing test failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def _load_tools_like_conversation(self):
        """Load tools exactly like conversation.py does."""
        self.log_debug("Loading tools like ConversationProcessor._load_tools()")
        
        try:
            # This is the exact logic from conversation.py _load_tools method
            if hasattr(self.chat_context.tool_manager, "get_adapted_tools_for_llm"):
                provider = getattr(self.chat_context, 'provider', 'openai')
                tools_and_mapping = await self.chat_context.tool_manager.get_adapted_tools_for_llm(provider)
                self.chat_context.openai_tools = tools_and_mapping[0]
                self.chat_context.tool_name_mapping = tools_and_mapping[1]
                
                self.log_debug(f"Loaded {len(self.chat_context.openai_tools)} adapted tools for {provider}")
                self.log_debug(f"Tool name mapping: {self.chat_context.tool_name_mapping}")
                
        except Exception as e:
            self.log_error(f"Tool loading failed: {e}")
            self.chat_context.openai_tools = []
            self.chat_context.tool_name_mapping = {}
    
    async def _test_streaming_completion(self):
        """Test streaming completion that likely contains the bug."""
        self.log_info("🔄 Testing streaming completion (where error likely occurs)")
        
        try:
            # Create a mock LLM response that would trigger tool calls
            mock_streaming_response = self._create_mock_streaming_response()
            
            # Test streaming handler processing
            await self._test_streaming_handler_processing(mock_streaming_response)
            
            # Test tool call extraction and execution
            await self._test_tool_call_extraction_and_execution()
            
        except Exception as e:
            self.log_error(f"Streaming completion test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_mock_streaming_response(self) -> Dict[str, Any]:
        """
        Create a mock streaming response that would come from the LLM.
        
        This simulates what the LLM would return for "select top 10 products from the database"
        """
        self.log_debug("Creating mock LLM streaming response")
        
        # This simulates the streaming response that would trigger the bug
        mock_response = {
            "response": "I'll help you select the top 10 products. Let me check the database structure first.",
            "tool_calls": [
                {
                    "id": "call_list_tables_123",
                    "type": "function", 
                    "function": {
                        "name": "stdio_list_tables",  # Note: sanitized name from universal compatibility
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
            ],
            "chunks_received": 10,
            "elapsed_time": 2.5,
            "streaming": True,
            "interrupted": False
        }
        
        self.log_debug(f"Mock response created with {len(mock_response['tool_calls'])} tool calls")
        return mock_response
    
    async def _test_streaming_handler_processing(self, mock_response: Dict[str, Any]):
        """Test how the streaming handler processes the response."""
        self.log_info("🔧 Testing StreamingResponseHandler processing")
        
        tool_calls = mock_response.get("tool_calls", [])
        
        # Log what we're processing
        for i, tc in enumerate(tool_calls):
            self.log_debug(f"Tool call {i}: {tc}")
        
        # Test streaming validation like conversation.py does
        for i, tc in enumerate(tool_calls):
            if not self._validate_streaming_tool_call(tc):
                self.log_error(f"Invalid tool call structure from streaming: {tc}")
                # Try to fix it
                fixed_tc = self._fix_tool_call_structure(tc)
                if fixed_tc:
                    tool_calls[i] = fixed_tc
                    self.log_debug(f"Fixed tool call {i}: {fixed_tc}")
                else:
                    self.log_error(f"Could not fix tool call {i}")
    
    def _validate_streaming_tool_call(self, tool_call: dict) -> bool:
        """Validate streaming tool call like conversation.py does."""
        try:
            if not isinstance(tool_call, dict):
                return False
            
            if "function" not in tool_call:
                return False
            
            function = tool_call["function"]
            if not isinstance(function, dict):
                return False
            
            if "name" not in function or not function["name"]:
                return False
            
            if "arguments" in function:
                args = function["arguments"]
                if isinstance(args, str):
                    try:
                        if args.strip():
                            json.loads(args)
                    except json.JSONDecodeError:
                        self.log_warning(f"Invalid JSON arguments in tool call: {args}")
                        return False
                elif not isinstance(args, dict):
                    return False
            
            return True
            
        except Exception as e:
            self.log_error(f"Error validating streaming tool call: {e}")
            return False
    
    def _fix_tool_call_structure(self, tool_call: dict) -> Optional[dict]:
        """Fix tool call structure like conversation.py does."""
        try:
            fixed = dict(tool_call)
            
            if "id" not in fixed:
                fixed["id"] = f"call_{hash(str(tool_call)) % 10000}"
            
            if "type" not in fixed:
                fixed["type"] = "function"
            
            if "function" not in fixed:
                return None
            
            function = fixed["function"]
            
            if not function.get("name"):
                return None
            
            if "arguments" not in function:
                function["arguments"] = "{}"
            elif function["arguments"] is None:
                function["arguments"] = "{}"
            elif isinstance(function["arguments"], dict):
                function["arguments"] = json.dumps(function["arguments"])
            elif not isinstance(function["arguments"], str):
                function["arguments"] = str(function["arguments"])
            
            if self._validate_streaming_tool_call(fixed):
                return fixed
            else:
                return None
                
        except Exception as e:
            self.log_error(f"Error fixing tool call structure: {e}")
            return None
    
    async def _test_tool_call_extraction_and_execution(self):
        """Test the tool call extraction and execution that causes the error."""
        self.log_info("⚡ Testing tool call extraction and execution")
        
        # Create the tool calls that would come from streaming
        tool_calls = [
            {
                "id": "call_list_tables_123",
                "type": "function",
                "function": {
                    "name": "stdio_list_tables",
                    "arguments": "{}"
                }
            },
            {
                "id": "call_read_query_456", 
                "type": "function",
                "function": {
                    "name": "stdio_read_query",
                    "arguments": '{"query": "SELECT * FROM products LIMIT 10"}'
                }
            }
        ]
        
        # Get name mapping like the real system
        name_mapping = getattr(self.chat_context, "tool_name_mapping", {})
        self.log_debug(f"Using name mapping: {name_mapping}")
        
        # Process tool calls like ToolProcessor does
        try:
            # Import the actual tool processor to test the real logic
            from mcp_cli.chat.tool_processor import ToolProcessor
            
            # Create a mock UI manager for the tool processor
            class MockUIManager:
                def print_tool_call(self, name, args):
                    self.log_info(f"Mock UI: Tool call {name} with {args}")
                    
                def finish_tool_calls(self):
                    pass
                    
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
            
            mock_ui = MockUIManager()
            tool_processor = ToolProcessor(self.chat_context, mock_ui)
            
            # This is where the error should occur!
            self.log_info("🚨 Processing tool calls (ERROR EXPECTED HERE)")
            await tool_processor.process_tool_calls(tool_calls, name_mapping)
            
            self.log_success("Tool calls processed successfully - no error occurred!")
            
        except Exception as e:
            self.log_error(f"❌ Tool call processing failed: {e}")
            self.log_error("This is likely the source of the 'Missing required parameter' error!")
            import traceback
            traceback.print_exc()
    
    async def _test_regular_completion(self):
        """Test regular (non-streaming) completion as fallback."""
        self.log_info("📝 Testing regular completion")
        
        try:
            # This would be the fallback path
            self.log_debug("Regular completion would bypass streaming issues")
            
        except Exception as e:
            self.log_error(f"Regular completion test failed: {e}")
    
    async def _cleanup(self):
        """Cleanup resources."""
        if self.tool_manager:
            try:
                await self.tool_manager.close()
            except Exception as e:
                self.log_warning(f"Tool manager cleanup failed: {e}")
    
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
    
    parser = argparse.ArgumentParser(description="CLI Conversation Flow Diagnostic")
    parser.add_argument("--config", default="server_config.json", help="Server config file")
    parser.add_argument("--server", action="append", default=[], help="Server names")
    
    args = parser.parse_args()
    
    if not args.server:
        args.server = ["sqlite"]
    
    diagnostic = ConversationFlowDiagnostic()
    
    try:
        success = await diagnostic.diagnose_conversation_flow(args.config, args.server)
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