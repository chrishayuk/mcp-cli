# mcp_cli/chat/chat_context.py
from rich import print
from rich.console import Console

# llm imports
from mcp_cli.llm.llm_client import get_llm_client
from mcp_cli.llm.tools_handler import convert_to_openai_tools, fetch_tools

# cli imports
from mcp_cli.chat.system_prompt import generate_system_prompt

class ChatContext:
    """Class to manage the chat context and state."""
    
    def __init__(self, server_streams, provider="openai", model="gpt-4o-mini", server_names=None):
        self.server_streams = server_streams
        self.provider = provider
        self.model = model
        self.server_info = []
        self.tools = []
        self.openai_tools = []
        self.conversation_history = []
        self.exit_requested = False
        self.tool_to_server_map = {}  # Maps tool names to server names
        
        # Store server names mapping
        self.server_names = server_names or {}
        
        # Initialize the client right away to ensure it's never None
        self.client = get_llm_client(provider=self.provider, model=self.model)
        
    async def initialize(self):
        """Initialize the chat context by fetching tools and setting up the client."""
        console = Console()
        
        with console.status("[bold cyan]Fetching available tools...[/bold cyan]", spinner="dots"):
            tool_index = 0
            for i, (read_stream, write_stream) in enumerate(self.server_streams):
                try:
                    # Get server name from mapping or use default
                    if isinstance(self.server_names, dict) and i in self.server_names:
                        server_name = self.server_names[i]
                    elif isinstance(self.server_names, list) and i < len(self.server_names):
                        server_name = self.server_names[i]
                    else:
                        server_name = f"Server {i+1}"
                    
                    fetched_tools = await fetch_tools(read_stream, write_stream)
                    
                    # Skip if None or empty
                    if not fetched_tools:
                        print(f"[yellow]Warning: No tools returned from {server_name}[/yellow]")
                        continue
                        
                    # Map each tool to its server
                    for tool in fetched_tools:
                        self.tool_to_server_map[tool["name"]] = server_name
                    
                    self.tools.extend(fetched_tools)
                    self.server_info.append({
                        "id": i+1,
                        "name": server_name,
                        "tools": len(fetched_tools),
                        "status": "Connected",
                        "tool_start_index": tool_index
                    })
                    tool_index += len(fetched_tools)
                except Exception as e:
                    # Use the proper server name in error message if available
                    server_name = server_name if 'server_name' in locals() else f"Server {i+1}"
                    
                    self.server_info.append({
                        "id": i+1,
                        "name": server_name,
                        "tools": 0,
                        "status": f"Error: {str(e)}",
                        "tool_start_index": tool_index
                    })
                    print(f"[yellow]Warning: Failed to fetch tools from {server_name}: {e}[/yellow]")
                    continue

        if not self.tools:
            print("[yellow]No tools available. Chat functionality may be limited.[/yellow]")
            # Don't exit - we can still chat without tools
            
        # Generate system prompt and convert tools to OpenAI format
        system_prompt = generate_system_prompt(self.tools)
        self.openai_tools = convert_to_openai_tools(self.tools)
        
        # Initialize the conversation history with the system prompt
        self.conversation_history = [{"role": "system", "content": system_prompt}]
        
        return True
    
    def get_server_for_tool(self, tool_name):
        """Get the server name that a tool belongs to."""
        return self.tool_to_server_map.get(tool_name, "Unknown")
    
    def to_dict(self):
        """Convert the context to a dictionary for command handling."""
        return {
            "conversation_history": self.conversation_history,
            "tools": self.tools,
            "client": self.client,
            "provider": self.provider,
            "model": self.model,
            "server_info": self.server_info,
            "server_streams": self.server_streams,
            "openai_tools": self.openai_tools,
            "exit_requested": self.exit_requested,
            "tool_to_server_map": self.tool_to_server_map
        }
        
    def update_from_dict(self, context_dict):
        """Update context from dictionary (after command handling)."""
        self.exit_requested = context_dict.get('exit_requested', self.exit_requested)
        
        # Also update the client if it was changed
        if 'client' in context_dict and context_dict['client'] is not None:
            self.client = context_dict['client']