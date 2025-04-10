# mcp_cli/chat/chat_context.py
from rich import print
from rich.console import Console

# llm imports
from mcp_cli.llm.llm_client import get_llm_client
from mcp_cli.llm.tools_handler import convert_to_openai_tools

# cli imports
from mcp_cli.chat.system_prompt import generate_system_prompt

# Import our stream manager
from mcp_cli.stream_manager import StreamManager

class ChatContext:
    """Class to manage the chat context and state."""
    
    def __init__(self, stream_manager, provider="openai", model="gpt-4o-mini"):
        """
        Initialize the chat context.
        
        Args:
            stream_manager: StreamManager instance (required)
            provider: LLM provider name (default: "openai")
            model: LLM model name (default: "gpt-4o-mini")
        """
        self.stream_manager = stream_manager
        self.provider = provider
        self.model = model
        self.exit_requested = False
        self.conversation_history = []
        
        # Initialize the client right away to ensure it's never None
        self.client = get_llm_client(provider=self.provider, model=self.model)
        
    async def initialize(self):
        """Initialize the chat context by setting up the tools and system prompt."""
        console = Console()
        
        with console.status("[bold cyan]Setting up chat environment...[/bold cyan]", spinner="dots"):
            # Get all data from the stream manager
            self.tools = self.stream_manager.get_all_tools()
            self.server_info = self.stream_manager.get_server_info()
            self.tool_to_server_map = self.stream_manager.tool_to_server_map

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
        return self.stream_manager.get_server_for_tool(tool_name)
    
    def to_dict(self):
        """Convert the context to a dictionary for command handling."""
        return {
            "conversation_history": self.conversation_history,
            "tools": self.tools,
            "client": self.client,
            "provider": self.provider,
            "model": self.model,
            "server_info": self.server_info,
            "openai_tools": self.openai_tools,
            "exit_requested": self.exit_requested,
            "tool_to_server_map": self.tool_to_server_map,
            "stream_manager": self.stream_manager  # Include stream_manager in the dict
        }
        
    def update_from_dict(self, context_dict):
        """Update context from dictionary (after command handling)."""
        self.exit_requested = context_dict.get('exit_requested', self.exit_requested)
        
        # Also update the client if it was changed
        if 'client' in context_dict and context_dict['client'] is not None:
            self.client = context_dict['client']