# src/cli/chat/chat_context.py
from rich import print
from rich.console import Console

# imports
from mcpcli.llm_client import LLMClient
from mcpcli.tools_handler import convert_to_openai_tools, fetch_tools
from cli.chat.system_prompt import generate_system_prompt

class ChatContext:
    """Class to manage the chat context and state."""
    
    def __init__(self, server_streams, provider="openai", model="gpt-4o-mini"):
        self.server_streams = server_streams
        self.provider = provider
        self.model = model
        self.server_info = []
        self.tools = []
        self.openai_tools = []
        self.client = None
        self.conversation_history = []
        self.exit_requested = False
        
    async def initialize(self):
        """Initialize the chat context by fetching tools and setting up the client."""
        console = Console()
        
        with console.status("[bold cyan]Fetching available tools...[/bold cyan]", spinner="dots"):
            for i, (read_stream, write_stream) in enumerate(self.server_streams):
                try:
                    server_name = f"Server {i+1}"
                    fetched_tools = await fetch_tools(read_stream, write_stream)
                    self.tools.extend(fetched_tools)
                    self.server_info.append({
                        "id": i+1,
                        "name": server_name,
                        "tools": len(fetched_tools),
                        "status": "Connected"
                    })
                except Exception as e:
                    self.server_info.append({
                        "id": i+1,
                        "name": f"Server {i+1}",
                        "tools": 0,
                        "status": f"Error: {str(e)}"
                    })
                    print(f"[yellow]Warning: Failed to fetch tools from Server {i+1}: {e}[/yellow]")
                    continue

        if not self.tools:
            print("[red]No tools available. Exiting chat mode.[/red]")
            return False

        print(f"[green]Loaded {len(self.tools)} tools successfully.[/green]")
        
        # Generate system prompt and convert tools to OpenAI format
        system_prompt = generate_system_prompt(self.tools)
        self.openai_tools = convert_to_openai_tools(self.tools)
        
        # Initialize the LLM client
        self.client = LLMClient(provider=self.provider, model=self.model)
        self.conversation_history = [{"role": "system", "content": system_prompt}]
        
        return True
    
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
            "exit_requested": self.exit_requested
        }
        
    def update_from_dict(self, context_dict):
        """Update context from dictionary (after command handling)."""
        self.exit_requested = context_dict.get('exit_requested', self.exit_requested)