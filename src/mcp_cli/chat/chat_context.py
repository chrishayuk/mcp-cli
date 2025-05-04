# mcp_cli/chat/chat_context.py
from rich import print
from rich.console import Console

# llm imports
from mcp_cli.llm.llm_client import get_llm_client

# cli imports
from mcp_cli.chat.system_prompt import generate_system_prompt

# Import the centralized tool manager
from mcp_cli.tools.manager import ToolManager
from mcp_cli.tools.models import ToolInfo

class ChatContext:
    """Class to manage the chat context and state."""
    
    def __init__(self, tool_manager: ToolManager, provider="openai", model="gpt-4o-mini"):
        """
        Initialize the chat context.
        
        Args:
            tool_manager: ToolManager instance (required)
            provider: LLM provider name (default: "openai")
            model: LLM model name (default: "gpt-4o-mini")
        """
        self.tool_manager = tool_manager
        self.provider = provider
        self.model = model
        self.exit_requested = False
        self.conversation_history = []
        
        # Initialize the client right away to ensure it's never None
        self.client = get_llm_client(provider=self.provider, model=self.model)
        
    # ------------------------------------------------------------------ #
    # initialisation – no duplicate tools                                #
    # ------------------------------------------------------------------ #
    async def initialize(self) -> bool:
        """
        Build the runtime context.

        *Display* lists should contain each real tool only once, even though
        the registry keeps an extra alias in the ``default`` namespace.
        """
        console = Console()
        with console.status(
            "[bold cyan]Setting up chat environment…[/bold cyan]", spinner="dots"
        ):
            # ── 1. get unique tools (no “default” duplicates) ────────────
            if hasattr(self.tool_manager, "get_unique_tools"):
                tool_infos = self.tool_manager.get_unique_tools()
            else:  # fallback: filter manually
                tool_infos = [
                    t
                    for t in self.tool_manager.get_all_tools()
                    if t.namespace != "default"
                ]

            # keep both a simple-dict *display* list and the “internal” copy
            self.tools: list[dict] = []
            self.internal_tools: list[dict] = []
            for t in tool_infos:
                t_dict = {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "namespace": t.namespace,
                }
                self.tools.append(t_dict)          # for UI tables
                self.internal_tools.append(t_dict)  # for system-prompt

            # ── 2. servers & helper maps ────────────────────────────────
            self.server_info = self._convert_server_info(
                self.tool_manager.get_server_info()
            )
            self.tool_to_server_map = {t.name: t.namespace for t in tool_infos}

            # place-holders kept for bw-compat
            self.namespaced_tool_map: dict = {}
            self.original_to_namespaced: dict = {}

        if not self.tools:
            print(
                "[yellow]No tools available. Chat functionality may be "
                "limited.[/yellow]"
            )

        # ── 3. system-prompt & OpenAI-style specs ───────────────────────
        system_prompt = generate_system_prompt(self.internal_tools)
        self.openai_tools = self.tool_manager.get_tools_for_llm()

        self.conversation_history = [{"role": "system", "content": system_prompt}]
        return True


    def _convert_server_info(self, server_infos):
        """Convert ServerInfo objects to the expected dict format."""
        result = []
        for server in server_infos:
            result.append({
                'id': server.id,
                'name': server.name,
                'tools': server.tool_count,
                'status': server.status
            })
        return result
    
    def get_server_for_tool(self, tool_name):
        """Get the server name that a tool belongs to."""
        return self.tool_manager.get_server_for_tool(tool_name) or "Unknown"
    
    def get_display_name_for_tool(self, namespaced_tool_name):
        """Get the display name (non-namespaced) for a tool if available."""
        # In CHUK, tools already have the right names
        return namespaced_tool_name
    
    def to_dict(self):
        """Convert the context to a dictionary for command handling."""
        return {
            "conversation_history": self.conversation_history,
            "tools": self.tools,  # Display tools
            "internal_tools": self.internal_tools,  # Namespaced tools
            "client": self.client,
            "provider": self.provider,
            "model": self.model,
            "server_info": self.server_info,
            "openai_tools": self.openai_tools,
            "exit_requested": self.exit_requested,
            "tool_to_server_map": self.tool_to_server_map,
            "namespaced_tool_map": self.namespaced_tool_map,
            "original_to_namespaced": self.original_to_namespaced,
            "tool_manager": self.tool_manager  # Include tool_manager in the dict
        }
        
    def update_from_dict(self, context_dict):
        """Update context from dictionary (after command handling)."""
        self.exit_requested = context_dict.get('exit_requested', self.exit_requested)
        
        # Also update the client if it was changed
        if 'client' in context_dict and context_dict['client'] is not None:
            self.client = context_dict['client']
        
        # Keep backward compatibility with these maps if they're used
        if 'namespaced_tool_map' in context_dict:
            self.namespaced_tool_map = context_dict['namespaced_tool_map']
        if 'original_to_namespaced' in context_dict:
            self.original_to_namespaced = context_dict['original_to_namespaced']