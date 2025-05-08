# mcp_cli/chat/chat_context.py
"""
Chat context handling for the MCP CLI.

This module glues the UI / REPL layer to either a *ToolManager* (the normal
runtime in the CLI) **or** a bare-bones "stream-manager–like" object that is
used by the unit-tests.  The public surface of *ChatContext* therefore works
with *either* source of tools:

* Production code passes a **ToolManager** instance.
* The test-suite instantiates the class with **stream_manager=…**.

Both execution paths build identical data-structures so the rest of the CLI
doesn't care where the information originally came from.
"""

from __future__ import annotations

import asyncio
import gc
from typing import Any, Dict, List, Optional

from rich import print
from rich.console import Console

# LLM utilities
from mcp_cli.llm.llm_client import get_llm_client

# Prompt generator
from mcp_cli.chat.system_prompt import generate_system_prompt

# Tools – only imported to expose convert_to_openai_tools for monkey-patching
from mcp_cli.tools.manager import ToolManager

# Provider configuration
from mcp_cli.provider_config import ProviderConfig

# The tests monkey-patch this symbol, so expose it at module level
convert_to_openai_tools = ToolManager.convert_to_openai_tools


class ChatContext:
    """Manage the end-to-end conversation state for the CLI session."""

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        tool_manager: Optional[ToolManager] = None,
        stream_manager: Optional[Any] = None,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        provider_config: Optional[ProviderConfig] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Create a new chat context.

        Exactly one of *tool_manager* **or** *stream_manager* must be
        supplied.

        Parameters
        ----------
        tool_manager
            Fully-featured tool manager used in the normal CLI runtime.
        stream_manager
            Minimal "manager" used by the test-suite.  Needs to expose
            ``get_all_tools()`` (or ``get_internal_tools()``),
            ``get_server_info()`` and ``get_server_for_tool()``.
        provider / model
            Which LLM backend to use.
        provider_config
            Optional ProviderConfig instance for LLM configurations.
        api_base / api_key
            Optional API settings that override provider_config.
        """
        if (tool_manager is None) == (stream_manager is None):
            raise ValueError("Pass either tool_manager *or* stream_manager, not both")

        self.tool_manager: Optional[ToolManager] = tool_manager
        self.stream_manager: Optional[Any] = stream_manager

        self.provider = provider
        self.model = model
        self.exit_requested: bool = False
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Initialize provider configuration
        self.provider_config = provider_config or ProviderConfig()
        
        # Update provider config if API settings were provided
        if api_base or api_key:
            config_updates = {}
            if api_base:
                config_updates["api_base"] = api_base
            if api_key:
                config_updates["api_key"] = api_key
                
            self.provider_config.set_provider_config(provider, config_updates)

        # initialise LLM client immediately so it's never None
        self.client = get_llm_client(
            provider=self.provider, 
            model=self.model,
            config=self.provider_config
        )

        # attributes filled in initialise()
        self.tools: List[Dict[str, Any]] = []
        self.internal_tools: List[Dict[str, Any]] = []
        self.server_info: List[Dict[str, Any]] = []
        self.tool_to_server_map: Dict[str, str] = {}
        self.openai_tools: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # initial setup                                                      #
    # ------------------------------------------------------------------ #
    async def initialize(self) -> bool:  # noqa: C901  (a bit long but straightforward)
        """
        Build the runtime context (tool list, server information, …).

        When a *ToolManager* is present we filter out the duplicate tools that
        live in the ``default`` namespace.  The stripped-down stream-manager
        used in unit-tests already exposes a flat list without duplicates.
        """
        console = Console()
        with console.status("[bold cyan]Setting up chat environment…[/bold cyan]", spinner="dots"):
            # ---- 1. obtain tool list -----------------------------------
            if self.tool_manager is not None:
                # production path — pull from ToolManager
                if hasattr(self.tool_manager, "get_unique_tools"):
                    tool_infos = self.tool_manager.get_unique_tools()
                else:                                     # pragma: no cover
                    tool_infos = [
                        t for t in self.tool_manager.get_all_tools()
                        if t.namespace != "default"
                    ]

                self.tools = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                        "namespace": t.namespace,
                    }
                    for t in tool_infos
                ]
                # identical copy for the system prompt
                self.internal_tools = list(self.tools)

                # servers
                raw_infos = self.tool_manager.get_server_info() or []
                self.server_info = self._convert_server_info(raw_infos)
                self.tool_to_server_map = {t["name"]: t["namespace"] for t in self.tools}

            else:
                # unit-test / lightweight path — pull directly
                if hasattr(self.stream_manager, "get_internal_tools"):
                    self.tools = list(self.stream_manager.get_internal_tools())
                else:
                    self.tools = list(self.stream_manager.get_all_tools())

                self.internal_tools = list(self.tools)

                self.server_info = list(self.stream_manager.get_server_info())
                self.tool_to_server_map = {
                    t["name"]: self.stream_manager.get_server_for_tool(t["name"])
                    for t in self.tools
                }

        # ---- 2. warn on empty tool-set ---------------------------------
        if not self.tools:
            print("[yellow]No tools available. Chat functionality may be limited.[/yellow]")

        # ---- 3. build system prompt & OpenAI-style spec ----------------
        system_prompt = generate_system_prompt(self.internal_tools)
        # Use the *module-level* convert_to_openai_tools — tests replace it
        self.openai_tools = convert_to_openai_tools(self.tools)

        self.conversation_history = [{"role": "system", "content": system_prompt}]
        return True

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _convert_server_info(server_infos):
        """Convert *ToolManager*'s ServerInfo objects into plain dictionaries."""
        result = []
        for server in server_infos:
            result.append(
                {
                    "id": server.id,
                    "name": server.name,
                    "tools": server.tool_count,
                    "status": server.status,
                }
            )
        return result

    # .................................................................. #
    # convenience wrappers                                               #
    # .................................................................. #
    def get_server_for_tool(self, tool_name: str) -> str:
        """
        Return the human-readable server name for *tool_name*.

        Falls back to ``"Unknown"`` if the mapping isn't available.
        """
        if self.tool_manager is not None:
            return self.tool_manager.get_server_for_tool(tool_name) or "Unknown"
        return self.stream_manager.get_server_for_tool(tool_name) or "Unknown"

    @staticmethod
    def get_display_name_for_tool(namespaced_tool_name: str) -> str:  # noqa: D401
        """Return a user-friendly name – here we already have one-to-one."""
        return namespaced_tool_name

    # .................................................................. #
    # serialisation helpers                                              #
    # .................................................................. #
    def to_dict(self) -> Dict[str, Any]:
        """Dump the context to a plain dict for command handlers / tests."""
        return {
            "conversation_history": self.conversation_history,
            "tools": self.tools,
            "internal_tools": self.internal_tools,
            "client": self.client,
            "provider": self.provider,
            "model": self.model,
            "provider_config": self.provider_config,  # Added provider_config
            "server_info": self.server_info,
            "openai_tools": self.openai_tools,
            "exit_requested": self.exit_requested,
            "tool_to_server_map": self.tool_to_server_map,
            "stream_manager": self.stream_manager,
            "tool_manager": self.tool_manager,
        }

    def update_from_dict(self, context_dict: Dict[str, Any]) -> None:
        """
        Update the context after it has been passed through a command handler.

        Only a subset of keys is honoured; missing keys are ignored so callers
        don't have to copy the entire structure back.
        """
        if "exit_requested" in context_dict:
            self.exit_requested = context_dict["exit_requested"]

        if "client" in context_dict and context_dict["client"] is not None:
            self.client = context_dict["client"]
            
        if "provider" in context_dict:
            self.provider = context_dict["provider"]
            
        if "model" in context_dict:
            self.model = context_dict["model"]
            
        if "provider_config" in context_dict and context_dict["provider_config"] is not None:
            self.provider_config = context_dict["provider_config"]

        if "stream_manager" in context_dict and context_dict["stream_manager"] is not None:
            self.stream_manager = context_dict["stream_manager"]

        if "tool_manager" in context_dict and context_dict["tool_manager"] is not None:
            self.tool_manager = context_dict["tool_manager"]

        if "tool_to_server_map" in context_dict:
            self.tool_to_server_map = context_dict["tool_to_server_map"]