"""
Selective tool binding extension for ToolManager.

This module extends ToolManager with selective binding capabilities
to optimize performance and reduce resource usage.
"""

from typing import Dict, Any, Set
import logging

logger = logging.getLogger(__name__)


class SelectiveBindingMixin:
    """
    Mixin to add selective tool binding capabilities to ToolManager.

    This allows binding only the tools needed for a specific plan,
    dramatically reducing memory usage and improving performance.
    """

    def __init__(self):
        self._all_tools_cache: Dict[str, Any] = {}
        self._bound_tools: Set[str] = set()
        self._selective_mode: bool = False

    async def enable_selective_mode(self) -> None:
        """Enable selective tool binding mode."""
        self._selective_mode = True
        # Cache all available tools
        if not self._all_tools_cache:
            tools = await self.get_all_tools()
            self._all_tools_cache = {tool.name: tool for tool in tools}
        logger.info(
            f"Selective mode enabled with {len(self._all_tools_cache)} tools available"
        )

    async def disable_selective_mode(self) -> None:
        """Disable selective mode and bind all tools."""
        self._selective_mode = False
        self._bound_tools.clear()
        logger.info("Selective mode disabled - all tools available")

    async def bind_tools(self, tool_names: Set[str]) -> Dict[str, Any]:
        """
        Bind only the specified tools.

        Args:
            tool_names: Set of tool names to bind

        Returns:
            Dictionary of bound tools
        """
        if not self._selective_mode:
            logger.warning("Selective binding called but selective mode is not enabled")
            return {}

        bound = {}
        for name in tool_names:
            if name in self._all_tools_cache:
                bound[name] = self._all_tools_cache[name]
                self._bound_tools.add(name)
            else:
                logger.warning(f"Tool '{name}' not found in available tools")

        logger.info(f"Bound {len(bound)} tools: {list(bound.keys())}")
        return bound

    async def unbind_tools(self, tool_names: Set[str]) -> None:
        """
        Unbind specified tools to free resources.

        Args:
            tool_names: Set of tool names to unbind
        """
        for name in tool_names:
            self._bound_tools.discard(name)
        logger.info(f"Unbound {len(tool_names)} tools")

    async def get_bound_tools(self) -> Set[str]:
        """Get the set of currently bound tools."""
        return self._bound_tools.copy()

    def is_tool_bound(self, tool_name: str) -> bool:
        """Check if a specific tool is currently bound."""
        if not self._selective_mode:
            return True  # All tools available when not in selective mode
        return tool_name in self._bound_tools

    async def bind_additional_tools(self, tool_names: Set[str]) -> Dict[str, Any]:
        """
        Bind additional tools to the existing set.

        Args:
            tool_names: Set of additional tool names to bind

        Returns:
            Dictionary of newly bound tools
        """
        new_tools = tool_names - self._bound_tools
        if new_tools:
            return await self.bind_tools(new_tools)
        return {}

    def get_binding_stats(self) -> Dict[str, int]:
        """
        Get statistics about tool binding.

        Returns:
            Dictionary with binding statistics
        """
        total = len(self._all_tools_cache)
        bound = len(self._bound_tools)
        return {
            "total_available": total,
            "tools_bound": bound,
            "tools_saved": total - bound,
            "optimization_percent": int((1 - bound / total) * 100) if total > 0 else 0,
        }
