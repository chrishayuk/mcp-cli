"""
Selective tool binder for optimized tool loading.

This module provides intelligent tool binding to:
- Analyze plans to determine required tools
- Bind only necessary tools for execution
- Track tool usage statistics
- Optimize memory and performance
"""

from typing import Dict, Any, Set
from dataclasses import dataclass
import logging

from mcp_cli.tools.tool_manager import ToolManager
from .task_planner import Task, TaskPlanner

logger = logging.getLogger(__name__)


@dataclass
class ToolStats:
    """Statistics about tool optimization."""

    total_available: int = 0
    tools_bound: int = 0
    tools_saved: int = 0
    memory_saved_mb: int = 0
    tokens_saved: int = 0
    optimization_percent: int = 0


class SelectiveToolBinder:
    """
    Intelligently binds only the tools needed for a specific plan.

    This reduces:
    - Memory usage (fewer tools loaded)
    - API tokens (smaller context)
    - Execution time (less overhead)
    """

    # Estimated costs per tool (memory in MB, tokens for description)
    TOOL_COSTS = {
        "default": {"memory_mb": 15, "tokens": 100},
        "complex": {"memory_mb": 30, "tokens": 200},
        "simple": {"memory_mb": 5, "tokens": 50},
    }

    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager
        self.all_tools: Dict[str, Any] = {}
        self.bound_tools: Set[str] = set()
        self.stats = ToolStats()

    async def initialize(self) -> None:
        """Load all available tools from MCP servers."""
        logger.info("Loading available tools from MCP servers...")
        try:
            self.all_tools = await self.tool_manager.get_all_tools()
            self.stats.total_available = len(self.all_tools)
            logger.info(f"Found {self.stats.total_available} available tools")
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")
            self.all_tools = {}

    def analyze_task_requirements(self, task: Task) -> Set[str]:
        """
        Analyze a task to determine which tools are required.

        Returns a set of tool names needed for the task.
        """
        required = set()

        # If task explicitly specifies a tool
        if task.tool:
            required.add(task.tool)
            return required

        # If it's a reasoning task, no tools needed
        if task.is_reasoning:
            return required

        # Otherwise, infer from description
        description = task.description.lower()

        # File operations
        if any(word in description for word in ["read", "load", "open"]):
            required.add("read_file")
        if any(word in description for word in ["write", "save", "create"]):
            if "file" in description or "document" in description:
                required.add("write_file")
        if any(word in description for word in ["list", "directory", "folder"]):
            required.add("list_files")
        if "delete" in description and "file" in description:
            required.add("delete_file")

        # Git operations
        if "git" in description or "commit" in description:
            required.update(["git_status", "git_add", "git_commit"])
        if "push" in description:
            required.add("git_push")
        if "branch" in description:
            required.add("git_branch")

        # Database operations
        if any(word in description for word in ["database", "query", "sql", "table"]):
            required.update(["execute_query", "list_tables"])

        # Testing
        if "test" in description and any(
            word in description for word in ["run", "execute"]
        ):
            required.add("run_tests")
        if "coverage" in description:
            required.add("coverage_report")

        # API/HTTP
        if any(word in description for word in ["api", "http", "request", "endpoint"]):
            required.add("http_request")

        # Docker/deployment
        if "docker" in description:
            if "build" in description:
                required.add("docker_build")
            if "push" in description:
                required.add("docker_push")
        if "deploy" in description:
            required.add("deploy_service")

        return required

    def analyze_plan_requirements(self, planner: TaskPlanner) -> Set[str]:
        """
        Analyze an entire plan to determine all required tools.

        Returns a set of all tool names needed for the plan.
        """
        required_tools = set()

        for task in planner.tasks:
            # Get explicitly defined tools
            if task.tool:
                required_tools.add(task.tool)
            else:
                # Infer tools from task description
                inferred = self.analyze_task_requirements(task)
                required_tools.update(inferred)

        return required_tools

    async def bind_tools_for_plan(self, planner: TaskPlanner) -> Dict[str, Any]:
        """
        Bind only the tools needed for a specific plan.

        Returns a dictionary of bound tools.
        """
        # Initialize if needed
        if not self.all_tools:
            await self.initialize()

        # Determine required tools
        required_tools = self.analyze_plan_requirements(planner)

        # Filter to only available tools
        available_required = required_tools.intersection(set(self.all_tools.keys()))

        # Bind the tools
        bound_tools = {}
        for tool_name in available_required:
            bound_tools[tool_name] = self.all_tools[tool_name]

        self.bound_tools = set(bound_tools.keys())

        # Calculate statistics
        self._calculate_stats()

        # Log optimization
        logger.info(f"Bound {len(bound_tools)}/{len(self.all_tools)} tools")
        logger.info(f"Optimization: {self.stats.optimization_percent}% reduction")

        return bound_tools

    def _calculate_stats(self) -> None:
        """Calculate optimization statistics."""
        self.stats.tools_bound = len(self.bound_tools)
        self.stats.tools_saved = self.stats.total_available - self.stats.tools_bound

        if self.stats.total_available > 0:
            self.stats.optimization_percent = int(
                (self.stats.tools_saved / self.stats.total_available) * 100
            )

        # Estimate memory and token savings
        cost_per_tool = self.TOOL_COSTS["default"]
        self.stats.memory_saved_mb = self.stats.tools_saved * cost_per_tool["memory_mb"]
        self.stats.tokens_saved = self.stats.tools_saved * cost_per_tool["tokens"]

    def get_unbound_tools(self) -> Set[str]:
        """Get the set of tools that were NOT bound."""
        return set(self.all_tools.keys()) - self.bound_tools

    def get_stats(self) -> ToolStats:
        """Get optimization statistics."""
        return self.stats

    def should_rebind(self, new_tools: Set[str]) -> bool:
        """
        Check if we need to rebind tools for new requirements.

        Returns True if any required tools are not currently bound.
        """
        return bool(new_tools - self.bound_tools)

    async def bind_additional_tools(self, tool_names: Set[str]) -> Dict[str, Any]:
        """
        Bind additional tools to the existing set.

        Used when adapting plans and new tools are needed.
        """
        new_tools = {}

        for tool_name in tool_names:
            if tool_name in self.all_tools and tool_name not in self.bound_tools:
                new_tools[tool_name] = self.all_tools[tool_name]
                self.bound_tools.add(tool_name)

        # Recalculate stats
        self._calculate_stats()

        logger.info(f"Bound {len(new_tools)} additional tools")

        return new_tools

    def get_tool_categories(self) -> Dict[str, Dict[str, Any]]:
        """
        Get tools organized by category.

        Returns a dictionary with category statistics.
        """
        categories = {}

        # Categorize tools based on name patterns
        for tool_name in self.all_tools.keys():
            category = self._determine_category(tool_name)

            if category not in categories:
                categories[category] = {"total": 0, "bound": 0, "tools": []}

            categories[category]["total"] += 1
            categories[category]["tools"].append(tool_name)

            if tool_name in self.bound_tools:
                categories[category]["bound"] += 1

        return categories

    def _determine_category(self, tool_name: str) -> str:
        """Determine the category of a tool based on its name."""
        name_lower = tool_name.lower()

        if any(word in name_lower for word in ["read", "write", "file", "directory"]):
            return "file"
        elif any(word in name_lower for word in ["git", "commit", "branch", "push"]):
            return "git"
        elif any(word in name_lower for word in ["test", "coverage", "pytest"]):
            return "testing"
        elif any(word in name_lower for word in ["query", "database", "sql", "table"]):
            return "database"
        elif any(word in name_lower for word in ["http", "api", "request", "endpoint"]):
            return "api"
        elif any(word in name_lower for word in ["docker", "deploy", "kubernetes"]):
            return "deployment"
        else:
            return "other"

    def reset(self) -> None:
        """Reset the tool binder state."""
        self.bound_tools.clear()
        self.stats = ToolStats(total_available=len(self.all_tools))
