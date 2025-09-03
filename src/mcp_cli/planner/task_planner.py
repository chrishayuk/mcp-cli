"""
Task planner for intelligent execution planning.

This module provides planning capabilities to:
- Create execution plans from user requests
- Identify task dependencies
- Determine which tools are needed
- Enable parallel execution where possible
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum


class TaskStatus(Enum):
    """Status of a task in the execution plan."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Represents a single task in an execution plan."""

    id: str
    description: str
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    tool: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    is_reasoning: bool = False  # True if this is a pure reasoning task
    can_parallel: bool = True  # Whether this can run in parallel with others


class TaskPlanner:
    """
    Intelligent task planner that creates optimized execution plans.

    This planner:
    - Analyzes user requests to create task breakdowns
    - Identifies which tasks need tools vs pure reasoning
    - Determines task dependencies
    - Enables parallel execution where possible
    """

    def __init__(self):
        self.tasks: List[Task] = []
        self.completed_tasks: Set[str] = set()

    def add_task(self, task: Task) -> None:
        """Add a task to the plan."""
        self.tasks.append(task)

    def clear_tasks(self) -> None:
        """Clear all tasks from the planner."""
        self.tasks.clear()
        self.completed_tasks.clear()

    def analyze_request(self, user_request: str) -> List[Task]:
        """
        Analyze a user request and create an execution plan.

        This is a simplified version - in production, this could use
        an LLM to intelligently break down the request.
        """
        request_lower = user_request.lower()
        tasks = []
        task_id = 1

        # Detect common patterns and create appropriate tasks

        # File operations
        if any(word in request_lower for word in ["create", "write", "file"]):
            if "test" in request_lower:
                tasks.append(
                    Task(
                        id=str(task_id),
                        description="Create test file",
                        tool="write_file",
                        priority=2,
                    )
                )
                task_id += 1

            if "readme" in request_lower or "documentation" in request_lower:
                tasks.append(
                    Task(
                        id=str(task_id),
                        description="Create documentation",
                        tool="write_file",
                        priority=3,
                    )
                )
                task_id += 1

        # Git operations
        if "commit" in request_lower or "git" in request_lower:
            # Need to check status first
            tasks.append(
                Task(
                    id=str(task_id),
                    description="Check git status",
                    tool="git_status",
                    priority=4,
                )
            )
            status_id = str(task_id)
            task_id += 1

            # Then stage files
            tasks.append(
                Task(
                    id=str(task_id),
                    description="Stage files",
                    tool="git_add",
                    priority=5,
                    dependencies=[status_id],
                )
            )
            stage_id = str(task_id)
            task_id += 1

            # Finally commit
            tasks.append(
                Task(
                    id=str(task_id),
                    description="Commit changes",
                    tool="git_commit",
                    priority=6,
                    dependencies=[stage_id],
                )
            )
            task_id += 1

        # Architecture/design tasks (no tools needed)
        if any(
            word in request_lower for word in ["design", "architect", "plan", "explain"]
        ):
            if not any(
                word in request_lower for word in ["create", "write", "implement"]
            ):
                tasks.append(
                    Task(
                        id=str(task_id),
                        description="Analyze requirements",
                        is_reasoning=True,
                        priority=1,
                    )
                )
                task_id += 1

                tasks.append(
                    Task(
                        id=str(task_id),
                        description="Design solution",
                        is_reasoning=True,
                        priority=2,
                        dependencies=[str(task_id - 1)],
                    )
                )
                task_id += 1

        # Testing
        if "test" in request_lower and "run" in request_lower:
            tasks.append(
                Task(
                    id=str(task_id),
                    description="Run tests",
                    tool="run_tests",
                    priority=3,
                )
            )
            task_id += 1

        # If no specific tasks identified, create a generic reasoning task
        if not tasks:
            tasks.append(
                Task(
                    id="1", description="Process request", is_reasoning=True, priority=1
                )
            )

        return tasks

    def create_plan(self) -> List[Task]:
        """
        Create an optimized execution plan respecting dependencies.

        Returns tasks in execution order.
        """
        if not self.tasks:
            return []

        # Topological sort with priority
        sorted_tasks = []
        completed = set()

        while len(sorted_tasks) < len(self.tasks):
            # Find tasks that can be executed (dependencies met)
            available = []
            for task in self.tasks:
                if task.id not in completed:
                    deps_met = all(dep in completed for dep in task.dependencies)
                    if deps_met:
                        available.append(task)

            if not available:
                # Circular dependency or error
                break

            # Sort by priority and select next task(s)
            available.sort(key=lambda t: t.priority)

            # Add all tasks with the same priority (can be parallel)
            current_priority = available[0].priority
            for task in available:
                if task.priority == current_priority:
                    sorted_tasks.append(task)
                    completed.add(task.id)
                else:
                    break

        return sorted_tasks

    def get_required_tools(self) -> Set[str]:
        """Get the set of tools required for the current plan."""
        tools = set()
        for task in self.tasks:
            if task.tool and not task.is_reasoning:
                tools.add(task.tool)
        return tools

    def get_parallel_groups(self) -> List[List[Task]]:
        """
        Group tasks that can be executed in parallel.

        Returns a list of task groups, where each group can run in parallel.
        """
        plan = self.create_plan()
        if not plan:
            return []

        groups = []
        completed = set()

        while len(completed) < len(plan):
            # Find all tasks that can run now
            group = []
            for task in plan:
                if task.id not in completed:
                    deps_met = all(dep in completed for dep in task.dependencies)
                    if deps_met and task.can_parallel:
                        group.append(task)

            if group:
                groups.append(group)
                for task in group:
                    completed.add(task.id)
            else:
                # Handle tasks that can't be parallelized
                for task in plan:
                    if task.id not in completed:
                        deps_met = all(dep in completed for dep in task.dependencies)
                        if deps_met:
                            groups.append([task])
                            completed.add(task.id)
                            break

        return groups

    def mark_completed(self, task_id: str, result: Any = None) -> None:
        """Mark a task as completed."""
        for task in self.tasks:
            if task.id == task_id:
                task.status = TaskStatus.COMPLETED
                task.result = result
                self.completed_tasks.add(task_id)
                break

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed."""
        for task in self.tasks:
            if task.id == task_id:
                task.status = TaskStatus.FAILED
                task.error = error
                break

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def should_adapt_plan(self) -> bool:
        """
        Determine if the plan should be adapted based on failures.

        Returns True if any critical task has failed.
        """
        for task in self.tasks:
            if task.status == TaskStatus.FAILED:
                # Check if any other tasks depend on this
                for other in self.tasks:
                    if task.id in other.dependencies:
                        return True
        return False

    def adapt_plan(self, failed_task_id: str) -> List[Task]:
        """
        Adapt the plan when a task fails.

        Creates recovery tasks and adjusts dependencies.
        """
        failed_task = self.get_task_by_id(failed_task_id)
        if not failed_task:
            return []

        recovery_tasks = []

        # Create recovery tasks based on the type of failure
        if failed_task.tool == "run_tests":
            # Tests failed - need to fix them
            recovery_tasks.extend(
                [
                    Task(
                        id=f"{failed_task_id}_fix",
                        description="Analyze test failures",
                        is_reasoning=True,
                        priority=failed_task.priority,
                    ),
                    Task(
                        id=f"{failed_task_id}_repair",
                        description="Fix failing tests",
                        tool="write_file",
                        priority=failed_task.priority + 1,
                        dependencies=[f"{failed_task_id}_fix"],
                    ),
                    Task(
                        id=f"{failed_task_id}_retry",
                        description="Re-run tests",
                        tool="run_tests",
                        priority=failed_task.priority + 2,
                        dependencies=[f"{failed_task_id}_repair"],
                    ),
                ]
            )

            # Update dependencies of tasks that depended on the failed task
            for task in self.tasks:
                if failed_task_id in task.dependencies:
                    task.dependencies.remove(failed_task_id)
                    task.dependencies.append(f"{failed_task_id}_retry")

        elif failed_task.tool in ["git_commit", "git_push"]:
            # Git operation failed
            recovery_tasks.append(
                Task(
                    id=f"{failed_task_id}_fix",
                    description="Resolve git issues",
                    is_reasoning=True,
                    priority=failed_task.priority,
                )
            )

        # Add recovery tasks to the plan
        for task in recovery_tasks:
            self.add_task(task)

        return recovery_tasks
