#!/usr/bin/env python
"""
MCP-CLI specific patterns demonstration.
Shows tool-calling, streaming responses, cancellation, and more.

Run with: uv run examples/mcp_cli_patterns_demo.py
"""

import asyncio
import random

from chuk_term.ui import (
    output,
    format_table,
    display_code,
    display_diff,
    format_code_snippet,
)
from chuk_term.ui.terminal import clear_screen, reset_terminal
from rich.markdown import Markdown


async def demo_tool_calling():
    """Demonstrate tool-calling patterns with progress."""
    output.rule("Tool Calling Patterns")

    # User request
    output.print("[bold cyan]User[/bold cyan]")
    output.print("List all Python files in the project and analyze their complexity")
    output.print("")

    # Assistant planning
    output.print("[bold green]Assistant[/bold green]")
    output.print(
        "I'll help you list Python files and analyze their complexity. Let me break this down into steps:"
    )
    output.print("")

    # Show tool planning
    output.hint("Planning tool execution:")
    tools_to_execute = [
        {
            "name": "list_files",
            "args": {"pattern": "**/*.py"},
            "description": "List all Python files",
        },
        {
            "name": "read_file",
            "args": {"path": "src/main.py"},
            "description": "Read main.py",
        },
        {
            "name": "analyze_complexity",
            "args": {"files": ["main.py", "utils.py"]},
            "description": "Analyze code complexity",
        },
    ]

    for i, tool in enumerate(tools_to_execute, 1):
        output.print(f"  {i}. {tool['description']}")
        output.print(f"     [dim]Tool: {tool['name']}[/dim]")
        output.print(f"     [dim]Args: {tool['args']}[/dim]")

    output.print("")

    # Execute tools with progress
    output.hint("Executing tools:")
    for tool in tools_to_execute:
        # Show tool header
        output.print(f"\n[yellow]→ Running {tool['name']}[/yellow]")
        output.print(f"  [dim]Arguments: {tool['args']}[/dim]")

        # Simulate progress
        await asyncio.sleep(0.5)

        # Show tool output based on type
        if tool["name"] == "list_files":
            output.print("  [dim]Searching for Python files...[/dim]")
            await asyncio.sleep(0.5)
            files = [
                "src/main.py",
                "src/utils.py",
                "src/config.py",
                "tests/test_main.py",
                "tests/test_utils.py",
            ]
            output.success(f"  ✓ Found {len(files)} Python files")
            for file in files[:3]:  # Show first 3
                output.print(f"    • {file}")
            if len(files) > 3:
                output.print(f"    [dim]... and {len(files) - 3} more[/dim]")

        elif tool["name"] == "read_file":
            output.print("  [dim]Reading file content...[/dim]")
            await asyncio.sleep(0.5)
            output.success("  ✓ Read 250 lines from src/main.py")

        elif tool["name"] == "analyze_complexity":
            output.print("  [dim]Analyzing complexity metrics...[/dim]")
            await asyncio.sleep(0.8)
            output.success("  ✓ Analysis complete")
            # Show metrics
            metrics = {
                "Cyclomatic Complexity": "Low (avg: 3.2)",
                "Maintainability Index": "High (82/100)",
                "Lines of Code": "1,245",
                "Test Coverage": "87%",
            }
            for metric, value in metrics.items():
                output.print(f"    • {metric}: {value}")

    output.print("")
    output.success("All tools executed successfully!")


async def demo_streaming_response():
    """Demonstrate streaming response with markdown and code."""
    output.rule("Streaming Response with Markdown & Code")

    # User request
    output.print("[bold cyan]User[/bold cyan]")
    output.print(
        "Explain how to implement async file operations in Python with an example"
    )
    output.print("")

    # Assistant response
    output.print("[bold green]Assistant[/bold green]")
    output.print(
        "I'll explain async file operations in Python and provide a practical example."
    )
    output.print("")

    # Show markdown content
    markdown_content = """## Async File Operations in Python

Python's `asyncio` library combined with `aiofiles` enables non-blocking file I/O. 
This is particularly useful when dealing with multiple files or large files that could block your application.

### Key Benefits:
- **Non-blocking I/O**: Other coroutines can run while waiting for file operations
- **Better scalability**: Handle many files concurrently
- **Improved performance**: Especially for I/O-bound operations

### Example Implementation:"""

    output.print(Markdown(markdown_content))
    output.print("")

    # Show code using chuk-term's display_code
    code_example = """import asyncio
import aiofiles
from typing import List, Optional

class AsyncFileHandler:
    \"\"\"Handles async file operations.\"\"\"
    
    async def read_file(self, filepath: str) -> Optional[str]:
        \"\"\"Read file content asynchronously.\"\"\"
        try:
            async with aiofiles.open(filepath, 'r') as file:
                content = await file.read()
                return content
        except FileNotFoundError:
            print(f\"File not found: {filepath}\")
            return None
    
    async def write_file(self, filepath: str, content: str) -> bool:
        \"\"\"Write content to file asynchronously.\"\"\"
        try:
            async with aiofiles.open(filepath, 'w') as file:
                await file.write(content)
                return True
        except Exception as e:
            print(f\"Error writing file: {e}\")
            return False
    
    async def process_multiple_files(self, filepaths: List[str]):
        \"\"\"Process multiple files concurrently.\"\"\"
        tasks = [self.read_file(fp) for fp in filepaths]
        results = await asyncio.gather(*tasks)
        return results

# Usage example
async def main():
    handler = AsyncFileHandler()
    
    # Read multiple files concurrently
    files = ['file1.txt', 'file2.txt', 'file3.txt']
    contents = await handler.process_multiple_files(files)
    
    # Process results
    for filepath, content in zip(files, contents):
        if content:
            print(f"Processed {filepath}: {len(content)} chars")

if __name__ == '__main__':
    asyncio.run(main())"""

    # Display code with chuk-term's code display
    display_code(code_example, language="python", title="async_file_handler.py")

    output.print("")
    output.print(
        "This approach allows you to handle file operations efficiently without blocking your application's event loop."
    )


async def demo_code_modifications():
    """Demonstrate code additions and deletions."""
    output.rule("Code Modifications Display")

    # User request
    output.print("[bold cyan]User[/bold cyan]")
    output.print("Add error handling and logging to this function")
    output.print("")

    # Show original code
    output.print("[bold green]Assistant[/bold green]")
    output.print(
        "I'll add comprehensive error handling and logging. Here's the original function:"
    )
    output.print("")

    original_code = """def process_data(data):
    result = data * 2
    return result"""

    # Show original with format_code_snippet
    output.print("[dim]Original code:[/dim]")
    formatted_original = format_code_snippet(original_code, language="python")
    output.print(formatted_original)
    output.print("")

    # Show updated code with additions
    updated_code = """import logging

def process_data(data):
    \"\"\"Process data with error handling and logging.\"\"\"
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input
        if data is None:
            raise ValueError("Data cannot be None")
        
        # Log processing start
        logger.info(f"Processing data: {data}")
        
        # Process data
        result = data * 2
        
        # Log successful processing
        logger.info(f"Successfully processed data. Result: {result}")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
        raise RuntimeError(f"Failed to process data: {e}") from e"""

    output.print("[dim]Updated code with error handling and logging:[/dim]")
    display_code(updated_code, language="python", title="process_data.py")

    output.print("")
    output.success("✅ Added comprehensive error handling and logging")
    output.hint("Changes made:")
    output.print("  [green]+ Added logging import and logger setup[/green]")
    output.print("  [green]+ Added input validation[/green]")
    output.print("  [green]+ Added try-except blocks for error handling[/green]")
    output.print("  [green]+ Added informative logging at key points[/green]")
    output.print("  [green]+ Added proper exception chaining[/green]")


async def demo_adaptive_planning():
    """Demonstrate adaptive planning with dynamic todo updates."""
    output.rule("Adaptive Planning with Dynamic Updates")

    # User request
    output.print("[bold cyan]User[/bold cyan]")
    output.print("Set up a new microservice with database, API, and tests")
    output.print("")

    # Initial plan
    output.print("[bold green]Assistant[/bold green]")
    output.print("I'll help you set up a new microservice. Creating initial plan...")
    output.print("")

    # Initial todos
    todos = [
        {"task": "Create project structure", "status": "completed", "icon": "✅"},
        {"task": "Initialize git repository", "status": "completed", "icon": "✅"},
        {"task": "Set up database schema", "status": "in_progress", "icon": "🔄"},
        {"task": "Create API endpoints", "status": "pending", "icon": "⏳"},
        {"task": "Write unit tests", "status": "pending", "icon": "⏳"},
        {"task": "Set up CI/CD pipeline", "status": "pending", "icon": "⏳"},
    ]

    # Display current plan
    output.hint("📋 Current Plan Status:")
    for todo in todos:
        output.print(f"  {todo['icon']} {todo['task']}")

    output.print("")
    output.print("[yellow]🔄 Working on: Set up database schema[/yellow]")
    await asyncio.sleep(0.5)

    # Discover additional requirements
    output.print("")
    output.warning("⚠️  Discovered: Database requires migration system")
    output.print("[dim]Adding new tasks to plan...[/dim]")
    await asyncio.sleep(0.3)

    # Update plan dynamically
    new_tasks = [
        {"task": "Install migration tool (Alembic)", "status": "pending", "icon": "⏳"},
        {"task": "Create initial migration", "status": "pending", "icon": "⏳"},
        {"task": "Set up migration scripts", "status": "pending", "icon": "⏳"},
    ]

    # Insert new tasks after database schema
    todos[2]["status"] = "completed"
    todos[2]["icon"] = "✅"
    todos = todos[:3] + new_tasks + todos[3:]

    output.print("")
    output.hint("📋 Updated Plan (+3 new tasks):")
    for todo in todos:
        if todo["task"] in [t["task"] for t in new_tasks]:
            output.print(
                f"  {todo['icon']} [green]+[/green] {todo['task']} [dim](new)[/dim]"
            )
        else:
            output.print(f"  {todo['icon']} {todo['task']}")

    output.print("")

    # Continue execution with new tasks
    output.success("✅ Database schema created")
    output.print("[yellow]🔄 Working on: Install migration tool (Alembic)[/yellow]")
    await asyncio.sleep(0.5)
    output.success("✅ Migration tool installed")

    output.print("[yellow]🔄 Working on: Create initial migration[/yellow]")
    await asyncio.sleep(0.5)
    output.success("✅ Initial migration created")

    # Show final status
    output.print("")
    output.success("🎆 Adaptive planning complete!")
    output.print("")
    output.hint("Plan adapted successfully:")
    output.print("  • Original tasks: 6")
    output.print("  • Tasks added during execution: 3")
    output.print("  • Total tasks completed: 5")
    output.print("  • Remaining tasks: 4")


async def demo_tool_cancellation():
    """Demonstrate tool cancellation patterns."""
    output.rule("Tool Cancellation Handling")

    # User request
    output.print("[bold cyan]User[/bold cyan]")
    output.print("Analyze all files in the repository (this might take a while)")
    output.print("")

    # Start execution
    output.print("[bold green]Assistant[/bold green]")
    output.print("I'll analyze all files in the repository. This may take some time...")
    output.print("")

    # Show multiple tools running
    output.hint("Starting analysis:")

    tools = [
        "file_scanner",
        "dependency_analyzer",
        "code_quality_checker",
        "security_scanner",
        "documentation_generator",
    ]

    # Simulate starting tools
    for i, tool in enumerate(tools[:3]):
        output.print(f"\n[yellow]→ Running {tool}[/yellow]")
        output.print("  [dim]Processing...[/dim]")
        await asyncio.sleep(0.5)

        if i == 2:  # Simulate cancellation on third tool
            output.print("")
            output.warning("⚠️  Operation cancelled by user (Ctrl+C)")
            break

    # Show cancellation handling
    output.print("")
    output.hint("Cleaning up cancelled operations:")

    cleanup_tasks = [
        "Stopping file_scanner...",
        "Stopping dependency_analyzer...",
        "Stopping code_quality_checker...",
        "Releasing resources...",
        "Saving partial results...",
    ]

    for task in cleanup_tasks:
        output.print(f"  [dim]{task}[/dim]")
        await asyncio.sleep(0.2)

    output.success("✓ Cleanup complete")

    # Show what was completed
    output.print("")
    output.hint("Partial results saved:")
    output.print("  • Scanned 127 files (incomplete)")
    output.print("  • Found 15 dependencies (incomplete)")
    output.print("  • Analysis stopped at 45%")

    output.print("")
    output.print(
        "[dim]You can resume the analysis with 'continue' or start fresh with 'restart'[/dim]"
    )


async def demo_successful_tool_execution():
    """Demonstrate successful tool execution with various result types."""
    output.rule("Successful Tool Execution Patterns")

    # User request
    output.print("[bold cyan]User[/bold cyan]")
    output.print("Create a new feature branch, make changes, and create a pull request")
    output.print("")

    # Assistant response
    output.print("[bold green]Assistant[/bold green]")
    output.print(
        "I'll help you create a feature branch, make changes, and create a pull request."
    )
    output.print("")

    # Tool execution sequence
    tools_sequence = [
        {
            "name": "git_status",
            "description": "Checking repository status",
            "result_type": "info",
            "result": "On branch main, nothing to commit, working tree clean",
        },
        {
            "name": "git_branch",
            "description": "Creating feature branch",
            "result_type": "success",
            "result": "Created and switched to branch 'feature/add-async-support'",
        },
        {
            "name": "file_modify",
            "description": "Modifying src/handler.py",
            "result_type": "diff",
            "result": {"added": 15, "removed": 3, "modified": 5},
        },
        {
            "name": "run_tests",
            "description": "Running test suite",
            "result_type": "test",
            "result": {"passed": 42, "failed": 0, "skipped": 3, "time": "2.3s"},
        },
        {
            "name": "git_commit",
            "description": "Committing changes",
            "result_type": "success",
            "result": "Created commit: feat: add async file operations (3a4b5c6)",
        },
        {
            "name": "git_push",
            "description": "Pushing to remote",
            "result_type": "success",
            "result": "Pushed to origin/feature/add-async-support",
        },
        {
            "name": "create_pr",
            "description": "Creating pull request",
            "result_type": "url",
            "result": "https://github.com/user/repo/pull/123",
        },
    ]

    for tool in tools_sequence:
        output.print(f"\n[yellow]→ {tool['description']}[/yellow]")
        output.print(f"  [dim]Tool: {tool['name']}[/dim]")

        # Simulate execution
        await asyncio.sleep(0.4)

        # Show result based on type
        if tool["result_type"] == "info":
            output.print(f"  ℹ️  {tool['result']}")

        elif tool["result_type"] == "success":
            output.success(f"  ✓ {tool['result']}")

        elif tool["result_type"] == "diff":
            output.success("  ✓ File modified successfully")
            diff = tool["result"]
            output.print(
                f"    [green]+{diff['added']}[/green] [red]-{diff['removed']}[/red] [yellow]~{diff['modified']}[/yellow] lines changed"
            )

        elif tool["result_type"] == "test":
            test = tool["result"]
            output.success("  ✓ All tests passed!")
            output.print(
                f"    [green]✓ {test['passed']} passed[/green] | [dim]{test['skipped']} skipped[/dim] | Time: {test['time']}"
            )

        elif tool["result_type"] == "url":
            output.success("  ✓ Pull request created")
            output.print(f"    [blue][link]{tool['result']}[/link][/blue]")

    # Summary
    output.print("")
    output.success("🎉 All operations completed successfully!")
    output.print("")
    output.panel(
        "Pull request #123 has been created with your changes.\n"
        "The async file operations feature is ready for review.\n\n"
        "Next steps:\n"
        "• Wait for CI/CD checks to complete\n"
        "• Request code review from team members\n"
        "• Address any feedback",
        title="Summary",
        style="green",
    )


async def demo_code_diff():
    """Demonstrate code diff display."""
    output.rule("Code Diff Display")

    # User request
    output.print("[bold cyan]User[/bold cyan]")
    output.print("Show me what changed in the async handler implementation")
    output.print("")

    # Assistant response
    output.print("[bold green]Assistant[/bold green]")
    output.print("Here are the changes made to the async handler:")
    output.print("")

    # Define old and new code for diff
    old_code = """import asyncio
import json

class FileHandler:
    \"\"\"Handles file operations.\"\"\"
    
    def read_file(self, filepath):
        \"\"\"Read file content.\"\"\"
        with open(filepath, 'r') as file:
            return file.read()
    
    def write_file(self, filepath, content):
        \"\"\"Write content to file.\"\"\"
        with open(filepath, 'w') as file:
            file.write(content)"""

    new_code = """import asyncio
import aiofiles
from typing import List, Optional

class AsyncFileHandler:
    \"\"\"Handles async file operations.\"\"\"
    
    async def read_file(self, filepath: str) -> Optional[str]:
        \"\"\"Read file content asynchronously.\"\"\"
        try:
            async with aiofiles.open(filepath, 'r') as file:
                content = await file.read()
                return content
        except FileNotFoundError:
            print(f\"File not found: {filepath}\")
            return None
    
    async def write_file(self, filepath: str, content: str) -> bool:
        \"\"\"Write content to file asynchronously.\"\"\"
        try:
            async with aiofiles.open(filepath, 'w') as file:
                await file.write(content)
                return True
        except Exception as e:
            print(f\"Error writing file: {e}\")
            return False
    
    async def process_multiple_files(self, filepaths: List[str]):
        \"\"\"Process multiple files concurrently.\"\"\"
        tasks = [self.read_file(fp) for fp in filepaths]
        results = await asyncio.gather(*tasks)
        return results"""

    # Display the diff using chuk-term's display_diff
    display_diff(old_code, new_code, title="handler.py changes")

    output.print("")
    output.hint("Summary of changes:")
    output.print("  • Converted from synchronous to async operations")
    output.print("  • Added type hints for better code clarity")
    output.print("  • Implemented error handling")
    output.print("  • Added concurrent file processing capability")
    output.print("")
    output.success("✅ All changes improve performance and maintainability")


async def demo_error_recovery():
    """Demonstrate error handling and recovery in tool execution."""
    output.rule("Error Handling & Recovery")

    # User request
    output.print("[bold cyan]User[/bold cyan]")
    output.print("Deploy the application to production")
    output.print("")

    # Assistant response
    output.print("[bold green]Assistant[/bold green]")
    output.print(
        "I'll help you deploy the application to production. Let me run through the deployment checklist."
    )
    output.print("")

    # Deployment sequence with error
    output.hint("Running deployment checks:")

    checks = [
        {"name": "lint_check", "status": "pass", "message": "No linting errors found"},
        {"name": "type_check", "status": "pass", "message": "Type checking passed"},
        {
            "name": "test_suite",
            "status": "fail",
            "message": "2 tests failing",
            "details": [
                "FAILED: test_async_handler.py::test_concurrent_writes",
                "FAILED: test_async_handler.py::test_error_handling",
            ],
        },
        {
            "name": "security_scan",
            "status": "skip",
            "message": "Skipped due to test failures",
        },
        {"name": "build", "status": "skip", "message": "Skipped due to test failures"},
    ]

    for check in checks:
        await asyncio.sleep(0.3)

        if check["status"] == "pass":
            output.success(f"✓ {check['name']}: {check['message']}")

        elif check["status"] == "fail":
            output.error(f"✗ {check['name']}: {check['message']}")
            if "details" in check:
                for detail in check["details"]:
                    output.print(f"    [red]{detail}[/red]")
            output.print("")
            output.warning("⚠️  Deployment blocked due to failing tests")
            break

        elif check["status"] == "skip":
            output.print(f"[dim]⊘ {check['name']}: {check['message']}[/dim]")

    # Recovery suggestions
    output.print("")
    output.hint("💡 Suggested actions:")
    output.print("  1. Fix the failing tests:")
    output.print("     [dim]`mcp-cli run pytest test_async_handler.py -v`[/dim]")
    output.print("  2. Re-run the deployment checks:")
    output.print("     [dim]`mcp-cli deploy --check`[/dim]")
    output.print("  3. Or deploy with --force flag (not recommended):")
    output.print("     [dim]`mcp-cli deploy --force --skip-tests`[/dim]")

    output.print("")
    output.print("[dim]Would you like me to help fix the failing tests?[/dim]")


async def demo_planner_and_execution():
    """Demonstrate planning and plan execution with todo lists."""
    output.rule("Planner & Plan Execution (Todo Lists)")

    # User request
    output.print("[bold cyan]User[/bold cyan]")
    output.print(
        "Refactor the authentication system to use JWT tokens instead of sessions"
    )
    output.print("")

    # Assistant planning phase
    output.print("[bold green]Assistant[/bold green]")
    output.print(
        "I'll help you refactor the authentication system to use JWT tokens. Let me create a comprehensive plan for this task."
    )
    output.print("")

    # Define the plan as todo items
    todos = [
        {
            "id": 1,
            "task": "Analyze current session-based auth implementation",
            "status": "pending",
            "deps": [],
        },
        {
            "id": 2,
            "task": "Install JWT dependencies (pyjwt, python-jose)",
            "status": "pending",
            "deps": [],
        },
        {
            "id": 3,
            "task": "Create JWT token generation utilities",
            "status": "pending",
            "deps": [2],
        },
        {
            "id": 4,
            "task": "Create JWT token validation middleware",
            "status": "pending",
            "deps": [3],
        },
        {
            "id": 5,
            "task": "Update user login endpoint to issue JWT tokens",
            "status": "pending",
            "deps": [3],
        },
        {
            "id": 6,
            "task": "Update protected endpoints to use JWT validation",
            "status": "pending",
            "deps": [4],
        },
        {
            "id": 7,
            "task": "Implement token refresh mechanism",
            "status": "pending",
            "deps": [3, 4],
        },
        {
            "id": 8,
            "task": "Update frontend to store and send JWT tokens",
            "status": "pending",
            "deps": [5],
        },
        {
            "id": 9,
            "task": "Write tests for JWT authentication",
            "status": "pending",
            "deps": [3, 4, 5, 6],
        },
        {
            "id": 10,
            "task": "Update API documentation",
            "status": "pending",
            "deps": [5, 6, 7],
        },
        {
            "id": 11,
            "task": "Run security audit on new implementation",
            "status": "pending",
            "deps": [9],
        },
        {
            "id": 12,
            "task": "Deploy to staging and test",
            "status": "pending",
            "deps": [11],
        },
    ]

    # Display initial plan
    output.hint("📋 Execution Plan (12 tasks):")
    output.print("")

    # Show todo list as a table
    table_data = []
    for todo in todos:
        status_icon = (
            "⏳"
            if todo["status"] == "pending"
            else "🔄"
            if todo["status"] == "in_progress"
            else "✅"
        )
        deps_str = (
            f"Depends on: {', '.join(f'#{d}' for d in todo['deps'])}"
            if todo["deps"]
            else "No dependencies"
        )
        table_data.append(
            {
                "#": str(todo["id"]),
                "Status": status_icon,
                "Task": todo["task"][:50] + "..."
                if len(todo["task"]) > 50
                else todo["task"],
                "Dependencies": deps_str,
            }
        )

    table = format_table(
        table_data,
        title="JWT Authentication Refactoring Plan",
        columns=["#", "Status", "Task", "Dependencies"],
    )
    output.print_table(table)

    output.print("")
    output.hint("Starting plan execution...")
    output.print("")

    # Execute plan with real-time updates
    completed = []
    in_progress = None

    while len(completed) < len(todos):
        # Find next task that can be executed (dependencies met)
        available_tasks = [
            t
            for t in todos
            if t["id"] not in completed
            and t["id"] != in_progress
            and all(dep in completed for dep in t["deps"])
        ]

        if available_tasks and in_progress is None:
            # Start next task
            task = available_tasks[0]
            in_progress = task["id"]
            task["status"] = "in_progress"

            output.print(f"[yellow]🔄 Task #{task['id']}: {task['task']}[/yellow]")
            output.print("   [dim]Starting execution...[/dim]")

            # Simulate work
            await asyncio.sleep(0.5)

            # Simulate different outcomes
            if task["id"] == 9:  # Tests might find issues
                output.warning("   ⚠️  Found 2 failing tests, fixing...")
                await asyncio.sleep(0.3)
                output.success("   ✓ Tests fixed and passing")
            elif task["id"] == 11:  # Security audit
                output.print("   [dim]Running security scan...[/dim]")
                await asyncio.sleep(0.3)
                output.success("   ✓ No vulnerabilities found")
            else:
                # Normal completion
                output.success("   ✓ Completed successfully")

            # Mark as completed
            task["status"] = "completed"
            completed.append(task["id"])
            in_progress = None

            # Show progress
            progress = len(completed)
            total = len(todos)
            progress_bar = "█" * (progress * 20 // total) + "░" * (
                (total - progress) * 20 // total
            )
            output.print("")
            output.print(
                f"[dim]Progress: [{progress_bar}] {progress}/{total} tasks completed[/dim]"
            )
            output.print("")

            await asyncio.sleep(0.2)

    # Final summary
    output.print("")
    output.success("🎉 Plan execution completed successfully!")
    output.print("")

    # Show completion summary
    output.panel(
        "JWT Authentication Refactoring Complete\n\n"
        "✅ All 12 tasks completed successfully\n"
        "📊 Results:\n"
        "  • Session-based auth replaced with JWT\n"
        "  • Token refresh mechanism implemented\n"
        "  • All tests passing (15 new tests added)\n"
        "  • Security audit passed\n"
        "  • Deployed to staging environment\n\n"
        "Next steps:\n"
        "  • Monitor staging for 24 hours\n"
        "  • Schedule production deployment\n"
        "  • Update client SDKs",
        title="Refactoring Summary",
        style="green",
    )


async def demo_parallel_tool_execution():
    """Demonstrate parallel tool execution with progress tracking."""
    output.rule("Parallel Tool Execution")

    # User request
    output.print("[bold cyan]User[/bold cyan]")
    output.print("Check the health of all our microservices")
    output.print("")

    # Assistant response
    output.print("[bold green]Assistant[/bold green]")
    output.print("I'll check the health status of all microservices in parallel.")
    output.print("")

    # Services to check
    services = [
        {"name": "auth-service", "port": 3001, "status": None, "response_time": None},
        {"name": "user-service", "port": 3002, "status": None, "response_time": None},
        {
            "name": "payment-service",
            "port": 3003,
            "status": None,
            "response_time": None,
        },
        {
            "name": "notification-service",
            "port": 3004,
            "status": None,
            "response_time": None,
        },
        {
            "name": "analytics-service",
            "port": 3005,
            "status": None,
            "response_time": None,
        },
    ]

    output.hint("Checking services (parallel execution):")
    output.print("")

    # Show all services as pending
    for service in services:
        output.print(f"  ⏳ {service['name']:<20} [dim]Checking...[/dim]")

    # Simulate parallel execution with random completion
    output.print("")
    completed = []

    while len(completed) < len(services):
        await asyncio.sleep(0.3)

        # Randomly complete a service
        remaining = [s for s in services if s["name"] not in completed]
        if remaining:
            service = random.choice(remaining)
            service["response_time"] = random.randint(50, 500)
            service["status"] = "healthy" if random.random() > 0.2 else "degraded"
            completed.append(service["name"])

            # Update display
            if service["status"] == "healthy":
                output.success(
                    f"  ✓ {service['name']:<20} [green]Healthy[/green] ({service['response_time']}ms)"
                )
            else:
                output.warning(
                    f"  ⚠ {service['name']:<20} [yellow]Degraded[/yellow] ({service['response_time']}ms)"
                )

    # Summary table
    output.print("")
    table_data = []
    for service in services:
        status_emoji = "🟢" if service["status"] == "healthy" else "🟡"
        table_data.append(
            {
                "Service": service["name"],
                "Status": f"{status_emoji} {service['status'].capitalize()}",
                "Port": str(service["port"]),
                "Response Time": f"{service['response_time']}ms",
            }
        )

    table = format_table(
        table_data,
        title="Service Health Summary",
        columns=["Service", "Status", "Port", "Response Time"],
    )
    output.print_table(table)

    # Overall status
    healthy_count = sum(1 for s in services if s["status"] == "healthy")
    if healthy_count == len(services):
        output.success(f"\n✅ All {len(services)} services are healthy!")
    else:
        output.warning(
            f"\n⚠️  {healthy_count}/{len(services)} services healthy, {len(services) - healthy_count} degraded"
        )


async def main():
    """Run all MCP-CLI pattern demos."""
    # Setup terminal
    clear_screen()

    # Show banner
    output.panel(
        """
MCP-CLI Patterns Demonstration

Showcasing real-world patterns for:
• Tool calling with progress tracking
• Streaming responses with markdown & code
• Cancellation handling
• Error recovery
• Parallel execution
    """.strip(),
        title="MCP-CLI Demo",
        style="cyan",
    )

    await asyncio.sleep(1)

    # Run demos
    demos = [
        ("Planner & Execution", demo_planner_and_execution),
        ("Adaptive Planning", demo_adaptive_planning),
        ("Tool Calling", demo_tool_calling),
        ("Streaming Response", demo_streaming_response),
        ("Code Modifications", demo_code_modifications),
        ("Code Diff", demo_code_diff),
        ("Tool Cancellation", demo_tool_cancellation),
        ("Successful Execution", demo_successful_tool_execution),
        ("Error Recovery", demo_error_recovery),
        ("Parallel Execution", demo_parallel_tool_execution),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        if i > 1:
            output.print("")
            output.print("")

        await demo_func()

        if i < len(demos):
            # Add spacing between demos
            output.print("")
            output.print("[dim]─" * 40 + "[/dim]")
            await asyncio.sleep(0.5)

    # Summary
    output.print("")
    output.print("")
    output.rule("Demo Complete")
    output.success(
        """
✅ All MCP-CLI patterns demonstrated!

Key patterns shown:
• Tool execution with real-time progress
• Streaming markdown and code responses  
• Graceful cancellation handling
• Error recovery with suggestions
• Parallel tool execution
• Rich output formatting with tables and panels
    """.strip()
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        output.warning("\n\nDemo interrupted by user")
    except Exception as e:
        output.error(f"Error: {e}")
    finally:
        reset_terminal()
