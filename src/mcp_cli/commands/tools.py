# src/mcp_cli/commands/tools.py
"""
Show **all tools** exposed by every connected MCP server, either as a
pretty Rich table or raw JSON.

ENHANCED: Now includes validation status and filtering information.

How to use
----------
* Chat / interactive : `/tools`, `/tools --all`, `/tools --raw`, `/tools --validation`
* CLI script         : `mcp-cli tools [--all|--raw|--validation]`

Both the chat & CLI layers call :pyfunc:`tools_action_async`; the
blocking helper :pyfunc:`tools_action` exists only for legacy sync code.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel

# MCP-CLI helpers
from mcp_cli.tools.formatting import create_tools_table
from mcp_cli.tools.manager import ToolManager
from mcp_cli.utils.async_utils import run_blocking
from mcp_cli.utils.rich_helpers import get_console

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# async (canonical) implementation
# ────────────────────────────────────────────────────────────────────────────────
async def tools_action_async(                    # noqa: D401
    tm: ToolManager,
    *,
    show_details: bool = False,
    show_raw: bool = False,
    show_validation: bool = False,
    provider: str = "openai",
) -> List[Dict[str, Any]]:
    """
    Fetch the **deduplicated** tool list from *all* servers and print it.

    Parameters
    ----------
    tm
        A fully-initialised :class:`~mcp_cli.tools.manager.ToolManager`.
    show_details
        When *True*, include parameter schemas in the table.
    show_raw
        When *True*, dump raw JSON definitions instead of a table.
    show_validation
        When *True*, show validation status and errors.
    provider
        Provider to validate tools for (default: openai).

    Returns
    -------
    list
        The list of tool-metadata dictionaries (always JSON-serialisable).
    """
    console = get_console()
    console.print("[cyan]\nFetching tool catalogue from all servers…[/cyan]")

    if show_validation:
        # Show validation-specific information
        return await _show_validation_info(tm, provider, console)
    
    # Get tools based on whether validation is available
    if hasattr(tm, 'get_adapted_tools_for_llm'):
        # Use validated tools
        try:
            valid_tools_defs, _ = await tm.get_adapted_tools_for_llm(provider)
            
            # Convert back to ToolInfo-like structure for display
            all_tools = []
            for tool_def in valid_tools_defs:
                func = tool_def.get("function", {})
                tool_name = func.get("name", "unknown")
                
                # Try to extract namespace from name
                if "_" in tool_name:
                    parts = tool_name.split("_", 1)
                    namespace = parts[0]
                    name = parts[1]
                else:
                    namespace = "unknown"
                    name = tool_name
                
                # Create a ToolInfo-like object
                tool_info = type('ToolInfo', (), {
                    'name': name,
                    'namespace': namespace,
                    'description': func.get("description", ""),
                    'parameters': func.get("parameters", {}),
                    'is_async': False,
                    'tags': [],
                    'supports_streaming': False
                })()
                
                all_tools.append(tool_info)
            
            # Show validation summary if available
            if hasattr(tm, 'get_validation_summary'):
                summary = tm.get_validation_summary()
                if summary.get('invalid_tools', 0) > 0:
                    console.print(f"[yellow]Note: {summary['invalid_tools']} tools filtered out due to validation errors[/yellow]")
                    console.print(f"[dim]Use --validation flag to see details[/dim]")
            
        except Exception as e:
            logger.warning(f"Error getting validated tools, falling back to all tools: {e}")
            all_tools = await tm.get_unique_tools()
    else:
        # Fallback to original method
        all_tools = await tm.get_unique_tools()

    if not all_tools:
        console.print("[yellow]No tools available from any server.[/yellow]")
        logger.debug("ToolManager returned an empty tools list")
        return []

    # ── raw JSON mode ───────────────────────────────────────────────────
    if show_raw:
        payload = [
            {
                "name":        t.name,
                "namespace":   t.namespace,
                "description": t.description,
                "parameters":  t.parameters,
                "is_async":    getattr(t, "is_async", False),
                "tags":        getattr(t, "tags", []),
                "aliases":     getattr(t, "aliases", []),
            }
            for t in all_tools
        ]
        # Use `console.print_json` for better formatting without line numbers. 
        # Cleaner output for devs working with MCP JSON schemas.
        console.print_json(json.dumps(payload, indent=2, ensure_ascii=False))
        
        return payload

    # ── Rich table mode ─────────────────────────────────────────────────
    table: Table = create_tools_table(all_tools, show_details=show_details)
    console.print(table)
    console.print(f"[green]Total tools available: {len(all_tools)}[/green]")

    # Show validation info if enhanced manager
    if hasattr(tm, 'get_validation_summary'):
        summary = tm.get_validation_summary()
        if summary.get('total_tools', 0) > len(all_tools):
            console.print(f"[dim]({summary['total_tools'] - len(all_tools)} tools hidden due to validation/filtering)[/dim]")

    # Return a safe JSON structure (no .to_dict() needed)
    return [
        {
            "name":        t.name,
            "namespace":   t.namespace,
            "description": t.description,
            "parameters":  t.parameters,
            "is_async":    getattr(t, "is_async", False),
            "tags":        getattr(t, "tags", []),
            "aliases":     getattr(t, "aliases", []),
        }
        for t in all_tools
    ]


async def _show_validation_info(tm: ToolManager, provider: str, console) -> List[Dict[str, Any]]:
    """Show detailed validation information."""
    console.print(f"[cyan]Tool Validation Report for {provider}[/cyan]")
    
    if not hasattr(tm, 'get_validation_summary'):
        console.print("[yellow]Validation not available - using basic ToolManager[/yellow]")
        return []
    
    # Get validation summary
    summary = tm.get_validation_summary()
    
    # Create validation summary table
    summary_table = Table(title="Validation Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="green")
    
    summary_table.add_row("Total Tools", str(summary.get("total_tools", 0)))
    summary_table.add_row("Valid Tools", str(summary.get("valid_tools", 0)))
    summary_table.add_row("Invalid Tools", str(summary.get("invalid_tools", 0)))
    summary_table.add_row("User Disabled", str(summary.get("disabled_by_user", 0)))
    summary_table.add_row("Validation Disabled", str(summary.get("disabled_by_validation", 0)))
    
    console.print(summary_table)
    
    # Show validation errors
    errors = summary.get("validation_errors", [])
    if errors:
        console.print(f"\n[red]Validation Errors ({len(errors)}):[/red]")
        
        errors_table = Table()
        errors_table.add_column("Tool", style="yellow")
        errors_table.add_column("Error", style="red")
        errors_table.add_column("Reason", style="dim")
        
        for error in errors[:10]:  # Show first 10 errors
            errors_table.add_row(
                error.get("tool", "unknown"),
                error.get("error", "No error message")[:80] + ("..." if len(error.get("error", "")) > 80 else ""),
                error.get("reason", "unknown")
            )
        
        console.print(errors_table)
        
        if len(errors) > 10:
            console.print(f"[dim]... and {len(errors) - 10} more errors[/dim]")
    
    # Show disabled tools
    disabled = summary.get("disabled_tools", {})
    if disabled:
        console.print(f"\n[yellow]Disabled Tools ({len(disabled)}):[/yellow]")
        
        disabled_table = Table()
        disabled_table.add_column("Tool", style="yellow")
        disabled_table.add_column("Reason", style="red")
        
        for tool, reason in disabled.items():
            disabled_table.add_row(tool, reason)
        
        console.print(disabled_table)
    
    # Show auto-fix status
    if hasattr(tm, 'is_auto_fix_enabled'):
        auto_fix_status = "Enabled" if tm.is_auto_fix_enabled() else "Disabled"
        console.print(f"\n[cyan]Auto-fix:[/cyan] {auto_fix_status}")
    
    # Show helpful commands
    console.print(f"\n[dim]Commands:[/dim]")
    console.print(f"  [dim]• /tools-disable <tool_name>  - Disable a tool[/dim]")
    console.print(f"  [dim]• /tools-enable <tool_name>   - Enable a tool[/dim]")
    console.print(f"  [dim]• /tools-validate             - Re-run validation[/dim]")
    console.print(f"  [dim]• /tools-autofix on          - Enable auto-fixing[/dim]")
    
    return [{"validation_summary": summary}]


# ────────────────────────────────────────────────────────────────────────────────
# sync wrapper - for legacy CLI paths
# ────────────────────────────────────────────────────────────────────────────────
def tools_action(
    tm: ToolManager,
    *,
    show_details: bool = False,
    show_raw: bool = False,
    show_validation: bool = False,
    provider: str = "openai",
) -> List[Dict[str, Any]]:
    """
    Blocking wrapper around :pyfunc:`tools_action_async`.

    Raises
    ------
    RuntimeError
        If called from inside a running event-loop.
    """
    return run_blocking(
        tools_action_async(
            tm, 
            show_details=show_details, 
            show_raw=show_raw, 
            show_validation=show_validation,
            provider=provider
        )
    )

__all__ = ["tools_action_async", "tools_action"]