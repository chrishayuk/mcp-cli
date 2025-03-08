# src/cli/chat/commands/tools.py
"""
Commands for working with MCP tools.
"""

from typing import List, Dict, Any
from rich import print
from rich.table import Table
from rich.console import Console

from cli.chat.commands import register_command


async def cmd_tools(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    List available tools or get details about a specific tool.
    
    Usage: /tools [tool_name]
    
    Examples:
    - /tools - Lists all available tools
    - /tools read_file - Shows details about the read_file tool
    """
    tools = context['tools']
    
    if len(cmd_parts) > 1:
        # Look up specific tool
        tool_name = cmd_parts[1]
        found = False
        for t in tools:
            if t['name'].lower() == tool_name.lower():
                found = True
                console = Console()
                console.print(f"[bold cyan]Tool: {t['name']}[/bold cyan]")
                
                # Print details
                if 'description' in t:
                    console.print(f"\n[green]Description:[/green] {t['description']}")
                
                if 'parameters' in t:
                    console.print("\n[green]Parameters:[/green]")
                    params_table = Table(show_header=True)
                    params_table.add_column("Name", style="cyan")
                    params_table.add_column("Type", style="magenta")
                    params_table.add_column("Required", style="yellow")
                    params_table.add_column("Description", style="green")
                    
                    for param_name, param_info in t['parameters'].items():
                        req = "Yes" if param_name in t.get('required', []) else "No"
                        desc = param_info.get('description', 'No description')
                        param_type = param_info.get('type', 'unknown')
                        params_table.add_row(param_name, param_type, req, desc)
                    
                    console.print(params_table)
                    
                # Print example usage if available
                if 'examples' in t:
                    console.print("\n[green]Examples:[/green]")
                    for i, example in enumerate(t['examples']):
                        console.print(f"\n[bold]Example {i+1}:[/bold]")
                        console.print(f"```json\n{example}\n```")
                break
        
        if not found:
            print(f"[yellow]Tool '{tool_name}' not found.[/yellow]")
    else:
        # List all tools
        tools_table = Table(title=f"{len(tools)} Available Tools")
        tools_table.add_column("Tool", style="cyan")
        tools_table.add_column("Description", style="green")
        
        for t in tools:
            name = t['name']
            desc = t.get('description', 'No description')
            # Truncate description if too long
            if len(desc) > 80:
                desc = desc[:77] + "..."
            tools_table.add_row(name, desc)
            
        console = Console()
        console.print(tools_table)
    
    return True


# Get tool names for auto-completion
def get_tool_completions(context):
    """Generate completions for tool names."""
    if 'tools' in context:
        return [f"/tools {t['name']}" for t in context['tools']]
    return []


# Register command with completions
# Note: We can't access context here at module level, 
# so completions will be added dynamically during runtime
register_command("/tools", cmd_tools)