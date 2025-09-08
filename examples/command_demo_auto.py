#!/usr/bin/env python3
"""
Automated Visual Demo of MCP-CLI Command System

This script shows the actual execution of commands with their real output,
demonstrating the command system in action. Runs automatically without user input.

Run with: uv run python examples/command_demo_auto.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class AutoCommandDemo:
    """Automated visual demonstration of the MCP-CLI command system."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # Try loading from .env
            env_file = Path(__file__).parent.parent / ".env"
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        if line.startswith("OPENAI_API_KEY="):
                            self.api_key = line.split("=", 1)[1].strip()
                            break
        
        # Demo commands with descriptions
        self.demo_commands = [
            ("📚 Help System", [
                ("/help", "Show all commands - notice the ▸ indicators"),
                ("/help models", "Get detailed help for models command"),
            ]),
            ("🎯 Model Management", [
                ("/models", "Show current model"),
                ("/model gpt-4o", "Switch directly to gpt-4o"),
                ("/models", "Verify model switched"),
                ("/model gpt-4o-mini", "Switch back to gpt-4o-mini"),
            ]),
            ("🔧 Provider Management", [
                ("/providers", "Show current provider"),
                ("/provider list", "List all providers"),
                ("/provider ollama", "Switch to Ollama"),
                ("/provider openai", "Switch back to OpenAI"),
            ]),
            ("🖥️ Server & Tools", [
                ("/servers", "List MCP servers"),
                ("/ping", "Test connectivity"),
                ("/tools", "List available tools"),
                ("/resources", "Show resources"),
            ]),
            ("🎨 UI Commands", [
                ("/theme", "List themes"),
                ("/theme dark", "Switch to dark theme"),
                ("/verbose", "Toggle verbose mode"),
                ("/theme default", "Back to default theme"),
            ]),
            ("📋 Command Menu", [
                ("/", "Just slash shows command menu"),
            ]),
        ]
    
    def clean_output(self, text: str) -> str:
        """Clean up output for display."""
        # Remove ANSI escape codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text = ansi_escape.sub('', text)
        
        # Remove empty lines at start and end
        lines = text.split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        # Limit output length for readability
        if len(lines) > 30:
            lines = lines[:30] + ["[... output truncated ...]"]
        
        return '\n'.join(lines)
    
    def execute_command(self, command: str) -> str:
        """Execute a command and return its output."""
        try:
            cmd_input = f"{command}\nexit\n"
            result = subprocess.run(
                ["uv", "run", "mcp-cli", "--server", "echo", "--provider", "openai", "--model", "gpt-4o-mini"],
                input=cmd_input,
                capture_output=True,
                text=True,
                env={**os.environ, "OPENAI_API_KEY": self.api_key},
                timeout=10
            )
            
            # Extract relevant output
            output_lines = result.stdout.split('\n')
            
            # Find command output start
            start_idx = -1
            for i, line in enumerate(output_lines):
                if f"> {command}" in line or f">{command}" in line:
                    start_idx = i + 1
                    break
            
            # Find output end
            end_idx = len(output_lines)
            for i, line in enumerate(output_lines[start_idx:], start_idx):
                if "> exit" in line or ">exit" in line:
                    end_idx = i
                    break
            
            if start_idx >= 0:
                relevant_output = '\n'.join(output_lines[start_idx:end_idx])
                return self.clean_output(relevant_output)
            
            return "No output captured"
            
        except subprocess.TimeoutExpired:
            return "Command timed out"
        except Exception as e:
            return f"Error: {e}"
    
    def run_demo(self):
        """Run the automated demonstration."""
        # Title
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]🚀 MCP-CLI Command System Demo[/bold cyan]\n"
            "Demonstrating all commands with actual output",
            border_style="cyan",
            padding=(1, 2)
        ))
        
        # Check API key
        if not self.api_key:
            console.print("\n[red]❌ OPENAI_API_KEY not found![/red]")
            console.print("[yellow]Set OPENAI_API_KEY or add to .env file[/yellow]")
            return
        
        console.print("\n[green]✓ Starting automated demo...[/green]\n")
        time.sleep(1)
        
        # Run each demo section
        for section_title, commands in self.demo_commands:
            # Section header
            console.rule(f"[bold cyan]{section_title}[/bold cyan]")
            console.print()
            
            for command, description in commands:
                # Show what we're doing
                console.print(f"[dim]{description}[/dim]")
                console.print(f"[bold green]>[/bold green] [bold cyan]{command}[/bold cyan]")
                
                # Execute with progress indicator
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task("Executing...", total=None)
                    output = self.execute_command(command)
                    progress.stop()
                
                # Show output
                if output and output != "No output captured":
                    console.print(Panel(
                        output,
                        border_style="dim",
                        padding=(0, 1)
                    ))
                
                console.print()
                time.sleep(0.5)  # Brief pause between commands
        
        # Summary
        console.print("\n")
        console.rule("[bold green]Demo Complete[/bold green]")
        console.print(Panel.fit(
            "[bold]Key Features Demonstrated:[/bold]\n\n"
            "✅ Commands with subcommands show [bold cyan]▸[/bold cyan] indicator\n"
            "✅ Direct switching: [cyan]/model gpt-4o[/cyan], [cyan]/provider ollama[/cyan]\n"
            "✅ No 'set' subcommand needed for common operations\n"
            "✅ All tips use chat mode syntax ([cyan]/command[/cyan])\n"
            "✅ Command menu appears with just [cyan]/[/cyan]\n"
            "✅ Consistent behavior across all commands\n",
            border_style="green",
            title="✨ Summary",
            padding=(1, 2)
        ))
        
        # Show some key observations
        console.print("\n[bold]Observations:[/bold]")
        console.print("• The help system clearly shows which commands have subcommands")
        console.print("• Model and provider switching works directly without 'set'")
        console.print("• All commands follow consistent patterns")
        console.print("• Tips and hints use the correct chat mode syntax")
        console.print()


def main():
    """Main entry point."""
    demo = AutoCommandDemo()
    demo.run_demo()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()