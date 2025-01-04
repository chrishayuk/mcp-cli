# mcpcli/__main__.py
import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from typing import List

import anyio

# Rich imports
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from mcpcli.chat_handler import handle_chat_mode
from mcpcli.config import load_config
from mcpcli.messages.send_ping import send_ping
from mcpcli.messages.send_prompts import send_prompts_list
from mcpcli.messages.send_resources import send_resources_list
from mcpcli.messages.send_initialize_message import send_initialize
from mcpcli.messages.send_call_tool import send_call_tool
from mcpcli.messages.send_tools_list import send_tools_list
from mcpcli.transport.stdio.stdio_client import stdio_client

# Default path for the configuration file
DEFAULT_CONFIG_FILE = "server_config.json"

# Configure logging
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)


def signal_handler(sig, frame):
    # Ignore subsequent SIGINT signals
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # pretty exit
    print("\n[bold red]Goodbye![/bold red]")

    # Immediately and forcibly kill the process
    os.kill(os.getpid(), signal.SIGKILL)


# signal handler
signal.signal(signal.SIGINT, signal_handler)


async def handle_command(
    command: str,
    server_streams: List[tuple],
    server_names: List[str],
    tool_name: str = None
) -> bool:
    """Handle specific commands dynamically with multiple servers."""
    try:
        if command == "list-servers":
            try:
                with open(DEFAULT_CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                print("[cyan]\nAvailable Server Configurations:[/cyan]")
                for server_name, details in config.items():
                    print(Panel(
                        Markdown(f"### {server_name}\n{json.dumps(details, indent=2)}"),
                        style="green"
                    ))
            except FileNotFoundError:
                print(f"[red]Error: Configuration file '{DEFAULT_CONFIG_FILE}' not found[/red]")
            except json.JSONDecodeError:
                print(f"[red]Error: Invalid JSON in configuration file '{DEFAULT_CONFIG_FILE}'[/red]")
            return True
            
        elif command == "ping":
            print("[cyan]\nPinging Servers...[/cyan]")
            for i, (read_stream, write_stream) in enumerate(server_streams):
                result = await send_ping(read_stream, write_stream)
                server_num = i + 1
                if result:
                    ping_md = f"## Server {server_num} Ping Result\n\n✅ **Server is up and running**"
                    print(Panel(Markdown(ping_md), style="bold green"))
                else:
                    ping_md = f"## Server {server_num} Ping Result\n\n❌ **Server ping failed**"
                    print(Panel(Markdown(ping_md), style="bold red"))

        elif command == "list-tools":
            for i, (read_stream, write_stream) in enumerate(server_streams):
                server_num = i + 1
                server_name = server_names[i] if i < len(server_names) else f"Server {server_num}"
                print(f"[cyan]\nFetching tools from {server_name}...[/cyan]")
                response = await send_tools_list(read_stream, write_stream)
                tools_list = response.get("tools", [])

                if not tools_list:
                    print(f"[yellow]No tools available for {server_name}[/yellow]")
                else:
                    print(f"\n[bold cyan]{server_name} Tools:[/bold cyan]")
                    for t in tools_list:
                        name = t.get('name', 'Unnamed')
                        desc = t.get('description', 'No description')
                        schema = t.get('inputSchema', {})
                        properties = schema.get('properties', {})
                        required = schema.get('required', [])
                        
                        # Format parameters info
                        params_list = []
                        for param, details in properties.items():
                            req_str = " [red bold]required[/red bold]" if param in required else ""
                            desc_str = details.get('description', 'No description')
                            type_str = details.get('type', 'unknown')
                            
                            param_header = f"  [cyan]■[/cyan] [bold white]{param}[/bold white]{req_str}"
                            param_desc = f"    [dim white]{desc_str}[/dim white]"
                            param_type = f"    [magenta]type: {type_str}[/magenta]"
                            
                            param_line = f"{param_header}\n{param_desc}\n{param_type}"
                            params_list.append(param_line)
                        
                        params_info = "\n".join(params_list) if params_list else "  [dim]No parameters required[/dim]"
                        
                        # Build panel content in parts to avoid long lines
                        panel_content = [
                            f"[bold white]{name}[/bold white]",
                            f"[dim]{desc}[/dim]",
                            "",
                            "[yellow]Parameters:[/yellow]",
                            params_info
                        ]
                        
                        print(Panel(
                            "\n".join(panel_content),
                            expand=True,
                            padding=(1, 2),
                            border_style="cyan"
                        ))

        elif command == "describe-tool":
            if not tool_name:
                print("[red]Error: --tool argument is required for describe-tool command[/red]")
                return True

            print(f"[cyan]\nFetching details for tool '{tool_name}'...[/cyan]")
            for i, (read_stream, write_stream) in enumerate(server_streams):
                response = await send_tools_list(read_stream, write_stream)
                tools_list = response.get("tools", [])
                server_num = i + 1

                tool = next((t for t in tools_list if t.get('name') == tool_name), None)
                if tool:
                    name = tool.get('name', 'Unnamed')
                    desc = tool.get('description', 'No description')
                    params = tool.get('parameters', {}).get('properties', {})
                    required = tool.get('parameters', {}).get('required', [])
                    
                    # Format parameters info
                    params_info = "\n".join([
                        f"### {param}\n- Description: {details.get('description', 'No description')}\n"
                        f"- Type: `{details.get('type', 'unknown')}`\n"
                        f"- Required: {'Yes' if param in required else 'No'}"
                        for param, details in params.items()
                    ])
                    
                    tool_md = f"# {name}\n\n{desc}\n\n## Parameters\n\n{params_info}"
                    print(Panel(Markdown(tool_md), style="green"))
                    return True

            print(f"[red]Tool '{tool_name}' not found[/red]")
            return True

        elif command == "call-tool":
            tool_name = Prompt.ask(
                "[bold magenta]Enter tool name[/bold magenta]"
            ).strip()
            if not tool_name:
                print("[red]Tool name cannot be empty.[/red]")
                return True

            arguments_str = Prompt.ask(
                "[bold magenta]Enter tool arguments as JSON (e.g., {'key': 'value'})[/bold magenta]"
            ).strip()
            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError as e:
                print(f"[red]Invalid JSON arguments format:[/red] {e}")
                return True

            print(f"[cyan]\nCalling tool '{tool_name}' with arguments:\n[/cyan]")
            print(
                Panel(
                    Markdown(f"```json\n{json.dumps(arguments, indent=2)}\n```"),
                    style="dim",
                )
            )

            result = await send_call_tool(tool_name, arguments, server_streams)
            if result.get("isError"):
                print(f"[red]Error calling tool:[/red] {result.get('error')}")
            else:
                response_content = result.get("content", "No content")
                print(
                    Panel(
                        Markdown(f"### Tool Response\n\n{response_content}"),
                        style="green",
                    )
                )

        elif command == "list-resources":
            print("[cyan]\nFetching Resources List from all servers...[/cyan]")
            for i, (read_stream, write_stream) in enumerate(server_streams):
                response = await send_resources_list(read_stream, write_stream)
                resources_list = response.get("resources", []) if response else None
                server_num = i + 1

                if not resources_list:
                    resources_md = f"## Server {server_num} Resources List\n\nNo resources available."
                else:
                    resources_md = f"## Server {server_num} Resources List\n"
                    for r in resources_list:
                        if isinstance(r, dict):
                            json_str = json.dumps(r, indent=2)
                            resources_md += f"\n```json\n{json_str}\n```"
                        else:
                            resources_md += f"\n- {r}"
                print(
                    Panel(
                        Markdown(resources_md),
                        title=f"Server {server_num} Resources",
                        style="bold cyan",
                    )
                )

        elif command == "list-prompts":
            print("[cyan]\nFetching Prompts List from all servers...[/cyan]")
            for i, (read_stream, write_stream) in enumerate(server_streams):
                response = await send_prompts_list(read_stream, write_stream)
                prompts_list = response.get("prompts", [])
                server_num = i + 1

                if not prompts_list:
                    prompts_md = (
                        f"## Server {server_num} Prompts List\n\nNo prompts available."
                    )
                else:
                    prompts_md = f"## Server {server_num} Prompts List\n\n" + "\n".join(
                        [f"- {p}" for p in prompts_list]
                    )
                print(
                    Panel(
                        Markdown(prompts_md),
                        title=f"Server {server_num} Prompts",
                        style="bold cyan",
                    )
                )

        elif command == "chat":
            provider = os.getenv("LLM_PROVIDER", "openai")
            model = os.getenv("LLM_MODEL", "gpt-4o-mini")

            # Clear the screen first
            if sys.platform == "win32":
                os.system("cls")
            else:
                os.system("clear")

            chat_info_text = (
                "Welcome to the Chat!\n\n"
                f"**Provider:** {provider}  |  **Model:** {model}\n\n"
                "Type 'exit' to quit."
            )

            print(
                Panel(
                    Markdown(chat_info_text),
                    style="bold cyan",
                    title="Chat Mode",
                    title_align="center",
                )
            )
            await handle_chat_mode(server_streams, provider, model)

        elif command in ["quit", "exit"]:
            print("\n[bold red]Goodbye![/bold red]")
            return False

        elif command == "clear":
            if sys.platform == "win32":
                os.system("cls")
            else:
                os.system("clear")

        elif command == "help":
            help_md = """
# MCP CLI Commands

## Server Management
[cyan]list-servers[/cyan]
  List all available server configurations

[cyan]ping[/cyan]
  Check if servers are responsive

## Tool Commands
[cyan]list-tools[/cyan]
  Display all available tools and their parameters

[cyan]describe-tool[/cyan] [dim]--tool <tool-name>[/dim]
  Show detailed information about a specific tool

[cyan]call-tool[/cyan] [dim]--tool <tool-name> --tool-args '{"param": "value"}'[/dim]
  Execute a tool with the specified arguments

## Resource Management
[cyan]list-resources[/cyan]
  Display available resources

[cyan]list-prompts[/cyan]
  Display available prompts

## Interactive Mode
[cyan]chat[/cyan]
  Enter interactive chat mode with the LLM

## General
[cyan]clear[/cyan]
  Clear the screen

[cyan]help[/cyan]
  Show this help message

[cyan]quit/exit[/cyan]
  Exit the program

## Examples
Call a tool:
  [dim]mcp-cli --server sqlite call-tool --tool read-query --tool-args '{"query": "SELECT * FROM products"}'[/dim]

List tools:
  [dim]mcp-cli --server sqlite list-tools[/dim]

Describe a tool:
  [dim]mcp-cli --server sqlite describe-tool --tool read-query[/dim]
"""
            print(Panel(Markdown(help_md), style="bold blue", title="Help", border_style="cyan"))

        else:
            print(f"[red]\nUnknown command: {command}[/red]")
            print("[yellow]Type 'help' for available commands[/yellow]")
    except Exception as e:
        print(f"\n[red]Error executing command:[/red] {e}")

    return True


async def get_input():
    """Get input asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input().strip().lower())


async def interactive_mode(server_streams: List[tuple], server_names: List[str]):
    """Run the CLI in interactive mode with multiple servers."""
    welcome_text = """
# Welcome to the Interactive MCP Command-Line Tool (Multi-Server Mode)

Type 'help' for available commands or 'quit' to exit.
"""
    print(Panel(Markdown(welcome_text), style="bold cyan"))

    while True:
        try:
            command = Prompt.ask("[bold green]\n>[/bold green]").strip().lower()
            if not command:
                continue
            should_continue = await handle_command(command, server_streams, server_names)
            if not should_continue:
                return
        except EOFError:
            break
        except Exception as e:
            print(f"\n[red]Error:[/red] {e}")


class GracefulExit(Exception):
    """Custom exception for handling graceful exits."""

    pass


async def run(
    config_path: str,
    server_names: List[str],
    command: str = None,
    tool_name: str = None,
    tool_args: dict = None
) -> None:
    """Main function to manage server initialization, communication, and shutdown."""
    # Clear screen before rendering anything
    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")

    # Special case for list-servers command - doesn't need server connections
    if command == "list-servers":
        await handle_command(command, [], [], None)
        return

    # Load server configurations and establish connections for all servers
    server_streams = []
    context_managers = []
    for server_name in server_names:
        server_params = await load_config(config_path, server_name)

        # Establish stdio communication for each server
        cm = stdio_client(server_params)
        (read_stream, write_stream) = await cm.__aenter__()
        context_managers.append(cm)
        server_streams.append((read_stream, write_stream))

        init_result = await send_initialize(read_stream, write_stream)
        if not init_result:
            print(f"[red]Server initialization failed for {server_name}[/red]")
            return

    try:
        if command:
            if command == "call-tool" and tool_name:
                # Direct tool call mode
                read_stream, write_stream = server_streams[0]  # Use first server for now
                result = await send_call_tool(tool_name, tool_args or {}, read_stream, write_stream)
                if result.get("isError"):
                    print(f"[red]Error calling tool:[/red] {result.get('error')}")
                else:
                    response_content = result.get("content", "No content")
                    print(
                        Panel(
                            Markdown(f"### Tool Response\n\n{response_content}"),
                            style="green",
                        )
                    )
            else:
                # Single command mode
                await handle_command(command, server_streams, server_names, tool_name)
        else:
            # Interactive mode
            await interactive_mode(server_streams, server_names)
    finally:
        # Clean up all streams with a timeout
        with anyio.fail_after(2):  # 2 second timeout for cleanup
            for cm in context_managers:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception as e:
                    logging.error(f"Error during cleanup: {e}")


def cli_main():
    # setup the parser
    parser = argparse.ArgumentParser(
        description="MCP (Model Context Provider) Command-Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  List available tools:
    mcp-cli --server sqlite list-tools

  Call a tool:
    mcp-cli --server sqlite call-tool \\
      --tool read-query \\
      --tool-args '{"query": "SELECT * FROM products"}'

  Describe a specific tool:
    mcp-cli --server sqlite describe-tool --tool read-query

  Start interactive mode:
    mcp-cli --server sqlite""")

    server_group = parser.add_argument_group("CLI Commands")
    server_group.add_argument(
        "command",
        nargs="?",
        choices=[
            "ping", "list-tools", "list-resources", "list-prompts",
            "call-tool", "list-servers", "describe-tool"
        ],
        metavar="COMMAND",
        help="Command to execute (see categories below)")

    server_group.description = """
Server Commands
  list-servers     List server configurations
  ping            Check server status

Tool Commands
  list-tools      List available tools
  describe-tool   Show tool details
  call-tool      Execute a tool

Resource Commands
  list-resources  List resources
  list-prompts    List prompts

Note: Running without a command starts interactive mode."""

    options_group = parser.add_argument_group("Options")
    options_group.add_argument(
        "--config-file",
        default=DEFAULT_CONFIG_FILE,
        help="Path to the JSON configuration file (default: %(default)s)",
        metavar="PATH",
    )

    options_group.add_argument(
        "--server",
        action="append",
        dest="servers",
        help="Server configuration to use (can be specified multiple times)",
        default=[],
        metavar="SERVER",
    )

    options_group.add_argument(
        "--tool",
        help="Name of the tool to call or describe (required for call-tool and describe-tool)",
        metavar="TOOL",
    )

    options_group.add_argument(
        "--tool-args",
        help="""JSON string of tool arguments.
Example: '{"query": "SELECT * FROM table"}'""",
        metavar="JSON",
    )

    llm_group = parser.add_argument_group("LLM Options")
    llm_group.add_argument(
        "--provider",
        choices=["openai", "anthropic", "ollama"],
        default="openai",
        help="LLM provider to use (default: %(default)s)",
    )

    llm_group.add_argument(
        "--model",
        help="""Model to use. Defaults:
  openai    = gpt-4o-mini
  anthropic = claude-3-5-haiku-latest
  ollama    = qwen2.5-coder""",
        metavar="MODEL",
    )

    args = parser.parse_args()

    # Set default model based on provider
    model = args.model or (
        "gpt-4o-mini" if args.provider == "openai"
        else "claude-3-5-haiku-latest" if args.provider == "anthropic"
        else "qwen2.5-coder"
    )
    os.environ["LLM_PROVIDER"] = args.provider
    os.environ["LLM_MODEL"] = model

    try:
        # Handle direct tool call if specified
        if args.command == "call-tool":
            if not args.tool:
                print("[red]Error: --tool argument is required for call-tool command[/red]")
                sys.exit(1)
            try:
                tool_args = json.loads(args.tool_args) if args.tool_args else {}
            except json.JSONDecodeError:
                print("[red]Error: --tool-args must be valid JSON[/red]")
                sys.exit(1)
            
            result = anyio.run(run, args.config_file, args.servers, args.command, args.tool, tool_args)
        elif args.command == "describe-tool":
            if not args.tool:
                print("[red]Error: --tool argument is required for describe-tool command[/red]")
                sys.exit(1)
            result = anyio.run(run, args.config_file, args.servers, args.command, args.tool)
        else:
            result = anyio.run(run, args.config_file, args.servers, args.command)
        sys.exit(result)
    except Exception as e:
        print(f"[red]Error occurred:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()

