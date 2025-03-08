# src/cli/main.py
"""
Fixed version of the CLI with proper option passing to commands
"""
import logging
import os
import json
import sys
import anyio
import typer
import asyncio

# imports
from cli.config import load_config

# import mcp
from mcp.transport.stdio.stdio_client import stdio_client
from mcp.messages.initialize.send_messages import send_initialize

# Import all command modules
from cli.commands import ping, chat, prompts, tools, resources, interactive

# Configure logging
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

# setup typer
app = typer.Typer()
prompts_app = typer.Typer(help="Prompts commands")
tools_app = typer.Typer(help="Tools commands")
resources_app = typer.Typer(help="Resources commands")

app.add_typer(prompts_app, name="prompts")
app.add_typer(tools_app, name="tools")
app.add_typer(resources_app, name="resources")

# Common option definitions to be reused
CONFIG_FILE_OPTION = typer.Option(
    "server_config.json",
    "--config-file",
    help="Path to the JSON configuration file containing server details."
)

SERVER_OPTION = typer.Option(
    None,
    "--server",
    help="Server configuration to use. Default is the filesystem MCP server."
)

PROVIDER_OPTION = typer.Option(
    "openai",
    "--provider",
    help="LLM provider to use. Defaults to 'openai'.",
    case_sensitive=False
)

MODEL_OPTION = typer.Option(
    None,
    "--model",
    help="Model to use. Defaults to 'gpt-4o-mini' for openai and 'qwen2.5-coder' for ollama."
)

DISABLE_FILESYSTEM_OPTION = typer.Option(
    True,
    "--disable-filesystem",
    help="Disable the default filesystem MCP server access."
)

@app.callback(invoke_without_command=True)
def common_options(
    ctx: typer.Context,
    config_file: str = CONFIG_FILE_OPTION,
    server: str = SERVER_OPTION,
    provider: str = PROVIDER_OPTION,
    model: str = MODEL_OPTION,
    disable_filesystem: bool = DISABLE_FILESYSTEM_OPTION,
):
    """
    MCP Command-Line Tool
    
    Global options are specified here.
    If no subcommand is provided, chat mode is launched by default.
    """
    # Handle servers input
    servers = []
    if server:
        # Allow comma-separated servers
        servers = [s.strip() for s in server.split(",")]
    if not servers and not disable_filesystem:
        servers.append("filesystem")
    if not model:
        model = "gpt-4o-mini" if provider.lower() == "openai" else "qwen2.5-coder"
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model
    if not disable_filesystem:
        os.environ["SOURCE_FILESYSTEMS"] = json.dumps([os.getcwd()])
    
    # Save these values in Typer's context so subcommands can access them.
    ctx.obj = {"config_file": config_file, "servers": servers}
    
    # If no subcommand is invoked, launch chat mode by default.
    if ctx.invoked_subcommand is None:
        # We're using the same options in the individual commands,
        # so we just invoke the chat function directly
        chat_command(
            config_file=config_file,
            server=server,
            provider=provider,
            model=model,
            disable_filesystem=disable_filesystem
        )
        raise typer.Exit()

@app.command("ping")
def ping_command(
    config_file: str = CONFIG_FILE_OPTION,
    server: str = SERVER_OPTION,
    provider: str = PROVIDER_OPTION,
    model: str = MODEL_OPTION,
    disable_filesystem: bool = DISABLE_FILESYSTEM_OPTION,
):
    """Simple ping command."""
    servers = process_options(server, disable_filesystem, provider, model)
    run_command(ping.ping_run, config_file, servers)
    return 0

@app.command("chat")
def chat_command(
    config_file: str = CONFIG_FILE_OPTION,
    server: str = SERVER_OPTION,
    provider: str = PROVIDER_OPTION,
    model: str = MODEL_OPTION,
    disable_filesystem: bool = DISABLE_FILESYSTEM_OPTION,
):
    """Start a chat session."""
    servers = process_options(server, disable_filesystem, provider, model)
    run_command(chat.chat_run, config_file, servers)
    return 0

@app.command("interactive")
def interactive_command(
    config_file: str = CONFIG_FILE_OPTION,
    server: str = SERVER_OPTION,
    provider: str = PROVIDER_OPTION,
    model: str = MODEL_OPTION,
    disable_filesystem: bool = DISABLE_FILESYSTEM_OPTION,
):
    """Enter interactive mode with a command prompt."""
    servers = process_options(server, disable_filesystem, provider, model)
    run_command(interactive.interactive_mode, config_file, servers)
    return 0

@prompts_app.command("list")
def prompts_list_command(
    config_file: str = CONFIG_FILE_OPTION,
    server: str = SERVER_OPTION,
    provider: str = PROVIDER_OPTION,
    model: str = MODEL_OPTION,
    disable_filesystem: bool = DISABLE_FILESYSTEM_OPTION,
):
    """List available prompts."""
    servers = process_options(server, disable_filesystem, provider, model)
    run_command(prompts.prompts_list, config_file, servers)
    return 0

@tools_app.command("list")
def tools_list_command(
    config_file: str = CONFIG_FILE_OPTION,
    server: str = SERVER_OPTION,
    provider: str = PROVIDER_OPTION,
    model: str = MODEL_OPTION,
    disable_filesystem: bool = DISABLE_FILESYSTEM_OPTION,
):
    """List available tools."""
    servers = process_options(server, disable_filesystem, provider, model)
    run_command(tools.tools_list, config_file, servers)
    return 0

@tools_app.command("call")
def tools_call_command(
    config_file: str = CONFIG_FILE_OPTION,
    server: str = SERVER_OPTION,
    provider: str = PROVIDER_OPTION,
    model: str = MODEL_OPTION,
    disable_filesystem: bool = DISABLE_FILESYSTEM_OPTION,
):
    """Call a tool with JSON arguments."""
    servers = process_options(server, disable_filesystem, provider, model)
    run_command(tools.tools_call, config_file, servers)
    return 0

@resources_app.command("list")
def resources_list_command(
    config_file: str = CONFIG_FILE_OPTION,
    server: str = SERVER_OPTION,
    provider: str = PROVIDER_OPTION,
    model: str = MODEL_OPTION,
    disable_filesystem: bool = DISABLE_FILESYSTEM_OPTION,
):
    """List available resources."""
    servers = process_options(server, disable_filesystem, provider, model)
    run_command(resources.resources_list, config_file, servers)
    return 0

def process_options(server, disable_filesystem, provider, model):
    """Process common options and return the servers list."""
    servers = []
    if server:
        # Allow comma-separated servers
        servers = [s.strip() for s in server.split(",")]
    
    # Always add filesystem unless explicitly disabled
    if not disable_filesystem and "filesystem" not in servers:
        servers.append("filesystem")
        
    if not model:
        model = "gpt-4o-mini" if provider.lower() == "openai" else "qwen2.5-coder"
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model
    if not disable_filesystem:
        os.environ["SOURCE_FILESYSTEMS"] = json.dumps([os.getcwd()])
    return servers

def run_command(command_func, config_file, server_names):
    """Run a command with the specified servers."""
    async def _run_clients():
        server_streams = []
        context_managers = []
        clean_exit = False
        
        # Create all server connections first
        for sname in server_names:
            try:
                server_params = await load_config(config_file, sname)
                cm = stdio_client(server_params)
                streams = await cm.__aenter__()
                context_managers.append((cm, streams))
                
                r_stream, w_stream = streams
                
                # Initialize this server
                init_result = await send_initialize(r_stream, w_stream)
                if not init_result:
                    print(f"Server initialization failed for {sname}")
                    # Close this specific context manager
                    try:
                        await cm.__aexit__(None, None, None)
                    except Exception as e:
                        print(f"Error closing connection to {sname}: {e}")
                    context_managers.pop()
                    continue
                
                server_streams.append(streams)
            except Exception as e:
                print(f"Error connecting to server {sname}: {e}")
                continue
        
        # Run the command if we have valid streams
        try:
            if server_streams:
                # Check if it's interactive mode or chat mode by looking at the function name
                is_interactive = command_func.__name__ == 'interactive_mode'
                is_chat = command_func.__name__ == 'chat_run'
                
                if is_interactive or is_chat:
                    # For interactive/chat mode, wait for the return value
                    result = await command_func(server_streams)
                    if result is True:
                        # Mode returned True, signal clean exit
                        clean_exit = True
                else:
                    # For other commands, just run them
                    await command_func(server_streams)
            else:
                print("No valid server connections established")
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected, cleaning up...")
        except Exception as e:
            print(f"\nError in command: {e}")
        finally:
            # Suppress cleanup messages for clean exits
            should_log = not clean_exit
            
            # Clean up all context managers - but handle each one separately
            for i, (cm, _) in enumerate(list(context_managers)):
                try:
                    # Close each context manager with a timeout to prevent hanging
                    close_task = asyncio.create_task(cm.__aexit__(None, None, None))
                    try:
                        # Add a timeout for the cleanup
                        await asyncio.wait_for(close_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        # If timeout, log and continue
                        if should_log:
                            print(f"Connection cleanup {i+1}/{len(context_managers)} timed out")
                    except (asyncio.CancelledError, RuntimeError):
                        # Suppress these errors for clean exits
                        pass
                except Exception as e:
                    if should_log and not isinstance(e, (asyncio.CancelledError, RuntimeError)):
                        print(f"Error during server shutdown {i+1}/{len(context_managers)}: {e}")
    
    # Clear the screen
    os.system("cls" if sys.platform == "win32" else "clear")
    
    # Run the async function with anyio
    try:
        anyio.run(_run_clients)
    except KeyboardInterrupt:
        print("\nOperation interrupted. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    app()