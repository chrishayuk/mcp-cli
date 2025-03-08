# src/cli/commands/register_commands.py
import typer
from cli.commands import ping, chat, prompts, tools, resources, interactive

# Import conversation history command from chat commands.
from cli.chat.commands.conversation_history import conversation_history_command

def ping_command(
    config_file: str = "server_config.json",
    server: str = None,
    provider: str = "openai",
    model: str = None,
    disable_filesystem: bool = False,
):
    """Simple ping command."""
    from cli.main import process_options, run_command
    servers, user_specified = process_options(server, disable_filesystem, provider, model)
    run_command(ping.ping_run, config_file, servers, user_specified)
    return 0

def chat_command(
    config_file: str = "server_config.json",
    server: str = None,
    provider: str = "openai",
    model: str = None,
    disable_filesystem: bool = False,
):
    """Start a chat session."""
    from cli.main import process_options, run_command
    servers, user_specified = process_options(server, disable_filesystem, provider, model)
    run_command(chat.chat_run, config_file, servers, user_specified)
    return 0

def interactive_command(
    config_file: str = "server_config.json",
    server: str = None,
    provider: str = "openai",
    model: str = None,
    disable_filesystem: bool = False,
):
    """Enter interactive mode with a command prompt."""
    from cli.main import process_options, run_command
    servers, user_specified = process_options(server, disable_filesystem, provider, model)
    run_command(interactive.interactive_mode, config_file, servers, user_specified)
    return 0

def prompts_list_command(
    config_file: str = "server_config.json",
    server: str = None,
    provider: str = "openai",
    model: str = None,
    disable_filesystem: bool = False,
):
    """List available prompts."""
    from cli.main import process_options, run_command
    servers, user_specified = process_options(server, disable_filesystem, provider, model)
    run_command(prompts.prompts_list, config_file, servers, user_specified)
    return 0

def tools_list_command(
    config_file: str = "server_config.json",
    server: str = None,
    provider: str = "openai",
    model: str = None,
    disable_filesystem: bool = False,
):
    """List available tools."""
    from cli.main import process_options, run_command
    servers, user_specified = process_options(server, disable_filesystem, provider, model)
    run_command(tools.tools_list, config_file, servers, user_specified)
    return 0

def tools_call_command(
    config_file: str = "server_config.json",
    server: str = None,
    provider: str = "openai",
    model: str = None,
    disable_filesystem: bool = False,
):
    """Call a tool with JSON arguments."""
    from cli.main import process_options, run_command
    servers, user_specified = process_options(server, disable_filesystem, provider, model)
    run_command(tools.tools_call, config_file, servers, user_specified)
    return 0

def resources_list_command(
    config_file: str = "server_config.json",
    server: str = None,
    provider: str = "openai",
    model: str = None,
    disable_filesystem: bool = False,
):
    """List available resources."""
    from cli.main import process_options, run_command
    servers, user_specified = process_options(server, disable_filesystem, provider, model)
    run_command(resources.resources_list, config_file, servers, user_specified)
    return 0

def register_commands(app: typer.Typer, process_options, run_command):
    """Register all commands on the provided Typer app."""
    app.command("ping")(ping_command)
    app.command("chat")(chat_command)
    app.command("interactive")(interactive_command)
    
    # Register conversation history command at the top level.
    app.command("conversation")(conversation_history_command)
    app.command("ch")(conversation_history_command)
    
    # Create sub-typer apps for prompts, tools, and resources.
    prompts_app = typer.Typer(help="Prompts commands")
    tools_app = typer.Typer(help="Tools commands")
    resources_app = typer.Typer(help="Resources commands")
    
    prompts_app.command("list")(prompts_list_command)
    tools_app.command("list")(tools_list_command)
    tools_app.command("call")(tools_call_command)
    resources_app.command("list")(resources_list_command)
    
    app.add_typer(prompts_app, name="prompts")
    app.add_typer(tools_app, name="tools")
    app.add_typer(resources_app, name="resources")

# Export chat_command so it can be imported in main.py.
__all__ = ["register_commands", "chat_command"]