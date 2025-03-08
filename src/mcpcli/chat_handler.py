# src/mcpcli/chat_handler.py
import json
import asyncio
import os
import time
from typing import Dict, List, Any, Tuple, Optional

from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.console import Console
from rich.spinner import Spinner

# Use prompt_toolkit for enhanced input with tab completion and history
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style

from mcpcli.llm_client import LLMClient
from mcpcli.system_prompt_generator import SystemPromptGenerator
from mcpcli.tools_handler import convert_to_openai_tools, fetch_tools, handle_tool_call

# Import the command handling system
from cli.chat.commands import handle_command, get_command_completions


# Define a custom completer for chat commands
class ChatCommandCompleter(Completer):
    def __init__(self, context):
        self.context = context
        
    def get_completions(self, document, complete_event):
        text = document.text
        
        # Only suggest completions for slash commands
        if text.lstrip().startswith('/'):
            word_before_cursor = document.get_word_before_cursor()
            
            # Get completions from command system
            completions = get_command_completions(text.lstrip())
            
            for completion in completions:
                # If completion already matches what's there, don't suggest it
                if text.lstrip() == completion:
                    continue
                    
                # For simple command completion, just return the command
                if ' ' not in completion:
                    yield Completion(
                        completion, 
                        start_position=-len(text.lstrip()),
                        style='fg:goldenrod'  # Darker gold/yellow color
                    )
                # For argument completion, provide the full arg
                else:
                    yield Completion(
                        completion.split()[-1], 
                        start_position=-len(word_before_cursor),
                        style='fg:goldenrod'
                    )


async def handle_chat_mode(server_streams, provider="openai", model="gpt-4o-mini"):
    """Enter chat mode with multi-call support for autonomous tool chaining."""
    console = Console()
    server_info = []  # Track server information for /servers command
    
    try:
        # Show spinner while fetching tools
        with console.status("[bold cyan]Fetching available tools...[/bold cyan]", spinner="dots"):
            tools = []
            for i, (read_stream, write_stream) in enumerate(server_streams):
                try:
                    server_name = f"Server {i+1}"
                    fetched_tools = await fetch_tools(read_stream, write_stream)
                    tools.extend(fetched_tools)
                    server_info.append({
                        "id": i+1,
                        "name": server_name,
                        "tools": len(fetched_tools),
                        "status": "Connected"
                    })
                except Exception as e:
                    server_info.append({
                        "id": i+1,
                        "name": f"Server {i+1}",
                        "tools": 0,
                        "status": f"Error: {str(e)}"
                    })
                    print(f"[yellow]Warning: Failed to fetch tools from Server {i+1}: {e}[/yellow]")
                    continue

        if not tools:
            print("[red]No tools available. Exiting chat mode.[/red]")
            return

        print(f"[green]Loaded {len(tools)} tools successfully.[/green]")
        
        system_prompt = generate_system_prompt(tools)
        openai_tools = convert_to_openai_tools(tools)
        
        # Initialize the LLM client
        client = LLMClient(provider=provider, model=model)
        conversation_history = [{"role": "system", "content": system_prompt}]

        # Create context dictionary used by commands
        context = {
            "conversation_history": conversation_history,
            "tools": tools,
            "client": client,
            "provider": provider,
            "model": model,
            "server_info": server_info,
            "server_streams": server_streams,
            "openai_tools": openai_tools,
            "exit_requested": False
        }
        
        # Custom styles for completion menu - simpler version
        style = Style.from_dict({
            # Don't highlight the completion menu background
            'completion-menu': 'bg:default',
            'completion-menu.completion': 'bg:default fg:goldenrod',
            'completion-menu.completion.current': 'bg:default fg:goldenrod bold',
            # Set auto-suggestion color to a very subtle shade
            'auto-suggestion': 'fg:ansibrightblack',
        })
        
        # Set up prompt_toolkit session with history and tab completion
        history_file = os.path.expanduser("~/.mcp_chat_history")
        session = PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=ChatCommandCompleter(context),
            complete_while_typing=True,
            style=style
        )

        while True:
            try:
                # Print the prompt with Rich to ensure consistent color
                print("[bold yellow]>[/bold yellow] ", end="", flush=True)
                
                # Use prompt_toolkit for enhanced input (tab completion, history)
                # Use an empty prompt since we already printed it with Rich
                user_message = await session.prompt_async("")
                user_message = user_message.strip()
                
                # Handle command inputs
                if not user_message:
                    continue
                
                if user_message.lower() in ["exit", "quit"]:
                    # Handle plain exit/quit commands without slash
                    print(Panel("Exiting chat mode.", style="bold red"))
                    break
                
                # Handle special commands
                if user_message.startswith('/'):
                    # Pass all context to command handler
                    handled = await handle_command(user_message, context)
                    
                    # Check if an exit was requested
                    if context.get('exit_requested', False):
                        break
                        
                    if handled:
                        continue
                
                # User panel in bold yellow
                user_panel_text = user_message if user_message else "[No Message]"
                print(Panel(user_panel_text, style="bold yellow", title="You"))

                # Add user message to history
                conversation_history.append({"role": "user", "content": user_message})
                
                # Use a spinner while waiting for the LLM response
                await process_conversation(context)

            except KeyboardInterrupt:
                print("\n[yellow]Chat interrupted. Type 'exit' to quit.[/yellow]")
            except EOFError:
                # EOF (Ctrl+D) should exit cleanly
                print(Panel("EOF detected. Exiting chat mode.", style="bold red"))
                break
            except Exception as e:
                print(f"[red]Error processing message:[/red] {e}")
                continue
    except asyncio.CancelledError:
        # Handle task cancellation gracefully
        print("[yellow]Chat task cancelled.[/yellow]")
    except Exception as e:
        print(f"[red]Error in chat mode:[/red] {e}")
    
    # Clean return to ensure proper exit
    return True


async def process_conversation(context: Dict[str, Any]):
    """
    Process the conversation loop, handling tool calls and responses.
    
    Args:
        context: Dictionary containing all required objects and state
    """
    client = context['client']
    conversation_history = context['conversation_history']
    openai_tools = context['openai_tools']
    server_streams = context['server_streams']
    
    try:
        while True:
            try:
                start_time = time.time()
                
                completion = client.create_completion(
                    messages=conversation_history,
                    tools=openai_tools,
                )

                response_content = completion.get("response", "No response")
                tool_calls = completion.get("tool_calls", [])
                
                # Calculate response time
                response_time = time.time() - start_time

                if tool_calls:
                    for tool_call in tool_calls:
                        # Extract tool_name and raw_arguments
                        if hasattr(tool_call, "function"):
                            tool_name = getattr(tool_call.function, "name", "unknown tool")
                            raw_arguments = getattr(tool_call.function, "arguments", {})
                        elif isinstance(tool_call, dict) and "function" in tool_call:
                            fn_info = tool_call["function"]
                            tool_name = fn_info.get("name", "unknown tool")
                            raw_arguments = fn_info.get("arguments", {})
                        else:
                            tool_name = "unknown tool"
                            raw_arguments = {}

                        # Handle JSON arguments
                        if isinstance(raw_arguments, str):
                            try:
                                raw_arguments = json.loads(raw_arguments)
                            except json.JSONDecodeError:
                                # If it's not valid JSON, just display as is
                                pass

                        # Now raw_arguments should be a dict or something we can pretty-print as JSON
                        tool_args_str = json.dumps(raw_arguments, indent=2)

                        tool_md = f"**Tool Call:** {tool_name}\n\n```json\n{tool_args_str}\n```"
                        print(
                            Panel(
                                Markdown(tool_md), style="bold magenta", title="Tool Invocation"
                            )
                        )

                        # Process tool call
                        with Console().status("[cyan]Executing tool...[/cyan]", spinner="dots"):
                            await handle_tool_call(tool_call, conversation_history, server_streams)
                    continue

                # Assistant panel with Markdown
                assistant_panel_text = response_content if response_content else "[No Response]"
                footer = f"Response time: {response_time:.2f}s"
                print(
                    Panel(
                        Markdown(assistant_panel_text), 
                        style="bold blue", 
                        title="Assistant",
                        subtitle=footer
                    )
                )
                conversation_history.append({"role": "assistant", "content": response_content})
                break
            except asyncio.CancelledError:
                # Handle cancellation during API calls
                raise
            except Exception as e:
                print(f"[red]Error during conversation processing:[/red] {e}")
                conversation_history.append(
                    {"role": "assistant", "content": f"I encountered an error: {str(e)}"}
                )
                break
    except asyncio.CancelledError:
        # Propagate cancellation up
        raise


def generate_system_prompt(tools):
    """
    Generate a concise system prompt for the assistant.

    This prompt is internal and not displayed to the user.
    """
    prompt_generator = SystemPromptGenerator()
    tools_json = {"tools": tools}

    system_prompt = prompt_generator.generate_prompt(tools_json)
    system_prompt += """

**GENERAL GUIDELINES:**

1. Step-by-step reasoning:
   - Analyze tasks systematically.
   - Break down complex problems into smaller, manageable parts.
   - Verify assumptions at each step to avoid errors.
   - Reflect on results to improve subsequent actions.

2. Effective tool usage:
   - Explore:
     - Identify available information and verify its structure.
     - Check assumptions and understand data relationships.
   - Iterate:
     - Start with simple queries or actions.
     - Build upon successes, adjusting based on observations.
   - Handle errors:
     - Carefully analyze error messages.
     - Use errors as a guide to refine your approach.
     - Document what went wrong and suggest fixes.

3. Clear communication:
   - Explain your reasoning and decisions at each step.
   - Share discoveries transparently with the user.
   - Outline next steps or ask clarifying questions as needed.

EXAMPLES OF BEST PRACTICES:

- Working with databases:
  - Check schema before writing queries.
  - Verify the existence of columns or tables.
  - Start with basic queries and refine based on results.

- Processing data:
  - Validate data formats and handle edge cases.
  - Ensure integrity and correctness of results.

- Accessing resources:
  - Confirm resource availability and permissions.
  - Handle missing or incomplete data gracefully.

REMEMBER:
- Be thorough and systematic.
- Each tool call should have a clear and well-explained purpose.
- Make reasonable assumptions if ambiguous.
- Minimize unnecessary user interactions by providing actionable insights.

EXAMPLES OF ASSUMPTIONS:
- Default sorting (e.g., descending order) if not specified.
- Assume basic user intentions, such as fetching top results by a common metric.
"""
    return system_prompt