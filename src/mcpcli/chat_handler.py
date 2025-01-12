# chat_handler.py
import json
import asyncio

from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from mcpcli.llm_client import LLMClient
from mcpcli.system_prompt_generator import SystemPromptGenerator
from mcpcli.tools_handler import convert_to_openai_tools, fetch_tools, handle_tool_call

async def get_input(prompt: str):
    """Get input asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: Prompt.ask(prompt).strip())

async def handle_chat_mode(server_streams, provider="openai", model="gpt-4o-mini"):
    """Enter chat mode with multi-call support for autonomous tool chaining."""
    try:
        tools = []
        for read_stream, write_stream in server_streams:
            tools.extend(await fetch_tools(read_stream, write_stream))

        # for (read_stream, write_stream) in server_streams:
        # tools = await fetch_tools(read_stream, write_stream)
        if not tools:
            print("[red]No tools available. Exiting chat mode.[/red]")
            return

        system_prompt = generate_system_prompt(tools)
        openai_tools = convert_to_openai_tools(tools)
        client = LLMClient(provider=provider, model=model)
        conversation_history = [{"role": "system", "content": system_prompt}]

        while True:
            try:
                # Change prompt to yellow
                user_message = await get_input("[bold yellow]>[/bold yellow]")
                if user_message.lower() in ["exit", "quit"]:
                    print(Panel("Exiting chat mode.", style="bold red"))
                    break

                # User panel in bold yellow
                user_panel_text = user_message if user_message else "[No Message]"
                print(Panel(user_panel_text, style="bold yellow", title="You"))

                conversation_history.append({"role": "user", "content": user_message})
                await process_conversation(
                    client, conversation_history, openai_tools, server_streams
                )

            except Exception as e:
                print(f"[red]Error processing message:[/red] {e}")
                continue
    except Exception as e:
        print(f"[red]Error in chat mode:[/red] {e}")


async def process_conversation(
    client, conversation_history, openai_tools, server_streams
):
    """Process the conversation loop, handling tool calls and responses."""
    while True:
        completion = client.create_completion(
            messages=conversation_history,
            tools=openai_tools,
        )

        response_content = completion.get("response", "No response")
        tool_calls = completion.get("tool_calls", [])

        if tool_calls:
            for tool_call in tool_calls:
                # Extract tool_name and raw_arguments as before
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

                # If raw_arguments is a string, try to parse it as JSON
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

                await handle_tool_call(tool_call, conversation_history, server_streams)
            continue

        # Assistant panel with Markdown
        assistant_panel_text = response_content if response_content else "[No Response]"
        print(
            Panel(Markdown(assistant_panel_text), style="bold blue", title="Assistant")
        )
        conversation_history.append({"role": "assistant", "content": response_content})
        break


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
