# mcp_cli/chat/system_prompt.py

# llm imports
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator
from mcp_cli.chat.prompt_templates.composer import PromptComposer
from mcp_cli.chat.prompt_templates.playbook import PlaybookPromptTemplate
from mcp_cli.chat.prompt_templates.general import GeneralGuidelinesTemplate


def generate_system_prompt(tools):
    """Generate a composable system prompt for the assistant.

    This uses a template-based system where different prompt sections
    can be conditionally included based on configuration/preferences.

    Args:
        tools: List of tool definitions for the LLM

    Returns:
        Composed system prompt string
    """
    # Generate base tool prompt
    prompt_generator = SystemPromptGenerator()
    tools_json = {"tools": tools}
    base_prompt = prompt_generator.generate_prompt(tools_json)

    # Compose additional prompt sections
    composer = PromptComposer()
    composer.add_template(PlaybookPromptTemplate())  # Auto-enabled if playbook is on
    composer.add_template(GeneralGuidelinesTemplate())  # Always enabled

    # Combine base prompt with composed sections
    additional_sections = composer.compose()

    return base_prompt + additional_sections
