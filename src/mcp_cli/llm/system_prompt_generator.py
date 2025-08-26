# mcp_cli/llm/system_prompt_generator.py
import json


class SystemPromptGenerator:
    """
    A class for generating system prompts dynamically based on tools JSON and user inputs.
    """

    def __init__(self):
        """
        Initialize the SystemPromptGenerator with a default system prompt template.
        """
        self.template = """
        In this environment you have access to a set of tools you can use to answer the user's question.
        {{ FORMATTING INSTRUCTIONS }}
        String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
        Here are the functions available in JSONSchema format:
        {{ TOOL DEFINITIONS IN JSON SCHEMA }}
        {{ USER SYSTEM PROMPT }}
        {{ TOOL CONFIGURATION }}
        """
        self.default_user_system_prompt = "You are an intelligent assistant capable of using tools to solve user queries effectively."
        self.default_tool_config = "No additional configuration is required."

    def generate_prompt(
        self, tools: dict, user_system_prompt: str = None, tool_config: str = None
    ) -> str:
        """
        Generate a system prompt based on the provided tools JSON, user prompt, and tool configuration.

        Args:
            tools (dict): The tools JSON containing definitions of the available tools.
            user_system_prompt (str): A user-provided description or instruction for the assistant (optional).
            tool_config (str): Additional tool configuration information (optional).

        Returns:
            str: The dynamically generated system prompt.
        """

        # set the user system prompt
        user_system_prompt = user_system_prompt or self.default_user_system_prompt

        # set the tools config
        tool_config = tool_config or self.default_tool_config

        # get the tools schema
        tools_json_schema = json.dumps(tools, indent=2)

        # perform replacements
        prompt = self.template.replace(
            "{{ TOOL DEFINITIONS IN JSON SCHEMA }}", tools_json_schema
        )
        prompt = prompt.replace("{{ FORMATTING INSTRUCTIONS }}", "")
        prompt = prompt.replace("{{ USER SYSTEM PROMPT }}", user_system_prompt)
        prompt = prompt.replace("{{ TOOL CONFIGURATION }}", tool_config)

        # return the prompt
        return prompt
