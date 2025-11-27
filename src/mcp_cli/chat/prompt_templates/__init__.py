"""Composable prompt templates for system prompts."""

from mcp_cli.chat.prompt_templates.base import PromptTemplate
from mcp_cli.chat.prompt_templates.playbook import PlaybookPromptTemplate
from mcp_cli.chat.prompt_templates.general import GeneralGuidelinesTemplate

__all__ = [
    "PromptTemplate",
    "PlaybookPromptTemplate",
    "GeneralGuidelinesTemplate",
]
