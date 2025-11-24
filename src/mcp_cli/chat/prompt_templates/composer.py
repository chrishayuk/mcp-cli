"""Prompt template composer for building system prompts."""

from typing import Any
from mcp_cli.chat.prompt_templates.base import PromptTemplate


class PromptComposer:
    """Compose multiple prompt templates into a single system prompt."""

    def __init__(self):
        """Initialize the composer."""
        self.templates: list[PromptTemplate] = []

    def add_template(self, template: PromptTemplate) -> "PromptComposer":
        """Add a template to the composition.

        Args:
            template: Template to add

        Returns:
            Self for chaining
        """
        self.templates.append(template)
        return self

    def compose(self, context: dict[str, Any] | None = None) -> str:
        """Compose all enabled templates into a single prompt.

        Args:
            context: Optional context dictionary passed to all templates

        Returns:
            Composed system prompt string
        """
        sections = []

        for template in self.templates:
            # Only include enabled templates
            if template.enabled:
                rendered = template.render(context)
                if rendered.strip():  # Only add non-empty sections
                    sections.append(rendered)

        # Join sections with blank lines
        return "\n".join(sections)

    def clear(self) -> None:
        """Clear all templates."""
        self.templates.clear()

    def get_enabled_templates(self) -> list[str]:
        """Get names of all enabled templates.

        Returns:
            List of template names that are currently enabled
        """
        return [t.name for t in self.templates if t.enabled]

    def __str__(self) -> str:
        """String representation showing enabled templates."""
        enabled = self.get_enabled_templates()
        return f"PromptComposer(enabled_templates={enabled})"
