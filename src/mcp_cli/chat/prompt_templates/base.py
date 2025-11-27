"""Base class for composable prompt templates."""

from abc import ABC, abstractmethod
from typing import Any


class PromptTemplate(ABC):
    """Base class for prompt templates."""

    @abstractmethod
    def render(self, context: dict[str, Any] | None = None) -> str:
        """Render the template with optional context.

        Args:
            context: Optional dictionary of context variables

        Returns:
            Rendered prompt string
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Template name for identification."""
        pass

    @property
    def enabled(self) -> bool:
        """Whether this template should be included.

        Override this to add conditional logic based on preferences, config, etc.
        Default is always enabled.
        """
        return True

    def __str__(self) -> str:
        """String representation."""
        return self.render()
