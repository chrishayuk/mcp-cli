# src/mcp_cli/commands/models/base_model.py
"""Base model for all command models with shared configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CommandBaseModel(BaseModel):
    """
    Base model for all command models.

    Provides shared configuration for validation, serialization, and behavior.
    """

    model_config = ConfigDict(
        frozen=False,  # Allow field modification after creation
        validate_assignment=True,  # Validate when fields are assigned
        str_strip_whitespace=True,  # Automatically strip whitespace from strings
        arbitrary_types_allowed=True,  # Allow non-Pydantic types (like type objects)
        use_enum_values=True,  # Use enum values instead of enum objects
        populate_by_name=True,  # Allow population by field name and alias
    )
