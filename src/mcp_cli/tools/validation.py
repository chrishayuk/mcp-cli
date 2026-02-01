"""mcp_cli.tools.validation - Tool schema validation and filtering system.

SIMPLIFIED: Focus on auto-fixing rather than strict validation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from mcp_cli.tools.models import ValidationResult

logger = logging.getLogger(__name__)


class ToolSchemaValidator:
    """Validates tool schemas for compatibility with different LLM providers."""

    @staticmethod
    def validate_openai_schema(tool_def: dict[str, Any]) -> ValidationResult:
        """
        Validate a tool definition against OpenAI's function calling schema.
        SIMPLIFIED: Always attempt auto-fix instead of strict validation.

        Returns:
            ValidationResult with validation status and any errors
        """
        try:
            # Basic structure check - tool_def is already typed as Dict
            if "function" not in tool_def:
                return ValidationResult.failure(
                    "Tool definition missing 'function' property"
                )

            function = tool_def.get("function", {})
            if not isinstance(function, dict):
                return ValidationResult.failure("Function must be a dictionary")

            # Check name
            name = function.get("name", "")
            if not name or not isinstance(name, str):
                return ValidationResult.failure(
                    "Function name must be a non-empty string"
                )

            # Check for forbidden characters in name
            import re

            if not re.match(r"^[a-zA-Z0-9_-]+$", name):
                return ValidationResult.failure(
                    f"Function name '{name}' contains invalid characters. Only a-z, A-Z, 0-9, _, - allowed"
                )

            # Check for unsupported properties that would cause OpenAI errors
            unsupported_props = ["title", "examples", "deprecated", "version", "tags"]
            for prop in unsupported_props:
                if prop in function:
                    return ValidationResult.failure(
                        f"Function contains unsupported property '{prop}'"
                    )

            # Check parameters if present
            if "parameters" in function:
                parameters = function["parameters"]
                if not isinstance(parameters, dict):
                    return ValidationResult.failure("Parameters must be a dictionary")

                # Check for array schemas without items
                array_errors = ToolSchemaValidator._check_array_schemas(parameters)
                if array_errors:
                    return ValidationResult.failure(
                        f"Array schema issues: {'; '.join(array_errors)}"
                    )

            # If we get here, the tool looks good
            return ValidationResult.success()

        except Exception as e:
            return ValidationResult.failure(f"Validation error: {str(e)}")

    @staticmethod
    def _check_array_schemas(obj: Any, path: str = "") -> list[str]:
        """Recursively check for array schemas missing 'items' property."""
        errors = []

        if isinstance(obj, dict):
            # Check if this is an array schema without items
            if obj.get("type") == "array" and "items" not in obj:
                errors.append(
                    f"Array schema at {path or 'root'} missing 'items' property"
                )

            # Check anyOf/oneOf/allOf for array issues
            for schema_key in ["anyOf", "oneOf", "allOf"]:
                if schema_key in obj and isinstance(obj[schema_key], list):
                    for i, schema in enumerate(obj[schema_key]):
                        sub_errors = ToolSchemaValidator._check_array_schemas(
                            schema,
                            f"{path}.{schema_key}[{i}]"
                            if path
                            else f"{schema_key}[{i}]",
                        )
                        errors.extend(sub_errors)

            # Recursively check all properties
            for key, value in obj.items():
                if key not in ["anyOf", "oneOf", "allOf"]:  # Already checked above
                    sub_path = f"{path}.{key}" if path else key
                    sub_errors = ToolSchemaValidator._check_array_schemas(
                        value, sub_path
                    )
                    errors.extend(sub_errors)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                sub_path = f"{path}[{i}]" if path else f"[{i}]"
                sub_errors = ToolSchemaValidator._check_array_schemas(item, sub_path)
                errors.extend(sub_errors)

        return errors

    @staticmethod
    def fix_array_schemas(parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Attempt to fix common array schema issues.

        Returns:
            Fixed parameters schema
        """
        # parameters is already typed as Dict, no need to check
        # if not isinstance(parameters, dict):
        #     return parameters

        fixed = json.loads(json.dumps(parameters))  # Deep copy
        ToolSchemaValidator._fix_array_schemas_recursive(fixed)
        return cast(dict[str, Any], fixed)

    @staticmethod
    def _fix_array_schemas_recursive(obj: Any) -> None:
        """Recursively fix array schemas in place."""
        if isinstance(obj, dict):
            # Fix array schemas without items
            if obj.get("type") == "array" and "items" not in obj:
                logger.debug(f"Fixing array schema without items: {obj}")
                obj["items"] = {"type": "string"}  # Default to string items

            # Fix anyOf/oneOf/allOf
            for schema_key in ["anyOf", "oneOf", "allOf"]:
                if schema_key in obj and isinstance(obj[schema_key], list):
                    for schema in obj[schema_key]:
                        ToolSchemaValidator._fix_array_schemas_recursive(schema)

            # Recursively fix all values
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    ToolSchemaValidator._fix_array_schemas_recursive(value)

        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    ToolSchemaValidator._fix_array_schemas_recursive(item)

    @staticmethod
    def fix_openai_compatibility(tool_def: dict[str, Any]) -> dict[str, Any]:
        """
        Fix OpenAI compatibility issues by removing unsupported properties.

        Returns:
            Fixed tool definition
        """
        # tool_def is already typed as Dict, no need to check
        # if not isinstance(tool_def, dict):
        #     return tool_def

        fixed = json.loads(json.dumps(tool_def))  # Deep copy

        # Remove unsupported properties from function level
        if "function" in fixed and isinstance(fixed["function"], dict):
            function = fixed["function"]

            # List of properties that OpenAI doesn't support but are commonly used
            unsupported_props = [
                "title",
                "examples",
                "deprecated",
                "version",
                "tags",
                "summary",
            ]

            removed_props = []
            for prop in unsupported_props:
                if prop in function:
                    removed_props.append(prop)
                    del function[prop]

            if removed_props:
                logger.debug(
                    f"Removed unsupported properties {removed_props} from function '{function.get('name', 'unknown')}'"
                )

            # Also clean parameters if present
            if "parameters" in function and isinstance(function["parameters"], dict):
                parameters = function["parameters"]

                # Fix array schemas in parameters
                fixed["function"]["parameters"] = ToolSchemaValidator.fix_array_schemas(
                    parameters
                )

                # Remove unsupported properties from parameters level too
                param_unsupported = ["title", "examples", "deprecated"]
                param_removed = []
                for prop in param_unsupported:
                    if prop in fixed["function"]["parameters"]:
                        param_removed.append(prop)
                        del fixed["function"]["parameters"][prop]

                if param_removed:
                    logger.debug(
                        f"Removed unsupported parameter properties {param_removed}"
                    )

        return cast(dict[str, Any], fixed)

    @staticmethod
    def validate_and_fix_tool(
        tool_def: dict[str, Any], provider: str = "openai"
    ) -> tuple[bool, dict[str, Any], str | None]:
        """
        Comprehensive tool validation and fixing.

        Returns:
            Tuple of (is_valid, fixed_tool_def, error_message)
        """
        if provider != "openai":
            # For non-OpenAI providers, assume valid
            return True, tool_def, None

        # First, try to fix the tool
        try:
            fixed_tool = ToolSchemaValidator.fix_openai_compatibility(tool_def)

            # Then validate the fixed version
            validation = ToolSchemaValidator.validate_openai_schema(fixed_tool)

            return validation.is_valid, fixed_tool, validation.error_message

        except Exception as e:
            return False, tool_def, f"Error during fix/validation: {str(e)}"
