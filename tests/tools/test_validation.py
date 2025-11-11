# tests/tools/test_validation.py
"""
Comprehensive tests for tools/validation.py module.
Target: 90%+ coverage
"""

import pytest
from mcp_cli.tools.validation import ToolSchemaValidator


class TestToolSchemaValidator:
    """Test ToolSchemaValidator class."""

    def test_validate_openai_schema_valid_tool(self):
        """Test validate_openai_schema with valid tool."""
        tool = {
            "type": "function",
            "function": {
                "name": "valid_tool",
                "description": "A valid tool",
                "parameters": {
                    "type": "object",
                    "properties": {"arg1": {"type": "string"}},
                },
            },
        }

        result = ToolSchemaValidator.validate_openai_schema(tool)
        assert result.is_valid is True
        assert result.error_message is None

    def test_validate_openai_schema_missing_function(self):
        """Test validate_openai_schema with missing function property."""
        tool = {"type": "function"}

        result = ToolSchemaValidator.validate_openai_schema(tool)
        assert result.is_valid is False
        assert "missing 'function' property" in result.error_message

    def test_validate_openai_schema_function_not_dict(self):
        """Test validate_openai_schema with non-dict function."""
        tool = {"type": "function", "function": "not a dict"}

        result = ToolSchemaValidator.validate_openai_schema(tool)
        assert result.is_valid is False
        assert "must be a dictionary" in result.error_message

    def test_validate_openai_schema_missing_name(self):
        """Test validate_openai_schema with missing name."""
        tool = {"type": "function", "function": {"description": "No name"}}

        result = ToolSchemaValidator.validate_openai_schema(tool)
        assert result.is_valid is False
        assert "name must be a non-empty string" in result.error_message

    def test_validate_openai_schema_empty_name(self):
        """Test validate_openai_schema with empty name."""
        tool = {"type": "function", "function": {"name": ""}}

        result = ToolSchemaValidator.validate_openai_schema(tool)
        assert result.is_valid is False
        assert "name must be a non-empty string" in result.error_message

    def test_validate_openai_schema_invalid_name_characters(self):
        """Test validate_openai_schema with invalid characters in name."""
        tool = {
            "type": "function",
            "function": {"name": "invalid@name", "description": "Test"},
        }

        result = ToolSchemaValidator.validate_openai_schema(tool)
        assert result.is_valid is False
        assert "contains invalid characters" in result.error_message

    @pytest.mark.parametrize(
        "name",
        ["valid_name", "ValidName", "valid-name", "valid123", "valid_name_123"],
    )
    def test_validate_openai_schema_valid_names(self, name):
        """Test validate_openai_schema with various valid names."""
        tool = {
            "type": "function",
            "function": {"name": name, "description": "Test"},
        }

        result = ToolSchemaValidator.validate_openai_schema(tool)
        assert result.is_valid is True
        assert result.error_message is None

    @pytest.mark.parametrize(
        "prop", ["title", "examples", "deprecated", "version", "tags"]
    )
    def test_validate_openai_schema_unsupported_properties(self, prop):
        """Test validate_openai_schema with unsupported properties."""
        tool = {
            "type": "function",
            "function": {"name": "tool", "description": "Test", prop: "value"},
        }

        result = ToolSchemaValidator.validate_openai_schema(tool)
        assert result.is_valid is False
        assert f"unsupported property '{prop}'" in result.error_message

    def test_validate_openai_schema_parameters_not_dict(self):
        """Test validate_openai_schema with non-dict parameters."""
        tool = {
            "type": "function",
            "function": {"name": "tool", "description": "Test", "parameters": "string"},
        }

        result = ToolSchemaValidator.validate_openai_schema(tool)
        assert result.is_valid is False
        assert "Parameters must be a dictionary" in result.error_message

    def test_validate_openai_schema_array_without_items(self):
        """Test validate_openai_schema with array missing items."""
        tool = {
            "type": "function",
            "function": {
                "name": "tool",
                "description": "Test",
                "parameters": {
                    "type": "object",
                    "properties": {"arr": {"type": "array"}},  # Missing items
                },
            },
        }

        result = ToolSchemaValidator.validate_openai_schema(tool)
        assert result.is_valid is False
        assert "Array schema" in result.error_message
        assert "missing 'items'" in result.error_message

    def test_check_array_schemas_nested(self):
        """Test _check_array_schemas with nested structures."""
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"arr": {"type": "array"}},  # Missing items
                }
            },
        }

        errors = ToolSchemaValidator._check_array_schemas(schema)
        assert len(errors) > 0
        assert any("missing 'items'" in e for e in errors)

    def test_check_array_schemas_anyof(self):
        """Test _check_array_schemas with anyOf."""
        schema = {
            "anyOf": [{"type": "array"}, {"type": "string"}]  # Array missing items
        }

        errors = ToolSchemaValidator._check_array_schemas(schema)
        assert len(errors) > 0

    def test_check_array_schemas_valid_array(self):
        """Test _check_array_schemas with valid array."""
        schema = {
            "type": "object",
            "properties": {"arr": {"type": "array", "items": {"type": "string"}}},
        }

        errors = ToolSchemaValidator._check_array_schemas(schema)
        assert len(errors) == 0

    def test_fix_array_schemas_simple(self):
        """Test fix_array_schemas with simple array."""
        parameters = {
            "type": "object",
            "properties": {"arr": {"type": "array"}},  # Missing items
        }

        fixed = ToolSchemaValidator.fix_array_schemas(parameters)

        assert fixed["properties"]["arr"]["items"] == {"type": "string"}

    def test_fix_array_schemas_nested(self):
        """Test fix_array_schemas with nested structures."""
        parameters = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"arr": {"type": "array"}},
                }
            },
        }

        fixed = ToolSchemaValidator.fix_array_schemas(parameters)

        assert fixed["properties"]["nested"]["properties"]["arr"]["items"] == {
            "type": "string"
        }

    def test_fix_array_schemas_preserves_existing_items(self):
        """Test fix_array_schemas preserves existing items."""
        parameters = {
            "type": "object",
            "properties": {"arr": {"type": "array", "items": {"type": "number"}}},
        }

        fixed = ToolSchemaValidator.fix_array_schemas(parameters)

        # Should not change existing items
        assert fixed["properties"]["arr"]["items"]["type"] == "number"

    def test_fix_array_schemas_with_anyof(self):
        """Test fix_array_schemas with anyOf."""
        parameters = {
            "anyOf": [
                {"type": "array"},  # Missing items
                {"type": "string"},
            ]
        }

        fixed = ToolSchemaValidator.fix_array_schemas(parameters)

        assert fixed["anyOf"][0]["items"] == {"type": "string"}

    def test_fix_openai_compatibility_removes_unsupported(self):
        """Test fix_openai_compatibility removes unsupported properties."""
        tool = {
            "type": "function",
            "function": {
                "name": "tool",
                "description": "Test",
                "title": "Remove me",
                "examples": ["example"],
                "deprecated": True,
                "version": "1.0",
                "tags": ["tag1"],
                "summary": "Summary",
                "parameters": {"type": "object"},
            },
        }

        fixed = ToolSchemaValidator.fix_openai_compatibility(tool)

        function = fixed["function"]
        assert "title" not in function
        assert "examples" not in function
        assert "deprecated" not in function
        assert "version" not in function
        assert "tags" not in function
        assert "summary" not in function
        assert "name" in function
        assert "description" in function

    def test_fix_openai_compatibility_preserves_valid_props(self):
        """Test fix_openai_compatibility preserves valid properties."""
        tool = {
            "type": "function",
            "function": {
                "name": "tool",
                "description": "Test",
                "parameters": {
                    "type": "object",
                    "properties": {"arg1": {"type": "string"}},
                },
            },
        }

        fixed = ToolSchemaValidator.fix_openai_compatibility(tool)

        assert fixed["function"]["name"] == "tool"
        assert fixed["function"]["description"] == "Test"
        assert "arg1" in fixed["function"]["parameters"]["properties"]

    def test_fix_openai_compatibility_fixes_arrays(self):
        """Test fix_openai_compatibility also fixes array schemas."""
        tool = {
            "type": "function",
            "function": {
                "name": "tool",
                "description": "Test",
                "parameters": {
                    "type": "object",
                    "properties": {"arr": {"type": "array"}},  # Missing items
                },
            },
        }

        fixed = ToolSchemaValidator.fix_openai_compatibility(tool)

        assert "items" in fixed["function"]["parameters"]["properties"]["arr"]

    def test_fix_openai_compatibility_removes_param_level_props(self):
        """Test fix_openai_compatibility removes unsupported param properties."""
        tool = {
            "type": "function",
            "function": {
                "name": "tool",
                "description": "Test",
                "parameters": {
                    "type": "object",
                    "title": "Remove me",
                    "examples": ["example"],
                    "deprecated": True,
                    "properties": {},
                },
            },
        }

        fixed = ToolSchemaValidator.fix_openai_compatibility(tool)

        params = fixed["function"]["parameters"]
        assert "title" not in params
        assert "examples" not in params
        assert "deprecated" not in params

    def test_validate_and_fix_tool_openai_success(self):
        """Test validate_and_fix_tool with OpenAI provider."""
        tool = {
            "type": "function",
            "function": {
                "name": "tool",
                "description": "Test",
                "title": "Remove me",
                "parameters": {"type": "object"},
            },
        }

        is_valid, fixed_tool, error = ToolSchemaValidator.validate_and_fix_tool(
            tool, "openai"
        )

        assert is_valid is True
        assert error is None
        assert "title" not in fixed_tool["function"]

    def test_validate_and_fix_tool_openai_failure(self):
        """Test validate_and_fix_tool with unfixable tool."""
        tool = {
            "type": "function",
            "function": {
                "name": "invalid@name",
                "description": "Test",
            },
        }

        is_valid, fixed_tool, error = ToolSchemaValidator.validate_and_fix_tool(
            tool, "openai"
        )

        assert is_valid is False
        assert error is not None
        assert "invalid characters" in error

    def test_validate_and_fix_tool_non_openai(self):
        """Test validate_and_fix_tool with non-OpenAI provider."""
        tool = {"name": "any_tool"}

        is_valid, fixed_tool, error = ToolSchemaValidator.validate_and_fix_tool(
            tool, "anthropic"
        )

        assert is_valid is True
        assert error is None
        assert fixed_tool == tool  # Should return unchanged

    def test_validate_and_fix_tool_exception_handling(self):
        """Test validate_and_fix_tool handles exceptions gracefully."""
        # Pass something that will cause an error
        tool = None

        is_valid, fixed_tool, error = ToolSchemaValidator.validate_and_fix_tool(
            tool, "openai"
        )

        assert is_valid is False
        assert error is not None
        assert "Error during fix/validation" in error

    def test_fix_array_schemas_recursive_with_list(self):
        """Test _fix_array_schemas_recursive with list input."""
        obj = [{"type": "array"}, {"type": "string"}]

        ToolSchemaValidator._fix_array_schemas_recursive(obj)

        assert obj[0]["items"] == {"type": "string"}

    def test_check_array_schemas_with_list(self):
        """Test _check_array_schemas with list input."""
        obj = [{"type": "array"}, {"type": "string"}]

        errors = ToolSchemaValidator._check_array_schemas(obj)

        assert len(errors) > 0

    def test_validate_openai_schema_exception_handling(self):
        """Test validate_openai_schema handles exceptions."""
        # Pass something that might cause an unexpected error
        tool = {"function": {"name": None}}  # Invalid name type

        result = ToolSchemaValidator.validate_openai_schema(tool)

        assert result.is_valid is False
        assert result.error_message is not None

    def test_fix_array_schemas_oneof_allof(self):
        """Test fix_array_schemas with oneOf and allOf."""
        parameters = {
            "oneOf": [{"type": "array"}],
            "allOf": [{"type": "object", "properties": {"arr": {"type": "array"}}}],
        }

        fixed = ToolSchemaValidator.fix_array_schemas(parameters)

        assert fixed["oneOf"][0]["items"] == {"type": "string"}
        assert fixed["allOf"][0]["properties"]["arr"]["items"] == {"type": "string"}
