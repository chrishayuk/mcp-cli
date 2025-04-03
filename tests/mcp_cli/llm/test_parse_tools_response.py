"""
Tests for the parse_tool_response function.
"""
import pytest
import json

from mcp_cli.llm.tools_handler import parse_tool_response

class TestParseToolResponse:
    """Tests for the parse_tool_response function."""
    
    def test_parse_valid_tool_response(self):
        """Test that a valid tool call in XML format is parsed correctly."""
        xml_response = "<function=testFunction>{\"arg1\": \"value1\", \"arg2\": 42}</function>"
        result = parse_tool_response(xml_response)
        
        assert result is not None
        assert result["function"] == "testFunction"
        assert result["arguments"] == {"arg1": "value1", "arg2": 42}
    
    def test_parse_invalid_tool_response(self):
        """Test that an invalid XML format returns None."""
        invalid_response = "This is not a valid function call"
        result = parse_tool_response(invalid_response)
        
        assert result is None
    
    def test_parse_tool_response_with_invalid_json(self):
        """Test parsing with invalid JSON but valid XML format."""
        xml_response = "<function=testFunction>not valid json</function>"
        result = parse_tool_response(xml_response)
        
        assert result is None
    
    def test_parse_with_complex_json(self):
        """Test parsing with complex nested JSON."""
        complex_json = {
            "nested": {
                "array": [1, 2, 3],
                "object": {"key": "value"}
            },
            "array_of_objects": [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"}
            ]
        }
        
        xml_response = f"<function=complexFunction>{json.dumps(complex_json)}</function>"
        result = parse_tool_response(xml_response)
        
        assert result is not None
        assert result["function"] == "complexFunction"
        assert result["arguments"] == complex_json
    
    def test_parse_with_empty_json(self):
        """Test parsing with empty JSON object."""
        xml_response = "<function=emptyArgs>{}</function>"
        result = parse_tool_response(xml_response)
        
        assert result is not None
        assert result["function"] == "emptyArgs"
        assert result["arguments"] == {}
    
    def test_parse_with_malformed_xml(self):
        """Test parsing with malformed XML tags."""
        malformed_responses = [
            "<function=test>{}",  # Missing closing tag
            "function=test>{}</function>",  # Missing opening bracket
            "<functiontest>{}</function>",  # Missing equals sign
            "<function=>{}</function>"  # Missing function name
        ]
        
        for response in malformed_responses:
            result = parse_tool_response(response)
            assert result is None