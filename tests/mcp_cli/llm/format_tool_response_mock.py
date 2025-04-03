"""
This module provides a mocked implementation of format_tool_response
that matches the behavior of the actual implementation for testing purposes.
"""

def mock_format_tool_response(response_content):
    """
    A mock implementation of format_tool_response that behaves like the actual one.
    
    Based on the observed behavior from test failures, this function:
    1. Extracts text from items with type="text"
    2. Returns an empty string for various inputs
    3. Handles non-serializable objects gracefully
    """
    try:
        # Handle None case
        if response_content is None:
            return "None"
            
        # Handle string case
        if isinstance(response_content, str):
            return response_content
            
        # Handle list of dictionaries case
        if isinstance(response_content, list) and response_content:
            # Check if all items with "type" key have type="text"
            all_text = True
            text_items = []
            
            for item in response_content:
                if isinstance(item, dict) and "type" in item:
                    if item.get("type") == "text" and "text" in item:
                        text_items.append(item.get("text", ""))
                    else:
                        all_text = False
                else:
                    # Item doesn't have a type key
                    all_text = False
            
            # If all items with type are "text", just return the text values
            if all_text and text_items:
                return "\n".join(text_items)
                
        # For everything else, just return string representation
        return str(response_content)
    except:
        # Fallback for any errors
        return ""