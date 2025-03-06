# mcp/messages/tools/tool_input_schema.py
from typing import Any, Dict, List
from pydantic import BaseModel

class ToolResult(BaseModel):
    """Model representing the result of a tool invocation."""
    content: List[Dict[str, Any]]
    isError: bool = False