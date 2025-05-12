# mcp_cli/tools/adapter.py
"""
Adapters for transforming tool names and definitions for different LLM providers.
"""
import re
from typing import Dict, List

from mcp_cli.tools.models import ToolInfo


class ToolNameAdapter:
    """Handles adaptation between OpenAI-compatible tool names and MCP original names."""
    
    @staticmethod
    def to_openai_compatible(namespace: str, name: str) -> str:
        """
        Convert MCP tool name with namespace to OpenAI-compatible format.
        
        OpenAI requires tool names to match pattern: ^[a-zA-Z0-9_-]+$
        
        Args:
            namespace: Tool namespace
            name: Tool name
            
        Returns:
            OpenAI-compatible name (namespace_name with invalid chars replaced)
        """
        # First combine namespace and name with underscore
        combined = f"{namespace}_{name}"
        
        # Replace any characters that don't comply with OpenAI's pattern
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', combined)
        
        return sanitized
    
    @staticmethod
    def from_openai_compatible(openai_name: str) -> str:
        """
        Convert OpenAI-compatible name back to MCP format.
        
        Args:
            openai_name: OpenAI-compatible tool name
            
        Returns:
            Original MCP tool name with namespace (namespace.name)
        """
        # Check if there's an underscore to convert back to dot notation
        if '_' in openai_name:
            parts = openai_name.split('_', 1)
            return f"{parts[0]}.{parts[1]}"
        return openai_name
    
    @staticmethod
    def build_mapping(tools: List[ToolInfo]) -> Dict[str, str]:
        """
        Build a mapping between OpenAI names and original names.
        
        Args:
            tools: List of ToolInfo objects
            
        Returns:
            Dictionary mapping OpenAI names to original names
        """
        mapping = {}
        for tool in tools:
            openai_name = ToolNameAdapter.to_openai_compatible(tool.namespace, tool.name)
            original_name = f"{tool.namespace}.{tool.name}"
            mapping[openai_name] = original_name
        return mapping