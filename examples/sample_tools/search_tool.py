# examples/sample_tools/search_tool.py
import time
from typing import Dict, List

# imports
from chuk_tool_processor.registry.decorators import register_tool
from chuk_tool_processor.models.validated_tool import ValidatedTool


@register_tool(name="search")
class SearchTool(ValidatedTool):
    """Search the web for information."""

    class Arguments(ValidatedTool.Arguments):
        query: str
        num_results: int = 3

    class Result(ValidatedTool.Result):
        results: List[Dict[str, str]]

    def _execute(self, query: str, num_results: int) -> Dict:
        """Simulate web search."""
        time.sleep(0.8)  # pretend latency
        return {
            "results": [
                {
                    "title": f"Result {i + 1} for {query}",
                    "url": f"https://example.com/result{i + 1}",
                    "snippet": f"This is a search result about {query}.",
                }
                for i in range(num_results)
            ]
        }
