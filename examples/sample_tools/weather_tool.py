# examples/sample_tools/weather_tool.py
import time

# imports
from chuk_tool_processor.registry.decorators import register_tool
from chuk_tool_processor.models.validated_tool import ValidatedTool


@register_tool(name="weather")
class WeatherTool(ValidatedTool):
    """Get current weather information for a location."""

    class Arguments(ValidatedTool.Arguments):
        location: str
        units: str = "metric"

    class Result(ValidatedTool.Result):
        temperature: float
        conditions: str
        humidity: float
        location: str

    def _execute(self, location: str, units: str) -> dict:  # sync implementation
        """Simulate a weather‑API call (demo only)."""
        time.sleep(0.5)  # pretend network latency
        return {
            "temperature": 22.5,
            "conditions": "Partly Cloudy",
            "humidity": 65.0,
            "location": location,
        }
