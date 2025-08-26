# examples/sample_tools/calculator_tool.py
from typing import Dict

# imports
from chuk_tool_processor.models.validated_tool import ValidatedTool
from chuk_tool_processor.registry.decorators import register_tool


@register_tool(name="calculator")
class CalculatorTool(ValidatedTool):
    """Perform mathematical calculations."""

    class Arguments(ValidatedTool.Arguments):
        operation: str
        x: float
        y: float

    class Result(ValidatedTool.Result):
        result: float
        operation: str

    def _execute(self, operation: str, x: float, y: float) -> Dict:
        """Perform calculations."""
        if operation == "add":
            result = x + y
        elif operation == "subtract":
            result = x - y
        elif operation == "multiply":
            result = x * y
        elif operation == "divide":
            if y == 0:
                raise ValueError("Division by zero")
            result = x / y
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return {"result": result, "operation": operation}
