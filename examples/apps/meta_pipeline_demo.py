#!/usr/bin/env python
"""
Demonstrates how _meta.ui survives through the tool pipeline.

Shows the data flow:
  raw tool dict (from MCP server) → ToolDefinitionInput → ToolInfo

No servers or API keys needed.

Usage:
    uv run python examples/apps/meta_pipeline_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_cli.tools.models import ToolDefinitionInput, ToolInfo


def main() -> None:
    print()
    print("=" * 60)
    print("  _meta.ui Pipeline Demo")
    print("=" * 60)
    print()

    # Step 1: Simulate raw tool dict from MCP server get_tools()
    raw_tool = {
        "name": "visualize-data",
        "namespace": "analytics",
        "description": "Render an interactive chart from a dataset",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "JSON-encoded data"},
                "chart_type": {"type": "string", "enum": ["bar", "line", "pie"]},
            },
            "required": ["data"],
        },
        "is_async": False,
        "tags": ["visualization"],
        "_meta": {
            "ui": {
                "resourceUri": "ui://charts/interactive",
                "visibility": ["model", "app"],
                "csp": {"connectDomains": ["https://cdn.example.com"]},
            }
        },
    }

    print("  1. Raw tool dict from MCP server:")
    print(f"     _meta present: {'_meta' in raw_tool}")
    print(f"     _meta.ui.resourceUri: {raw_tool['_meta']['ui']['resourceUri']}")
    print()

    # Step 2: Parse through ToolDefinitionInput (alias="_meta")
    tool_input = ToolDefinitionInput.model_validate(raw_tool)

    print("  2. After ToolDefinitionInput.model_validate():")
    print(f"     meta is not None: {tool_input.meta is not None}")
    print(f"     meta.ui.resourceUri: {tool_input.meta.ui.resourceUri}")
    print(f"     meta.ui.visibility: {tool_input.meta.ui.visibility}")
    print(f"     meta.ui.csp: {tool_input.meta.ui.csp}")
    print()

    # Step 3: Convert to ToolInfo (simulates _convert_to_tool_info)
    tool_info = ToolInfo(
        name=tool_input.name,
        namespace=tool_input.namespace,
        description=tool_input.description,
        parameters=tool_input.inputSchema,
        is_async=tool_input.is_async,
        tags=tool_input.tags,
        meta=tool_input.meta,
    )

    print("  3. After conversion to ToolInfo:")
    print(f"     has_app_ui: {tool_info.has_app_ui}")
    print(f"     app_resource_uri: {tool_info.app_resource_uri}")
    print()

    # Step 4: Simulate namespace override (preserves meta)
    overridden = ToolInfo(
        name=tool_info.name,
        namespace="production-analytics",
        description=tool_info.description,
        parameters=tool_info.parameters,
        is_async=tool_info.is_async,
        tags=tool_info.tags,
        supports_streaming=tool_info.supports_streaming,
        meta=tool_info.meta,
    )

    print("  4. After namespace override:")
    print(f"     namespace: {overridden.namespace}")
    print(f"     has_app_ui: {overridden.has_app_ui}")
    print(f"     app_resource_uri: {overridden.app_resource_uri}")
    print()

    # Step 5: Show a tool without _meta
    plain_tool = ToolDefinitionInput.model_validate(
        {
            "name": "get-weather",
            "namespace": "weather",
            "inputSchema": {"type": "object", "properties": {}},
        }
    )
    plain_info = ToolInfo(
        name=plain_tool.name,
        namespace=plain_tool.namespace,
        meta=plain_tool.meta,
    )

    print("  5. Tool without _meta:")
    print(f"     meta: {plain_info.meta}")
    print(f"     has_app_ui: {plain_info.has_app_ui}")
    print(f"     app_resource_uri: {plain_info.app_resource_uri}")
    print()

    print("=" * 60)
    print("  _meta.ui survives the full pipeline.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
