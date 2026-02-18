# tests/apps/test_meta_preservation.py
"""Tests that _meta.ui metadata survives through the tool pipeline."""

from __future__ import annotations


from mcp_cli.tools.models import (
    ToolDefinitionInput,
    ToolInfo,
    ToolMeta,
    ToolUIMeta,
)


class TestToolUIMeta:
    def test_basic(self):
        ui = ToolUIMeta(resourceUri="ui://test/app.html")
        assert ui.resourceUri == "ui://test/app.html"
        assert ui.visibility == ["model", "app"]
        assert ui.csp is None
        assert ui.permissions is None

    def test_with_csp(self):
        ui = ToolUIMeta(
            resourceUri="ui://test/app.html",
            csp={"connectDomains": ["https://api.example.com"]},
        )
        assert ui.csp is not None
        assert "connectDomains" in ui.csp

    def test_allows_extra_fields(self):
        ui = ToolUIMeta(resourceUri="ui://test/app.html", domain="example.com")
        assert ui.domain == "example.com"  # type: ignore[attr-defined]


class TestToolMeta:
    def test_empty(self):
        meta = ToolMeta()
        assert meta.ui is None

    def test_with_ui(self):
        meta = ToolMeta(ui=ToolUIMeta(resourceUri="ui://test/app.html"))
        assert meta.ui is not None
        assert meta.ui.resourceUri == "ui://test/app.html"

    def test_allows_extra_fields(self):
        meta = ToolMeta(custom="value")
        assert meta.custom == "value"  # type: ignore[attr-defined]


class TestToolDefinitionInputMetaCapture:
    """Verify that _meta from raw tool dicts is captured."""

    def test_meta_captured_from_alias(self):
        raw = {
            "name": "get-time",
            "namespace": "my-server",
            "inputSchema": {},
            "_meta": {
                "ui": {
                    "resourceUri": "ui://get-time/app.html",
                }
            },
        }
        parsed = ToolDefinitionInput.model_validate(raw)
        assert parsed.meta is not None
        assert parsed.meta.ui is not None
        assert parsed.meta.ui.resourceUri == "ui://get-time/app.html"

    def test_meta_none_when_absent(self):
        raw = {
            "name": "plain-tool",
            "namespace": "server",
            "inputSchema": {"type": "object", "properties": {}},
        }
        parsed = ToolDefinitionInput.model_validate(raw)
        assert parsed.meta is None

    def test_extra_fields_still_ignored(self):
        raw = {
            "name": "test",
            "namespace": "ns",
            "inputSchema": {},
            "unknown_field": "should be ignored",
            "_meta": {"ui": {"resourceUri": "ui://test/app.html"}},
        }
        parsed = ToolDefinitionInput.model_validate(raw)
        assert not hasattr(parsed, "unknown_field")
        assert parsed.meta is not None

    def test_meta_with_visibility(self):
        raw = {
            "name": "hidden-tool",
            "namespace": "ns",
            "inputSchema": {},
            "_meta": {
                "ui": {
                    "resourceUri": "ui://hidden/app.html",
                    "visibility": ["app"],
                }
            },
        }
        parsed = ToolDefinitionInput.model_validate(raw)
        assert parsed.meta.ui.visibility == ["app"]


class TestToolInfoMetaPropagation:
    """Verify ToolInfo properties for app detection."""

    def test_has_app_ui_true(self):
        tool = ToolInfo(
            name="test",
            namespace="ns",
            meta=ToolMeta(ui=ToolUIMeta(resourceUri="ui://test/app.html")),
        )
        assert tool.has_app_ui is True
        assert tool.app_resource_uri == "ui://test/app.html"

    def test_has_app_ui_false_no_meta(self):
        tool = ToolInfo(name="test", namespace="ns")
        assert tool.has_app_ui is False
        assert tool.app_resource_uri is None

    def test_has_app_ui_false_no_ui(self):
        tool = ToolInfo(name="test", namespace="ns", meta=ToolMeta())
        assert tool.has_app_ui is False
        assert tool.app_resource_uri is None


class TestEndToEndPipeline:
    """Simulate the full _convert_to_tool_info pipeline."""

    def test_meta_survives_conversion(self):
        """Raw tool dict with _meta → ToolDefinitionInput → ToolInfo."""
        raw = {
            "name": "visualize-data",
            "namespace": "analytics",
            "description": "Render interactive charts",
            "inputSchema": {
                "type": "object",
                "properties": {"data": {"type": "string"}},
            },
            "is_async": False,
            "tags": ["visualization"],
            "_meta": {
                "ui": {
                    "resourceUri": "ui://charts/interactive",
                    "csp": {"connectDomains": ["https://cdn.example.com"]},
                }
            },
        }

        # Step 1: Parse to input model (simulates ToolDefinitionInput.model_validate)
        tool_input = ToolDefinitionInput.model_validate(raw)
        assert tool_input.meta is not None
        assert tool_input.meta.ui.resourceUri == "ui://charts/interactive"

        # Step 2: Convert to ToolInfo (simulates _convert_to_tool_info)
        tool_info = ToolInfo(
            name=tool_input.name,
            namespace=tool_input.namespace,
            description=tool_input.description,
            parameters=tool_input.inputSchema,
            is_async=tool_input.is_async,
            tags=tool_input.tags,
            meta=tool_input.meta,
        )
        assert tool_info.has_app_ui is True
        assert tool_info.app_resource_uri == "ui://charts/interactive"
        assert tool_info.meta.ui.csp == {"connectDomains": ["https://cdn.example.com"]}

    def test_namespace_override_preserves_meta(self):
        """Simulates the namespace override path in get_all_tools."""
        original = ToolInfo(
            name="my-tool",
            namespace="default",
            description="desc",
            meta=ToolMeta(ui=ToolUIMeta(resourceUri="ui://my-tool/app.html")),
        )

        # Simulate the namespace override reconstruction
        overridden = ToolInfo(
            name=original.name,
            namespace="actual-server",
            description=original.description,
            parameters=original.parameters,
            is_async=original.is_async,
            tags=original.tags,
            supports_streaming=original.supports_streaming,
            meta=original.meta,
        )

        assert overridden.has_app_ui is True
        assert overridden.app_resource_uri == "ui://my-tool/app.html"
        assert overridden.namespace == "actual-server"
