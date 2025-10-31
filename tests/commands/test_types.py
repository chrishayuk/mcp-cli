"""Tests for command types module."""

from mcp_cli.commands.types import (
    ServerList,
    ResourceList,
    PromptList,
    ToolList,
    __all__,
)


def test_server_list_type_alias():
    """Test that ServerList is properly defined as a List type."""
    # ServerList should be a generic alias
    assert hasattr(ServerList, "__origin__") or str(ServerList).startswith(
        "typing.List"
    )


def test_resource_list_type_alias():
    """Test that ResourceList is properly defined as a List type."""
    assert hasattr(ResourceList, "__origin__") or str(ResourceList).startswith(
        "typing.List"
    )


def test_prompt_list_type_alias():
    """Test that PromptList is properly defined as a List type."""
    assert hasattr(PromptList, "__origin__") or str(PromptList).startswith(
        "typing.List"
    )


def test_tool_list_type_alias():
    """Test that ToolList is properly defined as a List type."""
    assert hasattr(ToolList, "__origin__") or str(ToolList).startswith("typing.List")


def test_all_exports():
    """Test that __all__ contains all expected type aliases."""
    assert "ServerList" in __all__
    assert "ResourceList" in __all__
    assert "PromptList" in __all__
    assert "ToolList" in __all__
    assert len(__all__) == 4


def test_type_aliases_are_lists():
    """Test that all type aliases are List types."""
    # Check that they're all list-based types
    type_aliases = [ServerList, ResourceList, PromptList, ToolList]
    for alias in type_aliases:
        type_str = str(alias)
        assert "List" in type_str or "list" in type_str.lower()


def test_module_imports():
    """Test that the module can be imported and all exports are available."""
    import mcp_cli.commands.types as types_module

    assert hasattr(types_module, "ServerList")
    assert hasattr(types_module, "ResourceList")
    assert hasattr(types_module, "PromptList")
    assert hasattr(types_module, "ToolList")
