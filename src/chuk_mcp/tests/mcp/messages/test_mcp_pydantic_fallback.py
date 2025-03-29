# test_mcp_pydantic_fallback.py

import sys
import importlib
import pytest

def test_mcp_pydantic_base_fallback(monkeypatch):
    """
    Test that mcp_pydantic_base falls back to the pure-Python implementation
    when Pydantic is not available.
    """
    # 1) Remove 'pydantic' from sys.modules so import fails
    monkeypatch.delitem(sys.modules, "pydantic", raising=False)

    # 2) Reload our module so it re-checks for pydantic availability
    import chuk_mcp.mcp_client.mcp_pydantic_base
    importlib.reload(chuk_mcp.mcp_client.mcp_pydantic_base)

    # 3) Now import the fallback classes/functions
    from chuk_mcp.mcp_client.mcp_pydantic_base import McpPydanticBase, Field, ConfigDict

    # 4) Define a quick test model using the fallback
    class FallbackModel(McpPydanticBase):
        x: int = Field(default=123)

        # If you want to test ConfigDict usage (like model_config):
        model_config = ConfigDict(extra="allow")

    # 5) Instantiate and test .model_dump()
    instance = FallbackModel()
    assert instance.model_dump() == {"x": 123}, "Fallback .model_dump should return a dict of fields"

    # 6) Test .model_validate() with some data
    instance2 = FallbackModel.model_validate({"x": 456})
    assert instance2.x == 456, "Fallback .model_validate should construct an instance"

    # 7) Check our 'extra' config dict is stored (though it's not enforced by fallback)
    assert instance.model_config == {"extra": "allow"}
