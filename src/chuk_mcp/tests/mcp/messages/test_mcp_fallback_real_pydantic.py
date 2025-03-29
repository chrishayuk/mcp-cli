def test_mcp_pydantic_base_real_pydantic():
    """
    Test that mcp_pydantic_base uses real Pydantic if available.
    This confirms that we do NOT trigger the fallback logic.
    """
    import sys

    assert "pydantic" in sys.modules, "Pydantic should be installed for this test."

    from chuk_mcp.mcp_client.mcp_pydantic_base import McpPydanticBase, Field, ConfigDict
    import pydantic

    # Define a test model
    class RealPydanticModel(McpPydanticBase):
        x: int = Field(default=123)
        model_config = ConfigDict(extra="forbid")

    # Check the MRO includes pydantic.BaseModel
    assert pydantic.BaseModel in RealPydanticModel.__mro__, (
        "When Pydantic is installed, McpPydanticBase should be pydantic.BaseModel."
    )

    # Check standard Pydantic behavior
    instance = RealPydanticModel()
    assert instance.model_dump() == {"x": 123}
    instance2 = RealPydanticModel.model_validate({"x": 456})
    assert instance2.x == 456
    # And you might check the config dict's effect, etc. if relevant
