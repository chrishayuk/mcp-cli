"""Tests for command decorators module."""

from unittest.mock import patch
import pytest
from pydantic import BaseModel

from mcp_cli.commands.decorators import validate_params, handle_errors


# Test models
class SimpleParams(BaseModel):
    """Simple test parameters."""

    name: str
    value: int = 10


class ComplexParams(BaseModel):
    """Complex test parameters."""

    required_field: str
    optional_field: str = "default"
    numeric_field: int


# Tests for validate_params decorator
class TestValidateParamsDecorator:
    """Tests for validate_params decorator."""

    def test_validate_params_sync_function_raises_type_error(self):
        """Test validate_params with sync function raises TypeError."""

        with pytest.raises(TypeError, match="only supports async functions"):

            @validate_params(SimpleParams)
            def sync_func(params: SimpleParams) -> str:
                return f"{params.name}:{params.value}"

    @pytest.mark.asyncio
    async def test_validate_params_async_function_with_kwargs(self):
        """Test validate_params with async function using kwargs."""

        @validate_params(SimpleParams)
        async def async_func(params: SimpleParams) -> str:
            return f"{params.name}:{params.value}"

        result = await async_func(name="test", value=40)
        assert result == "test:40"

    @pytest.mark.asyncio
    async def test_validate_params_async_function_with_defaults(self):
        """Test validate_params with async function using default values."""

        @validate_params(SimpleParams)
        async def async_func(params: SimpleParams) -> str:
            return f"{params.name}:{params.value}"

        result = await async_func(name="test")
        assert result == "test:10"

    @pytest.mark.asyncio
    async def test_validate_params_async_function_with_model_instance(self):
        """Test validate_params with async function receiving model instance."""

        @validate_params(SimpleParams)
        async def async_func(params: SimpleParams) -> str:
            return f"{params.name}:{params.value}"

        params = SimpleParams(name="test", value=50)
        result = await async_func(params)
        assert result == "test:50"

    @patch("mcp_cli.commands.decorators.output.error")
    @pytest.mark.asyncio
    async def test_validate_params_async_validation_error(self, mock_error):
        """Test validate_params with async function raising ValidationError."""

        @validate_params(ComplexParams)
        async def async_func(params: ComplexParams) -> str:
            return f"{params.required_field}"

        # Missing required field
        result = await async_func(optional_field="test")

        assert result is None
        assert mock_error.call_count >= 1

    @pytest.mark.asyncio
    async def test_validate_params_async_with_extra_kwargs(self):
        """Test validate_params with async function and extra kwargs."""

        @validate_params(SimpleParams)
        async def async_func(params: SimpleParams, **kwargs) -> str:
            extra = kwargs.get("extra", "none")
            return f"{params.name}:{params.value}:{extra}"

        result = await async_func(name="test", value=20, extra="bonus")
        assert result == "test:20:bonus"

    @patch("mcp_cli.commands.decorators.output.error")
    @pytest.mark.asyncio
    async def test_validate_params_async_generic_exception(self, mock_error):
        """Test validate_params with async function raising generic exception."""

        @validate_params(SimpleParams)
        async def async_func(params: SimpleParams) -> str:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await async_func(name="test")


# Tests for handle_errors decorator
class TestHandleErrorsDecorator:
    """Tests for handle_errors decorator."""

    def test_handle_errors_sync_function_raises_type_error(self):
        """Test handle_errors with sync function raises TypeError."""

        with pytest.raises(TypeError, match="only supports async functions"):

            @handle_errors("Operation failed")
            def sync_func(value: int) -> int:
                return value * 2

    @pytest.mark.asyncio
    async def test_handle_errors_async_function_success(self):
        """Test handle_errors with async function that succeeds."""

        @handle_errors("Operation failed")
        async def async_func(value: int) -> int:
            return value * 2

        result = await async_func(5)
        assert result == 10

    @patch("mcp_cli.commands.decorators.output.error")
    @pytest.mark.asyncio
    async def test_handle_errors_async_function_error(self, mock_error):
        """Test handle_errors with async function that raises error."""

        @handle_errors("Operation failed")
        async def async_func(value: int) -> int:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await async_func(5)

        mock_error.assert_called_once()
        call_args = str(mock_error.call_args)
        assert "Operation failed" in call_args
        assert "Test error" in call_args

    @patch("mcp_cli.commands.decorators.output.error")
    @pytest.mark.asyncio
    async def test_handle_errors_default_message(self, mock_error):
        """Test handle_errors with default error message."""

        @handle_errors()
        async def async_func() -> None:
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            await async_func()

        mock_error.assert_called_once()
        call_args = str(mock_error.call_args)
        assert "Command failed" in call_args

    @patch("mcp_cli.commands.decorators.output.error")
    @pytest.mark.asyncio
    async def test_handle_errors_custom_message(self, mock_error):
        """Test handle_errors with custom error message."""

        @handle_errors("Custom operation failed")
        async def async_func() -> None:
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            await async_func()

        mock_error.assert_called_once()
        call_args = str(mock_error.call_args)
        assert "Custom operation failed" in call_args

    @pytest.mark.asyncio
    async def test_handle_errors_async_return_value(self):
        """Test handle_errors preserves async function return value."""

        @handle_errors("Operation failed")
        async def async_func(x: int, y: int) -> tuple:
            return (x + y, x * y)

        result = await async_func(3, 4)
        assert result == (7, 12)


# Combined decorator tests
class TestCombinedDecorators:
    """Tests for combining multiple decorators."""

    @pytest.mark.asyncio
    async def test_validate_and_handle_errors_async(self):
        """Test combining validate_params and handle_errors on async function."""

        @handle_errors("Combined operation failed")
        @validate_params(SimpleParams)
        async def async_func(params: SimpleParams) -> str:
            return f"{params.name}:{params.value}"

        result = await async_func(name="test", value=100)
        assert result == "test:100"

    @patch("mcp_cli.commands.decorators.output.error")
    @pytest.mark.asyncio
    async def test_combined_decorators_validation_error(self, mock_error):
        """Test combined decorators handle validation errors."""

        @handle_errors("Combined operation failed")
        @validate_params(ComplexParams)
        async def async_func(params: ComplexParams) -> str:
            return f"{params.required_field}"

        result = await async_func(optional_field="test")
        assert result is None

    @patch("mcp_cli.commands.decorators.output.error")
    @pytest.mark.asyncio
    async def test_combined_decorators_runtime_error(self, mock_error):
        """Test combined decorators handle runtime errors."""

        @handle_errors("Combined operation failed")
        @validate_params(SimpleParams)
        async def async_func(params: SimpleParams) -> str:
            raise ValueError("Runtime error")

        with pytest.raises(ValueError, match="Runtime error"):
            await async_func(name="test")

        mock_error.assert_called()
