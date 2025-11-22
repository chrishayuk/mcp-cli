# src/mcp_cli/commands/decorators.py
"""
Decorators for command action functions.

Provides validation, error handling, and standardization for command actions.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Awaitable, Callable, TypeVar

from pydantic import BaseModel, ValidationError
from chuk_term.ui import output

# Type variable for the Pydantic model
ModelT = TypeVar("ModelT", bound=BaseModel)


def validate_params(
    model_class: type[ModelT],
) -> Callable[[Callable[[ModelT], Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """
    Decorator to validate function parameters using a Pydantic model.

    Automatically converts kwargs to a Pydantic model instance and validates.
    If validation fails, displays an error message and returns None.

    Args:
        model_class: The Pydantic model class to use for validation

    Example:
        >>> @validate_params(TokenListParams)
        >>> async def token_list_action_async(params: TokenListParams) -> None:
        >>>     # params is already validated
        >>>     pass
    """

    def decorator(
        func: Callable[[ModelT], Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                # If first arg is already the model, use it
                if args and isinstance(args[0], model_class):
                    return await func(*args, **kwargs)

                # Otherwise, create model from kwargs
                params = model_class(**kwargs)
                return await func(
                    params,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in model_class.model_fields
                    },
                )

            except ValidationError as e:
                output.error(f"Invalid parameters: {e}")
                for error in e.errors():
                    field = ".".join(str(loc) for loc in error["loc"])
                    output.error(f"  {field}: {error['msg']}")
                return None
            except Exception as e:
                output.error(f"Error: {e}")
                raise

        # Only return async wrapper since we only support async functions
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            raise TypeError(
                f"validate_params decorator only supports async functions, got {func}"
            )

    return decorator


def handle_errors(
    message: str = "Command failed",
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """
    Decorator to handle common errors in async command actions.

    Args:
        message: Error message prefix to display

    Example:
        >>> @handle_errors("Token operation failed")
        >>> async def token_action() -> None:
        >>>     # errors are caught and displayed nicely
        >>>     pass
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                output.error(f"{message}: {e}")
                raise

        # Only support async functions
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            raise TypeError(
                f"handle_errors decorator only supports async functions, got {func}"
            )

    return decorator
