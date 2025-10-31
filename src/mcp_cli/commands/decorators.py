# src/mcp_cli/commands/decorators.py
# mypy: disable-error-code="no-any-return,arg-type,return-value,misc,attr-defined,unused-ignore,no-redef"
"""
Decorators for command action functions.

Provides validation, error handling, and standardization for command actions.
"""

from __future__ import annotations

import functools
import inspect
from typing import Callable, TypeVar, ParamSpec

from pydantic import BaseModel, ValidationError
from chuk_term.ui import output

P = ParamSpec("P")
T = TypeVar("T")


def validate_params(model_class: type[BaseModel]):
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

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:  # type: ignore[return]
            try:
                # If first arg is already the model, use it
                if args and isinstance(args[0], model_class):
                    return await func(*args, **kwargs)  # type: ignore[return-value,misc]

                # Otherwise, create model from kwargs
                params = model_class(**kwargs)
                return await func(  # type: ignore[return-value,misc,arg-type]
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
                return None  # type: ignore[return-value]
            except Exception as e:
                output.error(f"Error: {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:  # type: ignore[return]
            try:
                # If first arg is already the model, use it
                if args and isinstance(args[0], model_class):
                    return func(*args, **kwargs)  # type: ignore[return-value]

                # Otherwise, create model from kwargs
                params = model_class(**kwargs)
                return func(  # type: ignore[arg-type]
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
                return None  # type: ignore[return-value]
            except Exception as e:
                output.error(f"Error: {e}")
                raise

        # Return appropriate wrapper based on whether function is async
        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        else:
            return sync_wrapper  # type: ignore[return-value]

    return decorator


def handle_errors(message: str = "Command failed"):
    """
    Decorator to handle common errors in command actions.

    Args:
        message: Error message prefix to display

    Example:
        >>> @handle_errors("Token operation failed")
        >>> async def token_action() -> None:
        >>>     # errors are caught and displayed nicely
        >>>     pass
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:  # type: ignore[return]
            try:
                return await func(*args, **kwargs)  # type: ignore[return-value,misc]
            except Exception as e:
                output.error(f"{message}: {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:  # type: ignore[return]
            try:
                return func(*args, **kwargs)  # type: ignore[return-value]
            except Exception as e:
                output.error(f"{message}: {e}")
                raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        else:
            return sync_wrapper  # type: ignore[return-value]

    return decorator
