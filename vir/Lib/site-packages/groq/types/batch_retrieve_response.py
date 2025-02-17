# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import builtins
from typing import List, Optional
from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["BatchRetrieveResponse", "Errors", "ErrorsData", "RequestCounts"]


class ErrorsData(BaseModel):
    code: Optional[str] = None
    """An error code identifying the error type."""

    line: Optional[int] = None
    """The line number of the input file where the error occurred, if applicable."""

    message: Optional[str] = None
    """A human-readable message providing more details about the error."""

    param: Optional[str] = None
    """The name of the parameter that caused the error, if applicable."""


class Errors(BaseModel):
    data: Optional[List[ErrorsData]] = None

    object: Optional[str] = None
    """The object type, which is always `list`."""


class RequestCounts(BaseModel):
    completed: int
    """Number of requests that have been completed successfully."""

    failed: int
    """Number of requests that have failed."""

    total: int
    """Total number of requests in the batch."""


class BatchRetrieveResponse(BaseModel):
    id: str

    completion_window: str
    """The time frame within which the batch should be processed."""

    created_at: int
    """The Unix timestamp (in seconds) for when the batch was created."""

    endpoint: str
    """The API endpoint used by the batch."""

    input_file_id: str
    """The ID of the input file for the batch."""

    object: Literal["batch"]
    """The object type, which is always `batch`."""

    status: Literal[
        "validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"
    ]
    """The current status of the batch."""

    cancelled_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch was cancelled."""

    cancelling_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch started cancelling."""

    completed_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch was completed."""

    error_file_id: Optional[str] = None
    """The ID of the file containing the outputs of requests with errors."""

    errors: Optional[Errors] = None

    expired_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch expired."""

    expires_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch will expire."""

    failed_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch failed."""

    finalizing_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch started finalizing."""

    in_progress_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the batch started processing."""

    metadata: Optional[builtins.object] = None
    """Set of key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format.
    """

    output_file_id: Optional[str] = None
    """The ID of the file containing the outputs of successfully executed requests."""

    request_counts: Optional[RequestCounts] = None
    """The request counts for different statuses within the batch."""
