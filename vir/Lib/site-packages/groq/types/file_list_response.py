# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["FileListResponse", "Data"]


class Data(BaseModel):
    id: Optional[str] = None
    """The file identifier, which can be referenced in the API endpoints."""

    bytes: Optional[int] = None
    """The size of the file, in bytes."""

    created_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the file was created."""

    filename: Optional[str] = None
    """The name of the file."""

    object: Optional[Literal["file"]] = None
    """The object type, which is always `file`."""

    purpose: Optional[Literal["batch", "batch_output"]] = None
    """The intended purpose of the file.

    Supported values are `batch`, and `batch_output`.
    """


class FileListResponse(BaseModel):
    data: List[Data]

    object: Literal["list"]
