# transport/stdio/stdio_server_parameters.py
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class StdioServerParameters(BaseModel):
    command: str
    args: list[str] = Field(default_factory=list)
    env: Optional[Dict[str, str]] = None