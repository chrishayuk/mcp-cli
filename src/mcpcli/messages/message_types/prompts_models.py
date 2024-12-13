from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field

# Content Types
class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str  # base64-encoded image data
    mimeType: str

class ResourceData(BaseModel):
    uri: str
    mimeType: str
    text: Optional[str] = None
    blob: Optional[str] = None  # if binary data is included, base64-encoded

class ResourceContent(BaseModel):
    type: Literal["resource"] = "resource"
    resource: ResourceData

# Union of all content types
MessageContent = Union[TextContent, ImageContent, ResourceContent]

class PromptMessage(BaseModel):
    role: str
    content: MessageContent

# Prompt Definition
class Prompt(BaseModel):
    name: str
    description: Optional[str] = None
    arguments: Optional[List[str]] = None

class PromptsGetResult(BaseModel):
    description: Optional[str]
    messages: List[PromptMessage]

class PromptsGetParams(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}
