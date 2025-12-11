"""JSON Schema type constants - no more magic strings!"""

# JSON Schema types
JSON_TYPE_STRING = "string"
JSON_TYPE_NUMBER = "number"
JSON_TYPE_INTEGER = "integer"
JSON_TYPE_BOOLEAN = "boolean"
JSON_TYPE_ARRAY = "array"
JSON_TYPE_OBJECT = "object"
JSON_TYPE_NULL = "null"

# All JSON Schema types
JSON_TYPES = [
    JSON_TYPE_STRING,
    JSON_TYPE_NUMBER,
    JSON_TYPE_INTEGER,
    JSON_TYPE_BOOLEAN,
    JSON_TYPE_ARRAY,
    JSON_TYPE_OBJECT,
    JSON_TYPE_NULL,
]

__all__ = [
    "JSON_TYPE_STRING",
    "JSON_TYPE_NUMBER",
    "JSON_TYPE_INTEGER",
    "JSON_TYPE_BOOLEAN",
    "JSON_TYPE_ARRAY",
    "JSON_TYPE_OBJECT",
    "JSON_TYPE_NULL",
    "JSON_TYPES",
]
