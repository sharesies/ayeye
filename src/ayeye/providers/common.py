from typing import Type


def python_type_to_basic_json_type(python_type: Type) -> str:
    """
    Convert a Python type to an OpenAI type string
    """
    if python_type == str:
        return "string"
    if python_type == int or python_type == float:
        return "number"
    if python_type == bool:
        return "boolean"
    if python_type == dict:
        return "object"
    if python_type == list:
        return "array"
    raise ValueError(f"Unknown Python type {python_type}")
