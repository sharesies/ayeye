import pytest
from typing import List, Dict
from ayeye.types import ToolArgumentSpec
from ayeye.tools import tool_spec_from_function


def test_function_valid_types():
    def func(a: str, b: int, c: bool, d: float, e: list, f: dict):
        """Test function with basic types."""
        pass

    spec = tool_spec_from_function(func)
    expected_arguments = [
        ToolArgumentSpec(name="a", type=str, required=True),
        ToolArgumentSpec(name="b", type=int, required=True),
        ToolArgumentSpec(name="c", type=bool, required=True),
        ToolArgumentSpec(name="d", type=float, required=True),
        ToolArgumentSpec(name="e", type=list, required=True),
        ToolArgumentSpec(name="f", type=dict, required=True),
    ]
    assert spec.name == "func"
    assert spec.fn is func
    assert spec.prompt == "Test function with basic types."
    assert spec.arguments == expected_arguments


def test_function_generic_types():
    # Using generic annotations; we expect the base types to be used.
    def func(a: List[int], b: Dict[str, int]):
        """Test function with generic types."""
        pass

    spec = tool_spec_from_function(func)
    expected_arguments = [
        ToolArgumentSpec(name="a", type=list, required=True),
        ToolArgumentSpec(name="b", type=dict, required=True),
    ]
    assert spec.arguments == expected_arguments


def test_function_optional_parameter():
    def func(a: str, b: int = 10):
        """Test function with default value."""
        pass

    spec = tool_spec_from_function(func)
    expected_arguments = [
        ToolArgumentSpec(name="a", type=str, required=True),
        ToolArgumentSpec(name="b", type=int, required=False),
    ]
    assert spec.arguments == expected_arguments


def test_function_invalid_type():
    # Using a type that is not allowed (e.g., tuple)
    def func(a: tuple):
        """Test function with unsupported type."""
        pass

    with pytest.raises(ValueError, match="Unsupported type"):
        tool_spec_from_function(func)


def test_function_missing_annotation():
    def func(a):
        """Test function missing type annotation."""
        pass

    with pytest.raises(ValueError, match="must have a type annotation"):
        tool_spec_from_function(func)
