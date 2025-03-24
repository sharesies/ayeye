"""
A set of general utility functions for tools
"""
import inspect

from ayeye.types import ToolSpec, ToolArgumentSpec

import typing
from typing import get_origin, Callable

# The only types allowed.
ALLOWED_TYPES = {str, int, bool, float, dict, list}


def normalize_type(param_type) -> type:
    """
    Normalize the provided type annotation so that only one of the allowed base types
    is returned. If the type is a generic alias (e.g. list[int]), this function will
    return the base type (e.g. list), ignoring the type arguments.

    Raises:
        ValueError: If the type is not one of the allowed types.
    """
    origin = get_origin(param_type)
    if origin is not None:
        # param_type is a generic alias such as list[...] or dict[...]
        if origin not in ALLOWED_TYPES:
            raise ValueError(
                f"Unsupported type {param_type!r}. Allowed types are: {ALLOWED_TYPES}."
            )
        # Ignore any type arguments and return the base type.
        return origin
    else:
        if param_type in ALLOWED_TYPES:
            return param_type
        else:
            raise ValueError(
                f"Unsupported type {param_type!r}. Allowed types are: {ALLOWED_TYPES}."
            )


def tool_spec_from_function(fn: callable) -> ToolSpec:
    """
    Convert a function to a tool spec. The spec uses:
      - the function name as the tool name,
      - the function docstring as the prompt, and
      - the function signature as the arguments.

    Each parameter's type annotation is normalized so that only one of these types is
    passed to the ToolArgumentSpec: str, int, bool, float, dict, or list.

    For example, if a function defines a parameter as `list[int]`, only the base type `list`
    will be used.

    Raises:
        ValueError: if a parameter has no type annotation or uses an unsupported type.
    """
    # Resolve type hints (this also handles forward references).
    type_hints = typing.get_type_hints(fn)

    arguments = []
    for name, param in inspect.signature(fn).parameters.items():
        # Get the resolved type hint, or fall back to the raw annotation.
        raw_type = type_hints.get(name, param.annotation)
        if raw_type is inspect.Parameter.empty:
            raise ValueError(f"Parameter '{name}' must have a type annotation.")

        normalized_type = normalize_type(raw_type)

        arguments.append(
            ToolArgumentSpec(
                name=name,
                type=normalized_type,  # Only one of the allowed types is used.
                required=(param.default == param.empty),
            )
        )

    return ToolSpec(
        name=fn.__name__,
        fn=fn,
        prompt=fn.__doc__.strip() if fn.__doc__ else "",
        arguments=arguments,
    )


def normalize_tool_spec_list(tools: list[ToolSpec | Callable]) -> list[ToolSpec]:
    """
    Convert a list of functions to a list of tool specs using tool_spec_from_function.
    """
    tool_specs = []
    for t in tools or []:
        if isinstance(t, ToolSpec):
            ts = t
        elif callable(t):
            ts = tool_spec_from_function(t)
        else:
            raise Exception(f"Tool {t} is not a ToolSpec or a callable function")
        tool_specs.append(ts)
    return tool_specs


def filter_args_from_signature(fn: Callable, args_to_remove: list) -> inspect.Signature:
    """ """
    original_signature = inspect.signature(fn)
    filtered_parameters = [
        p
        for name, p in original_signature.parameters.items()
        if name not in args_to_remove
    ]
    return inspect.Signature(
        parameters=filtered_parameters,
        return_annotation=original_signature.return_annotation,
    )
