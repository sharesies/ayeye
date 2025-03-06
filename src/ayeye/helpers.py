from typing import TypeVar, Type, List
import base64
import json

from ayeye.types import (
    Prompt,
    Message,
    TextPart,
    ImagePart,
    FilePart,
    ToolRequestPart,
    ToolResponsePart,
)


class BytesEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles bytes by encoding them to base64
    """
    def default(self, obj):
        if isinstance(obj, bytes):
            return {
                "__bytes__": True,
                "data": base64.b64encode(obj).decode('ascii')
            }
        return super().default(obj)


def bytes_decoder(obj):
    """
    Custom JSON decoder function that converts encoded bytes back to bytes objects
    """
    if isinstance(obj, dict) and obj.get("__bytes__") is True:
        return base64.b64decode(obj["data"])
    return obj


def prompt_to_dict(prompt: Prompt, skip_thinking=False, skip_binary=False) -> dict:
    """
    Convert a prompt to a structured dictionary suit
    :param prompt: prompt to convert
    :return: JSON representation of the prompt
    """
    result = {"system_prompt": prompt.system_prompt, "messages": []}
    for m in prompt.messages:
        message = {"role": m.role, "parts": []}
        for p in m.parts:
            # Optionally skip thinking parts. This is useful for token counting since if we send the
            # thinking back we still don't get charged for it and it doesn't count towards the context window
            if isinstance(p, TextPart) and skip_thinking and p.meta and "thinking" in p.meta.get("type"):
                continue
            if isinstance(p, ImagePart) and skip_binary:
                continue
            part = {"type": p.__class__.__name__}
            part.update(p.__dict__)
            message["parts"].append(part)
        result["messages"].append(message)

    return result


def prompt_from_dict(data: dict) -> Prompt:
    """
    Convert a dict from prompt_to_dict back into a Prompt object
    """
    messages = []
    for m in data["messages"]:
        parts = []
        for p in m["parts"]:
            part_type = p.pop("type")
            part = globals()[part_type](**p)
            parts.append(part)
        messages.append(Message(role=m["role"], parts=parts))
    return Prompt(system_prompt=data["system_prompt"], messages=messages)


T = TypeVar("T")


def get_all_parts_of_type(messages: list[Message], part_type: Type[T]) -> List[T]:
    """
    Get all parts of a given type from a list of messages
    :param messages: List of messages to search through
    :param part_type: Type of parts to find
    :return: List of parts matching the specified type
    """
    return [p for m in messages for p in m.parts if isinstance(p, part_type)]


def get_all_non_thinking_text(messages: list[Message]) -> str:
    """
    Get all non-thinking text parts from a list of messages
    :param messages: List of messages to search through
    :return: Concatenated string of non-thinking text parts
    """
    return "".join(
        p.text for p in get_all_parts_of_type(messages, TextPart) if not p.meta or "thinking" not in p.meta.get("type")
    )


def get_provider_module(provider_name: str):
    """
    Get the provider module for a given provider name
    :param provider_name: Name of the provider (e.g. 'openai', 'google')
    :return: The provider module
    :raises ImportError: If the provider module cannot be imported
    """
    try:
        return __import__(
            f"ayeye.providers.{provider_name}_provider", fromlist=["Provider"]
        )
    except ImportError as e:
        raise ImportError(
            f"Could not import provider module for '{provider_name}'"
        ) from e
