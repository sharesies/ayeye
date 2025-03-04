import base64
from typing import Generator, AsyncGenerator

import anthropic
from anthropic.types import (
    TextBlock,
    ToolUseBlock,
    ThinkingBlock,
    RedactedThinkingBlock,
)

from ayeye.providers.base import ProviderBase, AsyncProviderBase
from ayeye.providers.common import python_type_to_basic_json_type
from ayeye.types import (
    Message,
    Role,
    TextPart,
    ImagePart,
    FilePart,
    ToolRequestPart,
    ToolResponsePart,
    ToolSpec,
    Prompt,
)

ROLE_MAP = {
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.TOOL: "user",  # Anthropic likes their tool responses from "user" role
}


def encode_part(
    part: TextPart | ImagePart | FilePart,
) -> dict:
    """
    Take a part and figure out how to encode it
    """
    match part:
        case TextPart():
            if part.meta:
                if part.meta.get("type") == "anthropic_thinking":
                    return {
                        "type": "thinking",
                        "thinking": part.text,
                        "signature": part.meta["signature"],
                    }
                elif part.meta.get("type") == "anthropic_redacted_thinking":
                    return {
                        "type": "redacted_thinking",
                        "data": part.meta["data"],
                    }
            return {"type": "text", "text": part.text}
        case ImagePart():
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": part.mime_type,
                    "data": base64.b64encode(part.data).decode("utf-8"),
                },
            }
        case FilePart():
            raise NotImplementedError("Haven't implemented Anthropic file parts yet")
        case _:
            raise NotImplementedError(
                f"Anthropic provider does not support part type of {type(part)}"
            )


def encode_prompt(prompt: Prompt) -> list[dict]:
    """
    Encode the messages into an Anthropic prompt structure
    """
    result: list[dict] = []

    for m in prompt.messages:
        tool_request_parts = []
        tool_response_parts = []
        common_parts = []

        for part in m.parts:
            match part:
                case ToolRequestPart():
                    tool_request_parts.append(part)
                case ToolResponsePart():
                    tool_response_parts.append(part)
                case _:
                    common_parts.append(part)

        if tool_request_parts and m.role != Role.ASSISTANT:
            raise ValueError("Only assistants can ask for tools to run")
        if tool_response_parts and m.role != Role.TOOL:
            raise ValueError("Only users can response with tool results")

        content = [encode_part(part) for part in common_parts]

        for trp in tool_request_parts:
            content.append(
                {
                    "type": "tool_use",
                    "id": trp.id,
                    "name": trp.name,
                    "input": trp.arguments,
                }
            )

        for tr in tool_response_parts:
            content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tr.id,
                    "content": str(tr.result),
                }
            )

        if content:
            message = {"role": ROLE_MAP[m.role], "content": content}
            result.append(message)

    return result


def decode_response(response: anthropic.types.Message) -> list[Message]:
    parts = []

    for p in response.content:
        match p:
            case TextBlock():
                parts.append(TextPart(text=p.text))
            case ToolUseBlock():
                parts.append(
                    ToolRequestPart(
                        id=p.id,
                        name=p.name,
                        arguments=p.input,
                    )
                )
            case ThinkingBlock():
                parts.append(
                    TextPart(
                        text=p.thinking,
                        meta={"type": "anthropic_thinking", "signature": p.signature},
                    )
                )
            case RedactedThinkingBlock():
                parts.append(
                    TextPart(
                        text="(REDACTED)",
                        meta={"type": "anthropic_redacted_thinking", "data": p.data},
                    )
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported anthropic response content type {p}"
                )

    return [Message(role=Role.ASSISTANT, parts=parts)]


def build_parameters(
    model: str,
    prompt: Prompt,
    tools: list[ToolSpec] | None = None,
    max_output_tokens: int | None = None,
    config: dict | None = None,
):
    parameters = dict(model=model, messages=encode_prompt(prompt), **(config or {}))

    if prompt.system_prompt:
        parameters["system"] = prompt.system_prompt

    if tools:
        parameters["tools"] = [encode_tool_spec(tool) for tool in tools]

    if max_output_tokens:
        parameters["max_tokens"] = max_output_tokens
    else:
        raise ValueError("Anthropic requires max_output_tokens to be set")

    return parameters


def encode_tool_spec(tool_spec: ToolSpec):
    return {
        "name": tool_spec.name,
        "description": tool_spec.prompt,
        "input_schema": {
            "type": "object",
            "properties": {
                arg.name: {"type": python_type_to_basic_json_type(arg.type)}
                for arg in tool_spec.arguments
            },
            "required": [arg.name for arg in tool_spec.arguments if arg.required],
        },
    }


class Provider(ProviderBase):
    client: anthropic.Anthropic

    def __init__(self, api_key: str, base_url: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def complete(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        parameters = build_parameters(
            model=model,
            prompt=prompt,
            tools=tools,
            max_output_tokens=max_output_tokens,
            config=config,
        )

        response = self.client.messages.create(**parameters)
        return decode_response(response)

    def stream(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> Generator[list[Message]]:
        parameters = build_parameters(
            model=model,
            prompt=prompt,
            tools=tools,
            max_output_tokens=max_output_tokens,
            config=config,
        )

        with self.client.messages.stream(**parameters) as ccs:
            for event in ccs:
                if event.type in ["content_block_delta", "content_block_stop"]:
                    yield decode_response(ccs.current_message_snapshot)

            yield decode_response(ccs.get_final_message())


class AsyncProvider(AsyncProviderBase):
    client: anthropic.AsyncAnthropic

    def __init__(self, api_key: str, base_url: str | None = None):
        self.client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        parameters = build_parameters(
            model=model,
            prompt=prompt,
            tools=tools,
            max_output_tokens=max_output_tokens,
            config=config,
        )

        response = await self.client.messages.create(**parameters)
        return decode_response(response)

    async def stream(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> AsyncGenerator[list[Message]]:
        parameters = build_parameters(
            model=model,
            prompt=prompt,
            tools=tools,
            max_output_tokens=max_output_tokens,
            config=config,
        )

        async with self.client.messages.stream(**parameters) as ccs:
            async for event in ccs:
                if event.type in ["content_block_delta", "content_block_stop"]:
                    yield decode_response(ccs.current_message_snapshot)

            yield decode_response(await ccs.get_final_message())
