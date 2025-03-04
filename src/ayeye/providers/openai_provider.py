import base64
import json
from typing import Generator, AsyncGenerator

import openai
import openai.types.chat
from openai.lib.streaming.chat import ChatCompletionStream, AsyncChatCompletionStream
from openai._types import NOT_GIVEN

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
    Role.TOOL: "tool",
}


def decode_or_empty_dict(value: str) -> dict:
    """
    Decode a JSON string or return empty dict if it's empty or invalid
    """
    try:
        return json.loads(value) if value else {}
    except json.JSONDecodeError:
        return {}


def encode_part(
    part: TextPart | ImagePart | FilePart,
) -> dict:
    """
    Take a part and figure out how to encode it
    """
    match part:
        case TextPart():
            return {"type": "text", "text": part.text}
        case ImagePart():
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{part.mime_type};base64,{base64.b64encode(part.data).decode('utf-8')}",
                },
            }
        case FilePart():
            """
            OpenAI doesn't handle file uploads as a distinct part. For now we're going to raise on this, but
            it might make sense to hand in as text for files that are text-capable.
            """
            raise NotImplementedError("OpenAI provider does not support files")
        case _:
            raise NotImplementedError(
                f"OpenAI provider does not support part type of {type(part)}"
            )


def encode_prompt(prompt: Prompt, model: str | None = None) -> list[dict]:
    """
    Encode the messages into an OpenAI prompt structure

    OpenAI puts tools into their own role messages, so we have to break the structure up a little bit as we
    work through this. It also means that certain orderings don't work exactly, e.g. if a message contains
    a text part, then a tool result part, then a text part, that'll be represented as an assistant message with two
    text parts followed by a tool role message. In practice this is pretty harmless.
    """

    result: list[dict] = []

    # Create the developer (new "system") prompt message first if one is set
    if prompt.system_prompt:
        # Some older models don't support system messages
        if model not in ["o1-mini"]:
            result.append(
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt.system_prompt,
                        }
                    ],
                }
            )

    for m in prompt.messages:
        # Extract the parts
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

        # For tool requests we create a tool calls structure
        tool_calls = [
            {
                "id": trp.id,
                "type": "function",
                "function": {
                    "name": trp.name,
                    "arguments": json.dumps(trp.arguments),
                },
            }
            for trp in tool_request_parts
        ]

        # Encode the result of the parts as just parts for content list
        content = [encode_part(part) for part in common_parts]

        if tool_calls or content:
            """
            If we have either of these we should append a message from that role with that info. It's worth
            noting that in practice a non-assistant role should never be generated a tool call so we assert for that
            """
            message = {
                "role": ROLE_MAP[m.role],
            }
            if content:
                message["content"] = content
            if tool_calls:
                assert (
                    m.role == Role.ASSISTANT
                ), "Non-assistant role shouldn't generate tool calls"
                message["tool_calls"] = tool_calls
            result.append(message)

        for tr in tool_response_parts:
            """
            Tool responses get their own message in the OpenAI spec
            """
            result.append(
                {"role": "tool", "tool_call_id": tr.id, "content": str(tr.result)}
            )

    return result


def decode_response(response: openai.types.chat.ChatCompletion) -> list[Message]:
    """
    Decode the given response object into one or more Messages
    """
    m = response.choices[0].message

    # Collect the parts
    parts = []

    # Usually there's text content in here, we represent that in a TextPart
    if m.content:
        parts.append(TextPart(text=m.content))

    # The model may be trying to make one or more tool calls, we represent those here.
    for tc in m.tool_calls or []:
        parts.append(
            ToolRequestPart(
                id=tc.id,
                name=tc.function.name,
                arguments=decode_or_empty_dict(tc.function.arguments),
            )
        )

    # Return as a single message in a list
    return [
        Message(
            role=Role.ASSISTANT,
            parts=parts,
        )
    ]


def build_parameters(
    model: str,
    prompt: Prompt,
    tools: list[ToolSpec] | None = None,
    max_output_tokens: int | None = None,
    config: dict | None = None,
):
    parameters = dict(
        model=model,
        messages=encode_prompt(prompt, model),
        **(config or {}),
    )

    # We set these with `if` like this because the OpenAI `create` call defaults to a special type called
    # NOT_GIVEN which is different to None, and sadly isn't exported so we can't easily replicate.
    if tools:
        parameters["tools"] = [encode_tool_spec(ts) for ts in tools]

    if max_output_tokens:
        parameters["max_completion_tokens"] = max_output_tokens

    return parameters


def encode_tool_spec(tool_spec: ToolSpec):
    """
    Return an OpenAI compatible encoding of the ToolSpec
    """
    return {
        "type": "function",
        "function": {
            "name": tool_spec.name,
            "description": tool_spec.prompt,
            "parameters": {
                "type": "object",
                "properties": {
                    arg.name: {"type": python_type_to_basic_json_type(arg.type)}
                    for arg in tool_spec.arguments
                },
                "required": [arg.name for arg in tool_spec.arguments if arg.required],
            },
        },
    }


class Provider(ProviderBase):
    client: openai.OpenAI

    def __init__(self, api_key: str, base_url: str | None = None):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        """
        Perform an OpenAI completion call with the given prompt. The prompt must be encoded for OpenAI use.

        Note that while tools can be specified for the models choice, this method will not call the tools, that's
        up to the Session to handle.
        """

        response = self.client.chat.completions.create(
            **build_parameters(
                model=model,
                prompt=prompt,
                tools=tools,
                max_output_tokens=max_output_tokens,
                config=config,
            ),
        )

        return decode_response(response)

    def stream(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> Generator[list[Message]]:
        """
        Stream the completion of the given prompt. Much like complete except that partial completions are yielded as
        we go.
        """
        parameters = build_parameters(
            model=model,
            prompt=prompt,
            tools=tools,
            max_output_tokens=max_output_tokens,
            config=config,
        )
        parameters["stream"] = True

        # ChatCompletionStream is a utility class provided by OpenAI that handles the stream nicely, manages
        # accumulation and returns a stream of events (not just chunks) that indicate what has changed
        with ChatCompletionStream(
            raw_stream=self.client.chat.completions.create(**parameters),
            response_format=NOT_GIVEN,
            input_tools=parameters.get("tools", NOT_GIVEN),
        ) as ccs:
            for _ in ccs:
                # We don't particular care what the event (_) is, we're just going to yield an updated
                # snapshot of the completion
                #
                # With OpenAI there are some events, notably logprobs and refusal, that don't necessarily
                # result in a changed snapshot
                yield decode_response(ccs.current_completion_snapshot)

            # We end by yielding a parsed version of the final completion
            yield decode_response(ccs.get_final_completion())


class AsyncProvider(AsyncProviderBase):
    client: openai.AsyncOpenAI

    def __init__(self, api_key: str, base_url: str | None = None):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        """
        Perform an OpenAI completion call with the given prompt. The prompt must be encoded for OpenAI use.

        Note that while tools can be specified for the models choice, this method will not call the tools, that's
        up to the Session to handle.
        """

        response = await self.client.chat.completions.create(
            **build_parameters(
                model=model,
                prompt=prompt,
                tools=tools,
                max_output_tokens=max_output_tokens,
                config=config,
            ),
        )

        return decode_response(response)

    async def stream(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> AsyncGenerator[list[Message]]:
        """
        Stream the completion of the given prompt. Much like complete except that partial completions are yielded as
        we go.
        """
        parameters = build_parameters(
            model=model,
            prompt=prompt,
            tools=tools,
            max_output_tokens=max_output_tokens,
            config=config,
        )
        parameters["stream"] = True

        async with AsyncChatCompletionStream(
            raw_stream=await self.client.chat.completions.create(**parameters),
            response_format=NOT_GIVEN,
            input_tools=parameters.get("tools", NOT_GIVEN),
        ) as ccs:
            async for _ in ccs:
                yield decode_response(ccs.current_completion_snapshot)

            yield decode_response(await ccs.get_final_completion())
