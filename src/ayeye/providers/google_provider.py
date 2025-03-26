from copy import deepcopy
from itertools import zip_longest
from typing import Generator, AsyncGenerator

from google import genai
from google.genai import types as genai_types
from google.genai.types import GenerateContentResponse

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
    Role.ASSISTANT: "model",
    Role.TOOL: "tool",
}


def encode_part(
    part: TextPart | ImagePart | FilePart,
) -> genai_types.Part:
    """
    Take a part and figure out how to encode it
    """
    match part:
        case TextPart():
            return genai_types.Part.from_text(text=part.text)
        case _:
            raise NotImplementedError(
                f"Google GenAI provider does not support part type of {type(part)}"
            )


def accumulate_chunks(
    current: GenerateContentResponse,
    delta: GenerateContentResponse,
) -> GenerateContentResponse:
    """
    Google doesn't offer an accumulator so we have to implement our own. This function takes current as a response
    and applies chunk to it to create a new accumulated result
    """

    result = deepcopy(current)

    # Basics
    if delta.model_version:
        result.model_version = delta.model_version
    if delta.prompt_feedback:
        result.prompt_feedback = delta.prompt_feedback
    if delta.usage_metadata:
        result.usage_metadata = delta.usage_metadata

    # Merge candidate
    if delta.candidates:
        # We can only do one because it's not all that clear how to identify the ones to merge. _probably_ candidate.index
        assert len(delta.candidates) <= 1, "We can accumulate only one candidate"

        if result.candidates:
            for rp, dp in zip_longest(
                result.candidates[0].content.parts, delta.candidates[0].content.parts
            ):
                if rp and dp:
                    if rp.text and dp.text:
                        # We can just add the text on
                        rp.text += dp.text
                    elif rp.function_call and dp.function_call:
                        # For function calls we assume the args are the only things that are "Streaming", the name and
                        # id arrive in a single chunk
                        for key, value in dp.function_call.args.items():
                            rp.function_call.args[key] = (
                                rp.function_call.args.get(key, "") + value
                            )
                    elif rp.text and dp.function_call:
                        # Just had a function call turn up with no additional text
                        rp.function_call = dp.function_call
                    else:
                        raise ValueError(f"Unknown part types to merge: {rp} and {dp}")
                elif dp:
                    # If there's a delta part but no result part, we need to add it
                    result.candidates[0].content.parts.append(dp)
        else:
            result.candidates = delta.candidates

    return result


def encode_prompt(prompt: Prompt) -> list[genai_types.Content]:
    """
    Encode the messages into a Google GenAI prompt structure
    """
    result: list[genai_types.Content] = []

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

        # Some safety asserts to make sure the right roles are sending the right things
        if tool_request_parts and m.role != Role.ASSISTANT:
            raise ValueError("Only assistants can ask for tools to run")
        if tool_response_parts and m.role != Role.TOOL:
            raise ValueError("Only users can response with tool results")

        # Encode the result of the parts as just parts for content list
        content = [encode_part(part) for part in common_parts]

        # Add any function calls
        for trp in tool_request_parts:
            content.append(
                genai_types.Part.from_function_call(name=trp.name, args=trp.arguments)
            )

        # Add any function responses
        for tr in tool_response_parts:
            content.append(
                genai_types.Part.from_function_response(
                    name=tr.name,
                    response=dict(
                        result=tr.result
                    ),  # Google wants a dict always for the response
                )
            )

        if content:
            result.append(genai_types.Content(role=ROLE_MAP[m.role], parts=content))

    return result


def decode_response(response: genai_types.GenerateContentResponse) -> list[Message]:
    """
    Decode the given response object into one or more Messages
    """
    # List of genai parts from the response
    genai_parts = []
    if response.candidates and response.candidates[0].content:
        genai_parts = response.candidates[0].content.parts

    # Collect the parts
    parts = []

    # Google encodes the content as a list of blocks similar to our Parts
    for p in genai_parts:
        if p.text:
            parts.append(TextPart(text=p.text))
        if p.function_call:
            parts.append(
                ToolRequestPart(
                    id=p.function_call.id,
                    name=p.function_call.name,
                    arguments=p.function_call.args,
                )
            )

    if not parts:
        raise ValueError(f"Unknown parts in response: {response}")

    # Return as a single message in a list
    return [
        Message(
            role=Role.ASSISTANT,
            parts=parts,
        )
    ]


def encode_tool_spec(tool_spec: ToolSpec):
    """
    Return a Google GenAI compatible encoding of the ToolSpec
    """
    parameter_properties = dict(
        (
            arg.name,
            genai_types.Schema(
                type=genai_types.Type(python_type_to_basic_json_type(arg.type).upper())
            ),
        )
        for arg in tool_spec.arguments
    )

    # If there are no args, Google would like is to provide None instead of an empty dict
    parameters = None
    if tool_spec.arguments:
        parameters = genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties=parameter_properties,
            required=[arg.name for arg in tool_spec.arguments if arg.required],
        )

    return genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(
                name=tool_spec.name,
                description=tool_spec.prompt,
                parameters=parameters,
            )
        ],
    )


def build_parameters(
    prompt: Prompt,
    tools: list[ToolSpec] | None = None,
    max_output_tokens: int | None = None,
    config: dict | None = None,
):
    config_parameters = dict(**(config or {}))

    if prompt.system_prompt:
        config_parameters["system_instruction"] = prompt.system_prompt

    if tools:
        config_parameters["tools"] = [encode_tool_spec(tool) for tool in tools]

    if max_output_tokens:
        config_parameters["max_output_tokens"] = max_output_tokens

    return config_parameters


class Provider(ProviderBase):
    client: genai.Client

    def __init__(self, api_key: str, base_url: str | None = None):
        self.client = genai.Client(api_key=api_key)

    def complete(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        """
        Perform a Google GenAI completion call with the given prompt.
        """
        config_parameters = build_parameters(
            prompt=prompt,
            tools=tools,
            max_output_tokens=max_output_tokens,
            config=config,
        )

        response = self.client.models.generate_content(
            model=model,
            contents=encode_prompt(prompt),
            config=genai_types.GenerateContentConfig(**config_parameters),
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
        Perform a Google GenAI streaming completion call
        """
        config_parameters = build_parameters(
            prompt=prompt,
            tools=tools,
            max_output_tokens=max_output_tokens,
            config=config,
        )

        accumulated_result: genai_types.GenerateContentResponse = (
            genai_types.GenerateContentResponse()
        )

        for chunk in self.client.models.generate_content_stream(
            model=model,
            contents=encode_prompt(prompt),
            config=genai_types.GenerateContentConfig(**config_parameters),
        ):
            accumulated_result = accumulate_chunks(accumulated_result, chunk)
            if accumulated_result.candidates:  # There are weird cases where Google returns no candidates. It's odd
                yield decode_response(accumulated_result)


class AsyncProvider(AsyncProviderBase):
    client: genai.Client

    def __init__(self, api_key: str, base_url: str | None = None):
        if base_url:
            raise NotImplementedError("Google GenAI does not support base_url setting")
        self.client = genai.Client(api_key=api_key)

    async def complete(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        """
        Perform a Google GenAI completion call with the given prompt.
        """
        config_parameters = build_parameters(
            prompt=prompt,
            tools=tools,
            max_output_tokens=max_output_tokens,
            config=config,
        )

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=encode_prompt(prompt),
            config=genai_types.GenerateContentConfig(**config_parameters),
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
        Perform a Google GenAI streaming completion call
        """
        config_parameters = build_parameters(
            prompt=prompt,
            tools=tools,
            max_output_tokens=max_output_tokens,
            config=config,
        )

        accumulated_result: genai_types.GenerateContentResponse = (
            genai_types.GenerateContentResponse()
        )

        stream = await self.client.aio.models.generate_content_stream(
            model=model,
            contents=encode_prompt(prompt),
            config=genai_types.GenerateContentConfig(**config_parameters),
        )

        async for chunk in stream:
            accumulated_result = accumulate_chunks(accumulated_result, chunk)
            yield decode_response(accumulated_result)
