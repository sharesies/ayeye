"""
A session is a container for all the information relevant for a given set of completion calls.
"""
import asyncio
import inspect
from typing import Callable, Iterator, overload, AsyncGenerator

from ayeye.helpers import get_all_parts_of_type
from ayeye.providers.base import ProviderBase, AsyncProviderBase
from ayeye.tools import normalize_tool_spec_list
from ayeye.types import (
    Message,
    ToolSpec,
    ToolRequestPart,
    ToolResponsePart,
    Role,
    TextPart,
    Prompt,
)


def normalise_prompt(prompt: Prompt | str, system_prompt: str | None) -> Prompt:
    """
    Normalise the prompt into a structured Prompt object
    """
    if isinstance(prompt, str):
        return Prompt(
            system_prompt=system_prompt,
            messages=[Message(role=Role.USER, parts=[TextPart(text=prompt)])],
        )
    else:
        return prompt
    

class Session:
    model: str  # Model name e.g. "gpt-4o"
    provider: ProviderBase
    max_tool_calls = 10

    def __init__(self, model: str, provider: ProviderBase):
        """
        Initialise our session. The session is useless without a model and an appropriate api key
        :param model: Model string in the form "provider:model", e.g. "openai:gpt-4o"
        """

        self.model = model
        self.provider = provider

    def _complete_with_execute(
        self,
        prompt: Prompt,
        stream: bool,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> Iterator[list[Message]]:
        """
        This method executes an underlying completion or streaming request, then examines the response to see if any
        tool calls are required. If they are, it executes those and appends the relevant messages to the response, then
        re-executes the completion call with the results of the tool calls included.

        Because there's a large volume of common code between the streaming and non-streaming variants, this method
        takes a flag and behaves differently depending on whether `stream` is True or False. In both variants the return
        type is valid—an iterator of lists of messages will be returned.

        For the non-streaming version, this iterator will only ever have one result, and it'll be the final set of
        messages from the completion call.

        For the streaming version, the iterator will return a series of messages progressively as various events occur,
        apps displaying the message results should always use the most recent message list.
        """

        tool_call_counter = 0  # Protection in case the model gets loopy

        # Convert any callables into ToolSpecs
        tool_specs = normalize_tool_spec_list(tools or [])

        # Create a lookup table for the tools, useful for calling
        tools_lookup: dict[str, ToolSpec] = dict(
            (tool.name, tool) for tool in tool_specs
        )

        # List of new messages from this call to completion. Can be many if we have tools to call
        new_messages: list[Message] = []

        # Is our response complete yet or do we have tool calls to handle?
        response_complete = False

        while tool_call_counter < self.max_tool_calls and not response_complete:
            # Assume this one will be the last for now
            response_complete = True

            # Just in case we get nothing from the stream
            response: list[Message] = []

            params = dict(
                model=self.model,
                prompt=Prompt(
                    system_prompt=prompt.system_prompt,
                    messages=prompt.messages + new_messages,
                ),
                tools=tool_specs,
                max_output_tokens=max_output_tokens,
                config=config,
            )

            # Call completion on everything we've got so far
            if stream:
                for response in self.provider.stream(**params):
                    yield new_messages + response
            else:
                response = self.provider.complete(**params)

            # Add all the messages from the response to the new messages list
            new_messages.extend(response)

            # Scan through the response looking for any tools that need to be called
            response_parts: list[ToolResponsePart] = []

            # Group tool requests into sync and async
            sync_requests = []
            async_requests = []

            
            for part in get_all_parts_of_type(response, ToolRequestPart):
                # Check tool name is in the tools list
                if part.name not in tools_lookup:
                    raise ValueError(
                        f"Tool {part.name} requested by model but not found in tools list"
                    )
                
                tool = tools_lookup[part.name]
                if inspect.iscoroutinefunction(tool.fn):
                    async_requests.append((part, tool))
                else:
                    sync_requests.append((part, tool))

            # Handle synchronous tools
            for part, tool in sync_requests:
                tool_result = tool.fn(**part.arguments)
                response_parts.append(
                    ToolResponsePart(id=part.id, name=part.name, result=tool_result)
                )

            # Handle async tools if any exist
            if async_requests:
                import asyncio
                
                async def run_async_tools():
                    tasks = [
                        tool.fn(**part.arguments)
                        for part, tool in async_requests
                    ]
                    results = await asyncio.gather(*tasks)
                    for (part, _), result in zip(async_requests, results):
                        response_parts.append(
                            ToolResponsePart(id=part.id, name=part.name, result=result)
                        )

                asyncio.run(run_async_tools())

            # Got any tool responses to add to the prompt? create a message from the tool
            if response_parts:
                new_messages.append(Message(role=Role.TOOL, parts=response_parts))
                # Make sure we go around again
                response_complete = False

            # Increment the tool call counter
            tool_call_counter += 1

        # Send back all the new messages we've generated this call. Potentially a duplicate of the last iteration
        # of the stream but a no-op is always possible so this is fine.
        yield new_messages

    @overload
    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        ...

    @overload
    def complete(
        self,
        prompt: Prompt,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        ...

    def complete(
        self,
        prompt: Prompt | str,
        system_prompt: str | None = None,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        """
        Complete the given messages and tools.

        A few design decisions have been made here which we might revisit:

        1. No exception handler for tools. If the tool fails it'll throw rather than hand the error back to the model. I
           made this decision because it's easy enough to wrap your tool in a catcher if you want.
        2. No way to add session-related vars to the tool call. The args given to the tool function are only the ones
           provided by the model. The reason here is that you can create a partial function to capture anything you want
           and specifying it here would result in some counter-intuitive issues with overlapping vars (imagine one
           of your tools has a natural arg called `state` _and_ you specify `state` as a session var)
        """
        prompt = normalise_prompt(prompt, system_prompt)

        return next(
            self._complete_with_execute(
                prompt=prompt,
                stream=False,
                tools=tools,
                max_output_tokens=max_output_tokens,
                config=config,
            )
        )

    @overload
    def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> Iterator[list[Message]]:
        ...

    @overload
    def stream(
        self,
        prompt: Prompt,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> Iterator[list[Message]]:
        ...

    def stream(
        self,
        prompt: Prompt | str,
        system_prompt: str | None = None,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> Iterator[list[Message]]:
        """
        Complete the given messages and tools, streaming the model calls. By streaming we mean that the messages
        returned can update as they go.
        """
        prompt = normalise_prompt(prompt, system_prompt)

        return self._complete_with_execute(
            prompt=prompt,
            stream=True,
            tools=tools,
            max_output_tokens=max_output_tokens,
            config=config,
        )


class AsyncSession:
    model: str  # Model name e.g. "gpt-4o"
    provider: AsyncProviderBase
    max_tool_calls = 10

    def __init__(self, model: str, provider: AsyncProviderBase):
        """
        Initialise our session. The session is useless without a model and an appropriate api key
        :param model: Model string in the form "provider:model", e.g. "openai:gpt-4o"
        """

        self.model = model
        self.provider = provider

    async def _complete_with_execute(
        self,
        prompt: Prompt,
        stream: bool,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> AsyncGenerator[list[Message]]:
        """
        This method executes an underlying completion or streaming request, then examines the response to see if any
        tool calls are required. If they are, it executes those and appends the relevant messages to the response, then
        re-executes the completion call with the results of the tool calls included.

        Because there's a large volume of common code between the streaming and non-streaming variants, this method
        takes a flag and behaves differently depending on whether `stream` is True or False. In both variants the return
        type is valid—an iterator of lists of messages will be returned.

        For the non-streaming version, this iterator will only ever have one result, and it'll be the final set of
        messages from the completion call.

        For the streaming version, the iterator will return a series of messages progressively as various events occur,
        apps displaying the message results should always use the most recent message list.
        """

        tool_call_counter = 0  # Protection in case the model gets loopy

        # Convert any callables into ToolSpecs
        tool_specs = normalize_tool_spec_list(tools or [])

        # Create a lookup table for the tools, useful for calling
        tools_lookup: dict[str, ToolSpec] = dict(
            (tool.name, tool) for tool in tool_specs
        )

        # List of new messages from this call to completion. Can be many if we have tools to call
        new_messages: list[Message] = []

        # Is our response complete yet or do we have tool calls to handle?
        response_complete = False

        while tool_call_counter < self.max_tool_calls and not response_complete:
            # Assume this one will be the last for now
            response_complete = True

            # Just in case we get nothing from the stream
            response: list[Message] = []

            params = dict(
                model=self.model,
                prompt=Prompt(
                    system_prompt=prompt.system_prompt,
                    messages=prompt.messages + new_messages,
                ),
                tools=tool_specs,
                max_output_tokens=max_output_tokens,
                config=config,
            )

            # Call completion on everything we've got so far
            if stream:
                async for response in self.provider.stream(**params):
                    yield new_messages + response
            else:
                response = await self.provider.complete(**params)

            # Add all the messages from the response to the new messages list
            new_messages.extend(response)

            # Scan through the response looking for any tools that need to be called
            response_parts: list[ToolResponsePart] = []

            async_calls = []

            for part in get_all_parts_of_type(response, ToolRequestPart):
                # Check tool name is in the tools list
                if part.name not in tools_lookup:
                    raise ValueError(
                        f"Tool {part.name} requested by model but not found in tools list"
                    )

                # We've got a new tool request, call the tool and get the result
                # if the tool is async, we need to await it. Add it to a list so we can gather them all at once
                if asyncio.iscoroutinefunction(tools_lookup[part.name].fn):
                    async_calls.append(tools_lookup[part.name].fn(**part.arguments))
                else:
                    # Yes support for sync here could end up in the caller unintentionally blocking, but it also
                    # simplifies things for stuff that has no meaningful I/O
                    tool_result = tools_lookup[part.name].fn(**part.arguments)
                    # Collect the result into the response parts so we can put them in a message
                    response_parts.append(
                        ToolResponsePart(id=part.id, name=part.name, result=tool_result)
                    )

            if async_calls:
                for tool_result in await asyncio.gather(*async_calls):
                    response_parts.append(
                        ToolResponsePart(id=part.id, name=part.name, result=tool_result)
                    )

            # Got any tool responses to add to the prompt? create a message from the tool
            if response_parts:
                new_messages.append(Message(role=Role.TOOL, parts=response_parts))
                # Make sure we go around again
                response_complete = False

            # Increment the tool call counter
            tool_call_counter += 1

        # Send back all the new messages we've generated this call. Potentially a duplicate of the last iteration
        # of the stream but a no-op is always possible so this is fine.
        yield new_messages

    @overload
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        ...

    @overload
    async def complete(
        self,
        prompt: Prompt,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        ...

    async def complete(
        self,
        prompt: Prompt | str,
        system_prompt: str | None = None,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        """
        Complete the given messages and tools.

        A few design decisions have been made here which we might revisit:

        1. No exception handler for tools. If the tool fails it'll throw rather than hand the error back to the model. I
           made this decision because it's easy enough to wrap your tool in a catcher if you want.
        2. No way to add session-related vars to the tool call. The args given to the tool function are only the ones
           provided by the model. The reason here is that you can create a partial function to capture anything you want
           and specifying it here would result in some counter-intuitive issues with overlapping vars (imagine one
           of your tools has a natural arg called `state` _and_ you specify `state` as a session var)
        """
        prompt = normalise_prompt(prompt, system_prompt)

        async for result in self._complete_with_execute(
            prompt=prompt,
            stream=False,
            tools=tools,
            max_output_tokens=max_output_tokens,
            config=config,
        ):
            # Return the first result (we shouldn't get any more for a non-streaming output)
            return result

        return []

    @overload
    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> AsyncGenerator[list[Message]]:
        ...

    @overload
    async def stream(
        self,
        prompt: Prompt,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> AsyncGenerator[list[Message]]:
        ...

    async def stream(
        self,
        prompt: Prompt | str,
        system_prompt: str | None = None,
        tools: list[ToolSpec | Callable] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> AsyncGenerator[list[Message]]:
        """
        Complete the given messages and tools, streaming the model calls. By streaming we mean that the messages
        returned can update as they go.
        """

        prompt = normalise_prompt(prompt, system_prompt)

        async for result in self._complete_with_execute(
            prompt=prompt,
            stream=True,
            tools=tools,
            max_output_tokens=max_output_tokens,
            config=config,
        ):
            yield result
