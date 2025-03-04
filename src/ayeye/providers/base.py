# Provider as an abstract base class
from abc import ABC, abstractmethod
from typing import Generator

from ayeye.types import Message, ToolSpec, Prompt


class ProviderBase(ABC):
    @abstractmethod
    def __init__(self, api_key: str, base_url: str | None = None):
        """
        Initialize the Provider with the given API key and URL.
        :param api_key: API key to use.
        :param base_url: API endpoint URL. Largely used by OpenAI-compatible services
        """
        pass

    @abstractmethod
    def complete(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        """
        Perform a Provider completion call with the given prompt. The prompt must be encoded for Provider use.

        Note that while tools can be specified for the models choice, this method will not call the tools, that's
        up to the Session to handle.
        """
        pass

    @abstractmethod
    def stream(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> Generator[list[Message]]:
        """
        Perform a Provider streaming completion call with the given prompt, returning a generator that yields partial
        completions as they are received. The prompt must be encoded for Provider use.
        """
        pass


class AsyncProviderBase(ABC):
    @abstractmethod
    def __init__(self, api_key: str, base_url: str | None = None):
        """
        Initialize the Provider with the given API key and URL.
        :param api_key: API key to use.
        :param base_url: API endpoint URL. Largely used by OpenAI-compatible services
        """
        pass

    @abstractmethod
    async def complete(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> list[Message]:
        """
        Perform a Provider completion call with the given prompt. The prompt must be encoded for Provider use.

        Note that while tools can be specified for the models choice, this method will not call the tools, that's
        up to the Session to handle.
        """
        pass

    @abstractmethod
    async def stream(
        self,
        model: str,
        prompt: Prompt,
        tools: list[ToolSpec] | None = None,
        max_output_tokens: int | None = None,
        config: dict | None = None,
    ) -> Generator[list[Message]]:
        """
        Perform a Provider streaming completion call with the given prompt, returning a generator that yields partial
        completions as they are received. The prompt must be encoded for Provider use.
        """
        pass
