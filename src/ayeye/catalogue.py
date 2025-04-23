from dataclasses import dataclass


@dataclass
class Item:
    provider: str
    id: str
    streamable: bool
    reasoning_effort: bool
    function_calling: bool
    context_window: int
    max_output_tokens: int


models = [
    # Anthropic
    Item(
        provider="anthropic",
        id="claude-3-7-sonnet-latest",
        streamable=True,
        reasoning_effort=True,
        function_calling=True,
        context_window=200_000,
        max_output_tokens=64_000,
    ),
    Item(
        provider="anthropic",
        id="claude-3-5-sonnet-latest",
        streamable=True,
        reasoning_effort=False,
        function_calling=True,
        context_window=200_000,
        max_output_tokens=8_192,
    ),
    Item(
        provider="anthropic",
        id="claude-3-5-haiku-latest",
        streamable=True,
        reasoning_effort=False,
        function_calling=True,
        context_window=200_000,
        max_output_tokens=8_192,
    ),
    # Google
    Item(
        provider="google",
        id="gemini-2.0-flash",
        streamable=True,
        reasoning_effort=False,
        function_calling=True,
        context_window=1_048_576,
        max_output_tokens=8_192,
    ),
    Item(
        provider="google",
        id="gemini-2.0-flash-thinking-exp-01-21",
        streamable=True,
        reasoning_effort=False,
        function_calling=False,
        context_window=1_048_576,
        max_output_tokens=8_192,
    ),
    Item(
        provider="google",
        id="gemini-2.0-pro-exp-02-05",
        streamable=True,
        reasoning_effort=False,
        function_calling=True,
        context_window=2_097_152,
        max_output_tokens=1048576,
    ),
    Item(
        provider="google",
        id="gemini-2.5-pro-exp-03-25",
        streamable=True,
        reasoning_effort=False,
        function_calling=True,
        context_window=2_097_152,
        max_output_tokens=1048576,
    ),
    Item(
        provider="google",
        id="gemini-2.5-flash-preview-04-17",
        streamable=True,
        reasoning_effort=False,
        function_calling=True,
        context_window=1_048_576,
        max_output_tokens=65_536,
    ),
    # OpenAI
    Item(
        provider="openai",
        id="gpt-4.1",
        streamable=True,
        reasoning_effort=False,
        function_calling=True,
        context_window=1_000_000,
        max_output_tokens=16_384,
    ),
    Item(
        provider="openai",
        id="gpt-4.1-mini",
        streamable=True,
        reasoning_effort=False,
        function_calling=True,
        context_window=1_000_000,
        max_output_tokens=16_384,
    ),
    Item(
        provider="openai",
        id="gpt-4.1-nano",
        streamable=True,
        reasoning_effort=False,
        function_calling=True,
        context_window=1_000_000,
        max_output_tokens=16_384,
    ),
    Item(
        provider="openai",
        id="gpt-4o",
        streamable=True,
        reasoning_effort=False,
        function_calling=True,
        context_window=128_000,
        max_output_tokens=16_384,
    ),
    Item(
        provider="openai",
        id="gpt-4.5-preview",
        streamable=True,
        reasoning_effort=False,
        function_calling=True,
        context_window=128_000,
        max_output_tokens=16_384,
    ),
    Item(
        provider="openai",
        id="gpt-4o-mini",
        streamable=True,
        reasoning_effort=False,
        function_calling=True,
        context_window=128_000,
        max_output_tokens=16_384,
    ),
    Item(
        provider="openai",
        id="o1",
        streamable=True,
        reasoning_effort=True,
        function_calling=True,
        context_window=200_000,
        max_output_tokens=100_000,
    ),
    Item(
        provider="openai",
        id="o1-mini",
        streamable=False,
        reasoning_effort=False,
        function_calling=False,
        context_window=128_000,
        max_output_tokens=65_536,
    ),
    Item(
        provider="openai",
        id="o3-mini",
        streamable=False,
        reasoning_effort=True,
        function_calling=True,
        context_window=200_000,
        max_output_tokens=100_000,
    ),
]


def by_id(id: str) -> Item:
    """
    Obtain a model by id (name)

    This works for either the full id (provider:id) or just the id, e.g. "o3-mini" and "openai:o3-mini" are both valid.
    """
    return next(
        (m for m in models if (m.id == id or m.provider + ":" + m.id == id)), None
    )
