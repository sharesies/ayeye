"""
A quick simple prompt call to verify everything is working end to end
"""
import asyncio
import os

from ayeye.session import AsyncSession
from ayeye.providers.google_provider import AsyncProvider as GoogleProvider
from ayeye.providers.openai_provider import AsyncProvider as OpenAIProvider
from ayeye.providers.anthropic_provider import AsyncProvider as AnthropicProvider


# session = Session("openai:gpt-4o", os.environ["OPENAI_API_KEY"])
# session = Session("anthropic:claude-3-5-sonnet-latest", os.environ["ANTHROPIC_API_KEY"])
async def stream_session(session: AsyncSession):
    # We keep a record of the last value so that we don't re-print the same text, because stream gives us the whole text so
    # far, not just the new text.
    last_value = ""
    print(session.model)

    async for partial in session.stream(
        "Hello! can you tell me the words to 'I'm a little teapot'? as many verses as you know.",
        max_output_tokens=1024,
    ):
        if partial and partial[0].parts:
            print(partial[0].parts[0].text[len(last_value) :], end="")
            last_value = partial[0].parts[0].text
    print("\n-----\n")


sessions = [
    AsyncSession(
        "claude-3-5-sonnet-latest",
        AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"]),
    ),
    AsyncSession("gpt-4o", OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])),
    AsyncSession(
        "gemini-2.0-flash-exp", GoogleProvider(api_key=os.environ["GEMINI_API_KEY"])
    ),
]

for session in sessions:
    asyncio.run(stream_session(session))
