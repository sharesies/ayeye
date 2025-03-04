"""
A quick simple prompt call to verify everything is working end to end
"""
import os
import asyncio

from ayeye.session import AsyncSession
from ayeye.providers.openai_provider import AsyncProvider


async def main():
    session = AsyncSession(
        "gpt-4o", AsyncProvider(api_key=os.environ["OPENAI_API_KEY"])
    )

    response = await session.complete("Hello! what is the capital of New Zealand?")
    print(response[0].parts[0].text)


asyncio.run(main())
