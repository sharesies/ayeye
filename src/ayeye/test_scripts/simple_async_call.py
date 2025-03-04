"""
A quick simple prompt call to verify everything is working end to end
"""
import os
import asyncio

from ayeye.session import AsyncSession
from ayeye.providers.openai_provider import AsyncProvider

async def get_current_time() -> str:
    """
    Get the current time
    """
    return "The current time is 12:00"

async def main():
    session = AsyncSession(
        "gpt-4o", AsyncProvider(api_key=os.environ["OPENAI_API_KEY"])
    )

    response = await session.complete(
        "Hello! what is the current time?", tools=[get_current_time]
    )
    print(response[-1].parts[0].text)

asyncio.run(main())
