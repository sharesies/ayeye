"""
A quick simple prompt call to verify everything is working end to end
"""
import os

from ayeye.session import Session
from ayeye.providers.openai_provider import Provider as OpenAIProvider

session = Session("gpt-4o", OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"]))


def get_current_time() -> str:
    """
    Get the current time
    """
    return "The current time is 12:00"


response = session.complete(
    "Hello! what is the current time?", tools=[get_current_time]
)
print(response[-1].parts[0].text)
