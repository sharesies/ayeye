"""
A quick simple prompt call to verify everything is working end to end
"""
import os

from ayeye.session import Session
from ayeye.providers.openai_provider import Provider as OpenAIProvider
from ayeye.providers.anthropic_provider import Provider as AnthropicProvider
from ayeye.providers.google_provider import Provider as GoogleProvider

openai_session = Session("gpt-4o", OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"]))
response = openai_session.complete("Hello! what is the capital of New Zealand?")
print("OpenAI", response[0].parts[0].text)


anthropic_session = Session(
    "claude-3-5-sonnet-latest",
    AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"]),
)

response = anthropic_session.complete(
    "Hello! what is the capital of New Zealand?", max_output_tokens=1024
)
print("Anthropic", response[0].parts[0].text)


google_session = Session(
    "gemini-2.0-flash", GoogleProvider(api_key=os.environ["GEMINI_API_KEY"])
)

response = google_session.complete("Hello! what is the capital of New Zealand?")
print("Google", response[0].parts[0].text)
