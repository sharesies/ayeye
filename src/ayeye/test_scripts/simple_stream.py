"""
A quick simple prompt call to verify everything is working end to end
"""
import os

from ayeye.providers.google_provider import Provider
from ayeye.session import Session

session = Session("gemini-2.0-flash", Provider(os.environ["GEMINI_API_KEY"]))

# We keep a record of the last value so that we don't re-print the same text, because stream gives us the whole text so
# far, not just the new text.
last_value = ""

for partial in session.stream(
    "Hello! can you tell me the words to 'I'm a little teapot'? as many verses as you know.",
    max_output_tokens=1024,
):
    if partial and partial[0].parts:
        print(partial[0].parts[0].text[len(last_value) :], end="")
        last_value = partial[0].parts[0].text
