"""
Test the session module
"""
import json

import respx
from ayeye import session
from ayeye.providers.openai_provider import Provider, encode_tool_spec
from ayeye.tools import tool_spec_from_function
from ayeye.types import Message, Role, TextPart, Prompt


@respx.mock
def test_basic_complete():
    """
    Test a basic completion
    """

    # This mock is taken from openai-python/main/tests/lib/chat/test_completions:test_parse_nothing, it can no
    # doubt change sometimes. They use a trick where they pull golden snapshots every now and then but we're not going
    # to do that here.
    respx.post().respond(
        json=json.loads(
            '{"id": "chatcmpl-ABfvaueLEMLNYbT8YzpJxsmiQ6HSY", "object": "chat.completion", "created": 1727346142, "model": "gpt-4o-2024-08-06", "choices": [{"index": 0, "message": {"role": "assistant", "content": "I\'m unable to provide real-time weather updates. To get the current weather in San Francisco, I recommend checking a reliable weather website or app like the Weather Channel or a local news station.", "refusal": null}, "logprobs": null, "finish_reason": "stop"}], "usage": {"prompt_tokens": 14, "completion_tokens": 37, "total_tokens": 51, "completion_tokens_details": {"reasoning_tokens": 0}}, "system_fingerprint": "fp_b40fb1c6fb"}'
        )
    )

    s = session.Session("gpt-4o", Provider("API_KEY"))
    completion = s.complete(
        Prompt(
            system_prompt=None,
            messages=[
                Message(
                    role=Role.USER,
                    parts=[TextPart(text="What's the weather like in SF?")],
                )
            ],
        )
    )

    assert (
        completion[0].parts[0].text
        == "I'm unable to provide real-time weather updates. To get the current weather in San Francisco, I recommend checking a reliable weather website or app like the Weather Channel or a local news station."
    )


def test_tools_definition():
    """
    Test that tools are defined correctly
    """
    s = session.Session("gpt-4o", Provider("API_KEY"))

    def get_weather(location: str) -> str:
        """
        Get the weather for a location
        """
        return "Sunny"

    assert encode_tool_spec(tool_spec_from_function(get_weather)) == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
