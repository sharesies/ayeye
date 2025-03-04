"""
A quick simple prompt call to verify everything is working end to end
"""
import base64
import os

from ayeye.providers.openai_provider import Provider
from ayeye.session import Session
from ayeye.types import Message, Role, TextPart, ImagePart, Prompt

session = Session("gpt-4o", Provider(os.environ["OPENAI_API_KEY"]))

MOON = base64.b64decode(
    """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"""
)

response = session.complete(
    Prompt(
        system_prompt=None,
        messages=[
            Message(
                role=Role.USER,
                parts=[
                    TextPart(text="Hello! what is this a photo of?"),
                    ImagePart(name="photo.jpg", mime_type="image/jpeg", data=MOON),
                ],
            )
        ],
    )
)
print(response[0].all_text())
