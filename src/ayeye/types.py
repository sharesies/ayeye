from dataclasses import dataclass
from enum import Enum
from typing import Type, Callable


class Role(str, Enum):
    """
    The role is the role of a given message. Note that this may play out differently depending on the provider, for
    example TOOL might be sent back as `tool` for OpenAI, but for others it might be `user`. System / Developer prompts
    are sent separately in the prompt.
    """

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolRequestPart:
    """
    When the model requests that a tool be executed on its behalf, it is represented with this part of the message. The
    arguments can be of a wide variety of types and so long as they can be appropriately serialised we're not really
    worried what they are.
    """

    id: str | None  # ID is optional because not all providers use the ID pattern
    name: str
    arguments: dict


@dataclass
class ToolResponsePart:
    """
    When the tool has been executed and a response is available, it is represented with this part of the message. The
    result can be any serializable type really but in practice we mostly expect a string with the occasional dict. The
    result should not really be None, although that'll just get serialised.

    Depending on how the tool is configured, this response _could_ be the result of an error and the result in that
    case is a serialisation of the exception.
    """

    id: str | None
    name: str
    result: dict | str


@dataclass
class TextPart:
    """
    This is just a bit of text. It might be markdown or html or whatever but we don't care at this level.
    """

    text: str
    meta: dict | None = None


@dataclass
class ImagePart:
    """
    This is an image of some kind. We're just storing it as bytes with a name at this level, which is enough to get
    it to and from the provider. More advanced handling should be done elsewhere. We do want to know it's an image
    though as the providers generally care for routing, and the mime type is commonly required too.
    """

    name: str
    mime_type: str
    data: bytes


@dataclass
class FilePart:
    """
    This is a generic file part. It is useful for handing across non-image things like PDFs or other binary info. The
    exact handling is probably fairly provider-dependent. For example, Anthropic can read PDFs directly.
    """

    name: str
    mime_type: str
    data: bytes


@dataclass
class Message:
    """
    Each message represents an item in the current prompt. Each message has a role which is basically where it has come
    from, either the assistant or the user. A special role, system, is used to define things that should control the
    conversation. Typically you can only have one of those and it must be the first message in the prompt. Exactly
    how this gets serialised is provider-dependent.
    """

    role: Role
    parts: list[TextPart | ImagePart | FilePart | ToolRequestPart | ToolResponsePart]

    def all_text(self) -> str:
        """
        Helper to get all the text parts out of a message
        """
        return "".join([p.text for p in self.parts if isinstance(p, TextPart)])


@dataclass
class Prompt:
    """
    A prompt is a list of messages, plus key metadata such as the system prompt
    """

    system_prompt: str | None
    messages: list[Message]

    def add_text(self, text: str, role: Role = Role.USER):
        """
        Simple helper to add a text part since it's such a common operation.
        """
        self.messages.append(Message(role=role, parts=[TextPart(text=text)]))

    def add_image(self, name: str, mime_type: str, data: bytes, role: Role = Role.USER):
        """
        Simple helper to add an image part as a new message. There are cases where it's preferable to add the
        parts to a current message instead of creating a new one.
        """
        self.messages.append(
            Message(
                role=role, parts=[ImagePart(name=name, mime_type=mime_type, data=data)]
            )
        )


@dataclass
class ToolArgumentSpec:
    """
    Info for a tool argument
    """

    name: str
    type: Type
    required: bool


@dataclass
class ToolSpec:
    """
    All the necessary metadata for a tool. These can be built manually, or utility functions should be able to create
    them for functions
    """

    name: str
    prompt: str
    arguments: list[ToolArgumentSpec]
    fn: Callable
