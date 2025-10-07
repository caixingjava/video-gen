"""Tests for the OpenAI workflow client helpers."""

from video_gen.providers.openai_client import OpenAIWorkflowClient


class DummyMessage:
    def __init__(self, content):
        self.content = content


def test_extract_message_content_from_structured_parts() -> None:
    message = DummyMessage(
        [
            {"type": "text", "text": "{\n  \"sections\": ["},
            {"type": "text", "text": "{\"section\": \"intro\", \"summary\": \"Hi\"}"},
            {"type": "text", "text": "]\n}"},
        ]
    )

    content = OpenAIWorkflowClient._extract_message_content(message)

    assert content == '{\n  "sections": [{"section": "intro", "summary": "Hi"}]\n}'


class DumpingMessage:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def test_extract_message_content_from_model_dump_payload() -> None:
    message = DumpingMessage({"content": [{"type": "text", "text": "Hello"}]})

    content = OpenAIWorkflowClient._extract_message_content(message)

    assert content == "Hello"


def test_parse_script_sections_from_plain_string() -> None:
    data = {"sections": "Intro paragraph.\n\nSecond paragraph."}

    sections = OpenAIWorkflowClient._parse_script_sections(data)

    assert len(sections) == 2
    assert sections[0].summary == "Intro paragraph."
    assert sections[1].summary == "Second paragraph."


def test_parse_script_sections_from_script_summary() -> None:
    data = {"script": {"summary": "Single summary returned."}}

    sections = OpenAIWorkflowClient._parse_script_sections(data)

    assert len(sections) == 1
    assert sections[0].summary == "Single summary returned."
    assert sections[0].section.startswith("section_")
