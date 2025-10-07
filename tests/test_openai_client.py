"""Tests for the OpenAI workflow client helpers."""

import json

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


def test_parse_script_sections_from_nested_payload() -> None:
    data = {
        "script": {
            "items": [
                {
                    "title": "Introduction",
                    "time_range": ["1879", "1905"],
                    "content": ["Early life details", "Formative years."],
                    "citations": [
                        {"text": "biography:vol1"},
                        {"source": "archive:letters"},
                    ],
                },
                {
                    "heading": "Legacy",
                    "summary": {
                        "text": "Long lasting impact on science.",
                    },
                    "references": ["history-journal"],
                },
            ]
        }
    }

    sections = OpenAIWorkflowClient._parse_script_sections(data)

    assert [section.section for section in sections] == [
        "Introduction",
        "Legacy",
    ]
    assert sections[0].timeframe == "1879 - 1905"
    assert (
        sections[0].summary
        == "Early life details Formative years."
    )
    assert sections[0].citations == ["biography:vol1", "archive:letters"]
    assert sections[1].citations == ["history-journal"]


def test_parse_script_sections_from_stringified_payload() -> None:
    payload = {
        "sections": json.dumps(
            {
                "intro": {
                    "section": "Intro",
                    "timeframe": "1905",
                    "summary": "A big breakthrough.",
                    "citations": "science-magazine",
                }
            }
        )
    }

    sections = OpenAIWorkflowClient._parse_script_sections(payload)

    assert len(sections) == 1
    assert sections[0].section == "Intro"
    assert sections[0].timeframe == "1905"
    assert sections[0].citations == ["science-magazine"]


def test_parse_script_sections_from_requirements_payload() -> None:
    payload = {
        "persona": "Zhuge Liang",
        "requirements": {
            "sections": [
                {"introduction": "Zhuge Liang was a famed strategist of Shu Han."},
                {"climax": "He led repeated Northern Expeditions to restore the Han."},
                {"legacy": "His wisdom and loyalty became legendary in Chinese culture."},
            ],
            "citation_format": "short",
            "max_words": 320,
        },
    }

    sections = OpenAIWorkflowClient._parse_script_sections(payload)

    assert [section.section for section in sections] == [
        "introduction",
        "climax",
        "legacy",
    ]
    assert sections[0].summary.startswith("Zhuge Liang was a famed strategist")
