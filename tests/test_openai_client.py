"""Tests for the OpenAI workflow client helpers."""

from video_gen.providers.openai_client import OpenAIWorkflowClient


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
