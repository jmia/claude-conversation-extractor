"""
Tests for ClaudeConversationExtractor in extract_claude_logs.py.

Design principles:
- Each test covers exactly one behaviour.
- All JSONL input is built inline as dicts — no fixture files, no ~/.claude assumptions.
- tmp_path (pytest built-in) handles all temp directory creation and cleanup.
- No mocking of Path.home() or filesystem structure — output_dir is always passed explicitly.
"""

import json
import pytest
from pathlib import Path

from extract_claude_logs import ClaudeConversationExtractor
from helpers import write_jsonl, user_entry, assistant_entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_jsonl(tmp_path: Path, entries: list, filename: str = "session.jsonl") -> Path:
    return write_jsonl(tmp_path / filename, entries)


def make_extractor(tmp_path: Path) -> ClaudeConversationExtractor:
    """Return an extractor that writes to tmp_path."""
    return ClaudeConversationExtractor(output_dir=tmp_path)


# ---------------------------------------------------------------------------
# _extract_text_content
# ---------------------------------------------------------------------------

class TestExtractTextContent:

    def test_string_content_is_returned_as_is(self, tmp_path):
        extractor = make_extractor(tmp_path)
        assert extractor._extract_text_content("hello world") == "hello world"

    def test_list_of_text_blocks_are_joined(self, tmp_path):
        extractor = make_extractor(tmp_path)
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert extractor._extract_text_content(content) == "first\nsecond"

    def test_non_text_blocks_are_skipped_in_standard_mode(self, tmp_path):
        extractor = make_extractor(tmp_path)
        content = [
            {"type": "text", "text": "visible"},
            {"type": "tool_use", "name": "Read", "input": {}},
        ]
        assert extractor._extract_text_content(content) == "visible"

    def test_tool_use_blocks_included_in_detailed_mode(self, tmp_path):
        extractor = make_extractor(tmp_path)
        content = [
            {"type": "text", "text": "visible"},
            {"type": "tool_use", "name": "Read", "input": {"file": "foo.py"}},
        ]
        result = extractor._extract_text_content(content, detailed=True)
        assert "visible" in result
        assert "Read" in result

    def test_tool_result_rejection_extracts_user_message(self, tmp_path):
        extractor = make_extractor(tmp_path)
        content = [{
            "type": "tool_result",
            "is_error": True,
            "tool_use_id": "toolu_abc",
            "content": (
                "The user doesn't want to proceed with this tool use. "
                "The tool use was rejected. To tell you how to proceed, the user said:\n"
                "I love that your idea of minimum includes a router."
            ),
        }]
        result = extractor._extract_text_content(content)
        assert "_(rejected tool use)_" in result
        assert "I love that your idea of minimum includes a router." in result

    def test_tool_result_non_error_is_skipped(self, tmp_path):
        extractor = make_extractor(tmp_path)
        content = [{
            "type": "tool_result",
            "is_error": False,
            "tool_use_id": "toolu_abc",
            "content": "helm not found\n\nInitialization Summary",
        }]
        assert extractor._extract_text_content(content) == ""

    def test_non_string_non_list_content_is_stringified(self, tmp_path):
        extractor = make_extractor(tmp_path)
        assert extractor._extract_text_content(42) == "42"

    def test_tool_use_in_detailed_mode_uses_plain_format_by_default(self, tmp_path):
        extractor = make_extractor(tmp_path)
        content = [{"type": "tool_use", "name": "Edit", "input": {"old_string": "a\nb", "new_string": "c\nd"}}]
        result = extractor._extract_text_content(content, detailed=True)
        assert "```" not in result
        assert "a\nb" in result

    def test_tool_use_in_detailed_mode_uses_code_fence_when_markdown_true(self, tmp_path):
        extractor = make_extractor(tmp_path)
        content = [{"type": "tool_use", "name": "Edit", "input": {"old_string": "a\nb", "new_string": "c\nd"}}]
        result = extractor._extract_text_content(content, detailed=True, markdown=True)
        assert "```json" in result
        assert "a\nb" in result


# ---------------------------------------------------------------------------
# _format_tool_input
# ---------------------------------------------------------------------------

class TestFormatToolInput:

    def test_plain_format_has_no_code_fence(self, tmp_path):
        extractor = make_extractor(tmp_path)
        result = extractor._format_tool_input("Bash", {"command": "ls -la"})
        assert "```" not in result
        assert "Bash" in result
        assert "ls -la" in result

    def test_markdown_format_wraps_input_in_json_code_fence(self, tmp_path):
        extractor = make_extractor(tmp_path)
        result = extractor._format_tool_input("Bash", {"command": "ls -la"}, markdown=True)
        assert "```json" in result
        assert "ls -la" in result

    def test_markdown_format_wraps_tool_name_in_plain_fence(self, tmp_path):
        extractor = make_extractor(tmp_path)
        result = extractor._format_tool_input("Read", {"file_path": "foo.py"}, markdown=True)
        assert "```\n🔧 Using tool: Read\n```" in result

    def test_multiline_values_are_preserved_as_real_newlines(self, tmp_path):
        extractor = make_extractor(tmp_path)
        result = extractor._format_tool_input("Edit", {"old_string": "line1\nline2"}, markdown=True)
        assert "line1\nline2" in result
        assert r"\n" not in result

    def test_tool_name_appears_in_output(self, tmp_path):
        extractor = make_extractor(tmp_path)
        result = extractor._format_tool_input("Write", {"file_path": "foo.py", "content": "x"})
        assert "Write" in result


# ---------------------------------------------------------------------------
# _extract_project_name
# ---------------------------------------------------------------------------

class TestExtractProjectName:

    def test_returns_last_path_component_of_cwd(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [user_entry("hi", cwd="/home/user/projects/myapp")])
        assert extractor._extract_project_name(jsonl) == "myapp"

    def test_returns_unknown_when_no_cwd_field(self, tmp_path):
        extractor = make_extractor(tmp_path)
        entry = {"type": "user", "message": {"role": "user", "content": "hi"}}
        jsonl = make_jsonl(tmp_path, [entry])
        assert extractor._extract_project_name(jsonl) == "unknown"

    def test_returns_unknown_for_empty_file(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = tmp_path / "empty.jsonl"
        jsonl.write_text("", encoding="utf-8")
        assert extractor._extract_project_name(jsonl) == "unknown"

    def test_skips_malformed_lines_and_reads_valid_ones(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = tmp_path / "mixed.jsonl"
        jsonl.write_text(
            'not valid json\n' + json.dumps({"cwd": "/projects/goodapp"}) + "\n",
            encoding="utf-8",
        )
        assert extractor._extract_project_name(jsonl) == "goodapp"


# ---------------------------------------------------------------------------
# _build_filename
# ---------------------------------------------------------------------------

class TestBuildFilename:

    def test_filename_uses_cc_prefix(self, tmp_path):
        extractor = make_extractor(tmp_path)
        conversation = [{"role": "user", "content": "hi", "timestamp": "2025-05-25T10:30:00Z"}]
        jsonl = make_jsonl(tmp_path, [user_entry("hi")])
        result = extractor._build_filename(conversation, "abcdef1234567890", "md", jsonl)
        assert result.name.startswith("cc-")

    def test_filename_contains_date_and_time_from_first_message(self, tmp_path):
        extractor = make_extractor(tmp_path)
        conversation = [{"role": "user", "content": "hi", "timestamp": "2025-05-25T10:30:00Z"}]
        jsonl = make_jsonl(tmp_path, [user_entry("hi")])
        result = extractor._build_filename(conversation, "abcdef1234567890", "md", jsonl)
        assert "2025-05-25" in result.name
        assert "10-30" in result.name

    def test_filename_uses_first_8_chars_of_session_id(self, tmp_path):
        extractor = make_extractor(tmp_path)
        conversation = [{"role": "user", "content": "hi", "timestamp": "2025-05-25T10:00:00Z"}]
        jsonl = make_jsonl(tmp_path, [user_entry("hi")])
        result = extractor._build_filename(conversation, "abcdef1234567890", "md", jsonl)
        assert "abcdef12" in result.name

    def test_suffix_is_appended_before_extension(self, tmp_path):
        extractor = make_extractor(tmp_path)
        conversation = [{"role": "user", "content": "hi", "timestamp": "2025-05-25T10:00:00Z"}]
        jsonl = make_jsonl(tmp_path, [user_entry("hi")])
        result = extractor._build_filename(conversation, "abcdef1234567890", "md", jsonl, suffix="-detailed")
        assert result.name.endswith("-detailed.md")

    def test_output_is_inside_project_subfolder(self, tmp_path):
        extractor = make_extractor(tmp_path)
        conversation = [{"role": "user", "content": "hi", "timestamp": "2025-05-25T10:00:00Z"}]
        jsonl = make_jsonl(tmp_path, [user_entry("hi", cwd="/projects/myapp")])
        result = extractor._build_filename(conversation, "abcdef1234567890", "md", jsonl)
        assert result.parent.name == "myapp"

    def test_falls_back_to_current_time_when_no_timestamp(self, tmp_path):
        extractor = make_extractor(tmp_path)
        conversation = [{"role": "user", "content": "hi", "timestamp": ""}]
        jsonl = make_jsonl(tmp_path, [user_entry("hi")])
        # Should not raise, and should still produce a valid path
        result = extractor._build_filename(conversation, "abcdef1234567890", "md", jsonl)
        assert result.suffix == ".md"


# ---------------------------------------------------------------------------
# extract_conversation
# ---------------------------------------------------------------------------

class TestExtractConversation:

    def test_returns_empty_list_for_nonexistent_file(self, tmp_path):
        extractor = make_extractor(tmp_path)
        result = extractor.extract_conversation(tmp_path / "ghost.jsonl")
        assert result == []

    def test_extracts_user_and_assistant_messages(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [
            user_entry("Hello Claude"),
            assistant_entry("Hello! How can I help?"),
        ])
        result = extractor.extract_conversation(jsonl)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello Claude"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hello! How can I help?"

    def test_extracts_assistant_content_from_text_block_list(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [
            user_entry("hi"),
            assistant_entry("I can help with that"),
        ])
        result = extractor.extract_conversation(jsonl)
        assert result[1]["content"] == "I can help with that"

    def test_skips_malformed_json_lines(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = tmp_path / "bad.jsonl"
        jsonl.write_text(
            "not json\n" + json.dumps(user_entry("real message")) + "\n" +
            json.dumps(assistant_entry("real response")) + "\n",
            encoding="utf-8",
        )
        result = extractor.extract_conversation(jsonl)
        assert len(result) == 2

    def test_exit_only_session_returns_empty_list(self, tmp_path):
        extractor = make_extractor(tmp_path)
        exit_entry = {
            "type": "user",
            "message": {
                "role": "user",
                "content": "<command-name>/exit</command-name>",
            },
            "timestamp": "2025-05-25T10:00:00Z",
        }
        jsonl = make_jsonl(tmp_path, [exit_entry])
        assert extractor.extract_conversation(jsonl) == []

    def test_exit_session_with_assistant_response_is_not_skipped(self, tmp_path):
        extractor = make_extractor(tmp_path)
        exit_entry = {
            "type": "user",
            "message": {
                "role": "user",
                "content": "<command-name>/exit</command-name>",
            },
            "timestamp": "2025-05-25T10:00:00Z",
        }
        jsonl = make_jsonl(tmp_path, [exit_entry, assistant_entry("Goodbye!")])
        result = extractor.extract_conversation(jsonl)
        assert len(result) > 0

    def test_detailed_mode_includes_tool_use_in_assistant_content(self, tmp_path):
        extractor = make_extractor(tmp_path)
        entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check that."},
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "foo.py"}},
                ],
            },
            "timestamp": "2025-05-25T10:01:00Z",
        }
        jsonl = make_jsonl(tmp_path, [user_entry("hi"), entry])
        result = extractor.extract_conversation(jsonl, detailed=True)
        assistant_msg = next(m for m in result if m["role"] == "assistant")
        assert "Read" in assistant_msg["content"]

    def test_standard_mode_excludes_tool_use_from_assistant_content(self, tmp_path):
        extractor = make_extractor(tmp_path)
        entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here is your answer."},
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "foo.py"}},
                ],
            },
            "timestamp": "2025-05-25T10:01:00Z",
        }
        jsonl = make_jsonl(tmp_path, [user_entry("hi"), entry])
        result = extractor.extract_conversation(jsonl, detailed=False)
        assistant_msg = next(m for m in result if m["role"] == "assistant")
        assert "Read" not in assistant_msg["content"]


# ---------------------------------------------------------------------------
# save_as_markdown
# ---------------------------------------------------------------------------

class TestSaveAsMarkdown:

    def test_returns_none_for_empty_conversation(self, tmp_path):
        extractor = make_extractor(tmp_path)
        assert extractor.save_as_markdown([], "session-abc") is None

    def test_creates_file_with_cc_prefix(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [user_entry("hi"), assistant_entry("hello")])
        conversation = extractor.extract_conversation(jsonl)
        result = extractor.save_as_markdown(conversation, jsonl.stem, jsonl_path=jsonl)
        assert result is not None
        assert result.name.startswith("cc-")

    def test_output_file_has_md_extension(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [user_entry("hi"), assistant_entry("hello")])
        conversation = extractor.extract_conversation(jsonl)
        result = extractor.save_as_markdown(conversation, jsonl.stem, jsonl_path=jsonl)
        assert result.suffix == ".md"

    def test_content_includes_session_id_header(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [user_entry("hi"), assistant_entry("hello")])
        conversation = extractor.extract_conversation(jsonl)
        result = extractor.save_as_markdown(conversation, "my-session-id", jsonl_path=jsonl)
        assert "my-session-id" in result.read_text(encoding="utf-8")

    def test_content_includes_user_and_assistant_messages(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [user_entry("Hello Claude"), assistant_entry("Hello human")])
        conversation = extractor.extract_conversation(jsonl)
        result = extractor.save_as_markdown(conversation, jsonl.stem, jsonl_path=jsonl)
        text = result.read_text(encoding="utf-8")
        assert "Hello Claude" in text
        assert "Hello human" in text

    def test_detailed_suffix_is_reflected_in_filename(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [user_entry("hi"), assistant_entry("hello")])
        conversation = extractor.extract_conversation(jsonl)
        result = extractor.save_as_markdown(conversation, jsonl.stem, jsonl_path=jsonl, suffix="-detailed")
        assert result.name.endswith("-detailed.md")


# ---------------------------------------------------------------------------
# save_as_json
# ---------------------------------------------------------------------------

class TestSaveAsJson:

    def test_returns_none_for_empty_conversation(self, tmp_path):
        extractor = make_extractor(tmp_path)
        assert extractor.save_as_json([], "session-abc") is None

    def test_output_file_has_json_extension(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [user_entry("hi"), assistant_entry("hello")])
        conversation = extractor.extract_conversation(jsonl)
        result = extractor.save_as_json(conversation, jsonl.stem, jsonl_path=jsonl)
        assert result.suffix == ".json"

    def test_output_is_valid_json_with_expected_structure(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [user_entry("hi"), assistant_entry("hello")])
        conversation = extractor.extract_conversation(jsonl)
        result = extractor.save_as_json(conversation, "my-session", jsonl_path=jsonl)
        data = json.loads(result.read_text(encoding="utf-8"))
        assert data["session_id"] == "my-session"
        assert data["message_count"] == 2
        assert len(data["messages"]) == 2

    def test_message_count_matches_actual_messages(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [
            user_entry("one"), assistant_entry("two"),
            user_entry("three"), assistant_entry("four"),
        ])
        conversation = extractor.extract_conversation(jsonl)
        result = extractor.save_as_json(conversation, jsonl.stem, jsonl_path=jsonl)
        data = json.loads(result.read_text(encoding="utf-8"))
        assert data["message_count"] == len(conversation)


# ---------------------------------------------------------------------------
# save_as_html
# ---------------------------------------------------------------------------

class TestSaveAsHtml:

    def test_returns_none_for_empty_conversation(self, tmp_path):
        extractor = make_extractor(tmp_path)
        assert extractor.save_as_html([], "session-abc") is None

    def test_output_file_has_html_extension(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [user_entry("hi"), assistant_entry("hello")])
        conversation = extractor.extract_conversation(jsonl)
        result = extractor.save_as_html(conversation, jsonl.stem, jsonl_path=jsonl)
        assert result.suffix == ".html"

    def test_html_content_includes_messages(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [user_entry("Hello Claude"), assistant_entry("Hello human")])
        conversation = extractor.extract_conversation(jsonl)
        result = extractor.save_as_html(conversation, jsonl.stem, jsonl_path=jsonl)
        text = result.read_text(encoding="utf-8")
        assert "Hello Claude" in text
        assert "Hello human" in text

    def test_html_escapes_angle_brackets_in_content(self, tmp_path):
        extractor = make_extractor(tmp_path)
        jsonl = make_jsonl(tmp_path, [
            user_entry("use <b>bold</b> tags"),
            assistant_entry("okay"),
        ])
        conversation = extractor.extract_conversation(jsonl)
        result = extractor.save_as_html(conversation, jsonl.stem, jsonl_path=jsonl)
        text = result.read_text(encoding="utf-8")
        assert "<b>" not in text
        assert "&lt;b&gt;" in text
