"""
Tests for ConversationSearcher in search_conversations.py.

Covers the pure logic methods and the search() integration against
real temporary JSONL files — no ~/.claude assumptions.
"""

import os
import time
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from search_conversations import ConversationSearcher, SearchResult
from helpers import write_jsonl, user_entry as user_msg, assistant_entry as assistant_msg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_searcher(tmp_path: Path) -> ConversationSearcher:
    """Return a searcher that uses tmp_path as its cache dir."""
    return ConversationSearcher(cache_dir=tmp_path / ".cache")


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------

class TestSearchResult:

    def test_fields_are_accessible(self):
        result = SearchResult(
            file_path=Path("/tmp/session.jsonl"),
            conversation_id="abc123",
            matched_content="hello",
            context="hello world",
            speaker="human",
            relevance_score=0.8,
        )
        assert result.conversation_id == "abc123"
        assert result.speaker == "human"
        assert result.relevance_score == 0.8

    def test_str_includes_speaker_and_context(self):
        result = SearchResult(
            file_path=Path("/tmp/session.jsonl"),
            conversation_id="abc123",
            matched_content="hello",
            context="hello world",
            speaker="human",
            relevance_score=0.75,
        )
        text = str(result)
        assert "Human" in text
        assert "hello world" in text
        assert "75%" in text


# ---------------------------------------------------------------------------
# _extract_content
# ---------------------------------------------------------------------------

class TestExtractContent:

    def test_extracts_string_from_message_dict(self, tmp_path):
        searcher = make_searcher(tmp_path)
        entry = {"type": "user", "message": {"role": "user", "content": "hello"}}
        assert searcher._extract_content(entry) == "hello"

    def test_extracts_text_from_content_list(self, tmp_path):
        searcher = make_searcher(tmp_path)
        entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            },
        }
        result = searcher._extract_content(entry)
        assert "part one" in result
        assert "part two" in result

    def test_returns_empty_string_for_unknown_entry(self, tmp_path):
        searcher = make_searcher(tmp_path)
        assert searcher._extract_content({"type": "system"}) == ""

    def test_returns_empty_string_for_missing_content(self, tmp_path):
        searcher = make_searcher(tmp_path)
        entry = {"type": "user", "message": {"role": "user"}}
        assert searcher._extract_content(entry) == ""


# ---------------------------------------------------------------------------
# _calculate_relevance
# ---------------------------------------------------------------------------

class TestCalculateRelevance:

    def test_exact_match_scores_higher_than_no_match(self, tmp_path):
        searcher = make_searcher(tmp_path)
        query = "python error"
        tokens = {"python", "error"}
        with_match = searcher._calculate_relevance(
            "how to fix a python error in your code", query, tokens, False
        )
        without_match = searcher._calculate_relevance(
            "the weather in london today", query, tokens, False
        )
        assert with_match > without_match

    def test_score_is_between_zero_and_one(self, tmp_path):
        searcher = make_searcher(tmp_path)
        score = searcher._calculate_relevance(
            "python python python python python", "python", {"python"}, False
        )
        assert 0.0 <= score <= 1.0

    def test_no_match_returns_zero(self, tmp_path):
        searcher = make_searcher(tmp_path)
        score = searcher._calculate_relevance(
            "completely unrelated content", "zephyr", {"zephyr"}, False
        )
        assert score == 0.0

    def test_case_insensitive_matching(self, tmp_path):
        searcher = make_searcher(tmp_path)
        score = searcher._calculate_relevance(
            "Python is great", "python", {"python"}, False
        )
        assert score > 0.0


# ---------------------------------------------------------------------------
# _extract_context
# ---------------------------------------------------------------------------

class TestExtractContext:

    def test_returns_content_around_match(self, tmp_path):
        searcher = make_searcher(tmp_path)
        content = "The quick brown fox jumps over the lazy dog"
        context = searcher._extract_context(content, "fox", False)
        assert "fox" in context.lower()

    def test_highlights_match_with_asterisks(self, tmp_path):
        searcher = make_searcher(tmp_path)
        context = searcher._extract_context("find the keyword here", "keyword", False)
        assert "**" in context

    def test_returns_start_of_content_when_no_match(self, tmp_path):
        searcher = make_searcher(tmp_path)
        content = "This is some long content without the search term"
        context = searcher._extract_context(content, "zzznomatch", False)
        assert content[:20] in context

    def test_adds_ellipsis_for_truncated_context(self, tmp_path):
        searcher = make_searcher(tmp_path)
        long_content = "x " * 200 + "target" + " x" * 200
        context = searcher._extract_context(long_content, "target", False)
        assert "..." in context


# ---------------------------------------------------------------------------
# search() — integration with real JSONL files
# ---------------------------------------------------------------------------

class TestSearch:

    def test_empty_query_returns_no_results(self, tmp_path):
        searcher = make_searcher(tmp_path)
        write_jsonl(tmp_path / "session.jsonl", [user_msg("hello"), assistant_msg("hi")])
        results = searcher.search("", search_dir=tmp_path)
        assert results == []

    def test_finds_matching_content_in_user_message(self, tmp_path):
        searcher = make_searcher(tmp_path)
        write_jsonl(tmp_path / "session.jsonl", [
            user_msg("how do I fix a segfault"),
            assistant_msg("check your pointers"),
        ])
        results = searcher.search("segfault", search_dir=tmp_path)
        assert len(results) > 0

    def test_finds_matching_content_in_assistant_message(self, tmp_path):
        searcher = make_searcher(tmp_path)
        write_jsonl(tmp_path / "session.jsonl", [
            user_msg("help me"),
            assistant_msg("use a dictionary comprehension here"),
        ])
        results = searcher.search("dictionary comprehension", search_dir=tmp_path)
        assert len(results) > 0

    def test_returns_empty_list_when_no_match(self, tmp_path):
        searcher = make_searcher(tmp_path)
        write_jsonl(tmp_path / "session.jsonl", [
            user_msg("tell me about cats"),
            assistant_msg("cats are great pets"),
        ])
        results = searcher.search("quantum physics", search_dir=tmp_path)
        assert results == []

    def test_max_results_limits_output(self, tmp_path):
        searcher = make_searcher(tmp_path)
        # Write 5 separate files all containing the query
        for i in range(5):
            write_jsonl(tmp_path / f"session{i}.jsonl", [
                user_msg(f"python question number {i}"),
                assistant_msg(f"python answer number {i}"),
            ])
        results = searcher.search("python", search_dir=tmp_path, max_results=2)
        assert len(results) <= 2

    def test_speaker_filter_human_excludes_assistant_messages(self, tmp_path):
        searcher = make_searcher(tmp_path)
        write_jsonl(tmp_path / "session.jsonl", [
            user_msg("user mentions python"),
            assistant_msg("assistant also mentions python"),
        ])
        results = searcher.search(
            "python", search_dir=tmp_path, speaker_filter="human"
        )
        assert all(r.speaker == "human" for r in results)

    def test_speaker_filter_assistant_excludes_human_messages(self, tmp_path):
        searcher = make_searcher(tmp_path)
        write_jsonl(tmp_path / "session.jsonl", [
            user_msg("user mentions python"),
            assistant_msg("assistant also mentions python"),
        ])
        results = searcher.search(
            "python", search_dir=tmp_path, speaker_filter="assistant"
        )
        assert all(r.speaker == "assistant" for r in results)

    def test_exact_mode_finds_literal_match(self, tmp_path):
        searcher = make_searcher(tmp_path)
        write_jsonl(tmp_path / "session.jsonl", [
            user_msg("I need help with ValueError handling"),
            assistant_msg("catch it with except ValueError"),
        ])
        results = searcher.search("ValueError", search_dir=tmp_path, mode="exact")
        assert len(results) > 0

    def test_regex_mode_matches_pattern(self, tmp_path):
        searcher = make_searcher(tmp_path)
        write_jsonl(tmp_path / "session.jsonl", [
            user_msg("import os and import sys"),
            assistant_msg("those are standard library modules"),
        ])
        results = searcher.search(r"import\s+\w+", search_dir=tmp_path, mode="regex")
        assert len(results) > 0

    def test_regex_mode_returns_empty_for_invalid_pattern(self, tmp_path):
        searcher = make_searcher(tmp_path)
        write_jsonl(tmp_path / "session.jsonl", [user_msg("hello")])
        results = searcher.search("[invalid(regex", search_dir=tmp_path, mode="regex")
        assert results == []

    def test_results_are_sorted_by_relevance_descending(self, tmp_path):
        searcher = make_searcher(tmp_path)
        # One file mentions "python" once, another mentions it many times
        write_jsonl(tmp_path / "low.jsonl", [user_msg("python"), assistant_msg("ok")])
        write_jsonl(tmp_path / "high.jsonl", [
            user_msg("python python python python"),
            assistant_msg("python python python"),
        ])
        results = searcher.search("python", search_dir=tmp_path)
        assert results[0].relevance_score >= results[-1].relevance_score

    def test_raises_for_nonexistent_search_dir(self, tmp_path):
        searcher = make_searcher(tmp_path)
        with pytest.raises(ValueError):
            searcher.search("anything", search_dir=tmp_path / "does_not_exist")

    def test_date_filter_excludes_old_files(self, tmp_path):
        searcher = make_searcher(tmp_path)
        old_file = write_jsonl(tmp_path / "old.jsonl", [user_msg("python old")])
        # Set mtime to 10 days ago
        old_mtime = time.time() - (10 * 86400)
        os.utime(old_file, (old_mtime, old_mtime))

        future = datetime.now() - timedelta(days=1)
        results = searcher.search("python", search_dir=tmp_path, date_from=future)
        file_names = [r.file_path.name for r in results]
        assert "old.jsonl" not in file_names
