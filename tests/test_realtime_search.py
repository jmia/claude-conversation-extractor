"""
Tests for realtime_search.py.

We test the pure state-management logic (SearchState, handle_input,
trigger_search, create_smart_searcher) which has no terminal I/O dependency.

KeyboardHandler and TerminalDisplay require a real TTY and are intentionally
not covered here — they are thin wrappers around platform OS calls.
"""

import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from realtime_search import RealTimeSearch, SearchState, create_smart_searcher
from search_conversations import ConversationSearcher, SearchResult
from helpers import write_jsonl, user_entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(file_path: Path, speaker: str = "human", score: float = 0.5) -> SearchResult:
    return SearchResult(
        file_path=file_path,
        conversation_id=file_path.stem,
        matched_content="match",
        context="some context",
        speaker=speaker,
        relevance_score=score,
    )


def make_rts() -> RealTimeSearch:
    """Return a RealTimeSearch with mocked searcher and extractor."""
    searcher = MagicMock()
    searcher.search.return_value = []
    extractor = MagicMock()
    return RealTimeSearch(searcher, extractor)


# ---------------------------------------------------------------------------
# SearchState
# ---------------------------------------------------------------------------

class TestSearchState:

    def test_default_query_is_empty(self):
        state = SearchState()
        assert state.query == ""

    def test_default_results_is_empty_list(self):
        state = SearchState()
        assert state.results == []

    def test_default_cursor_pos_is_zero(self):
        state = SearchState()
        assert state.cursor_pos == 0

    def test_default_selected_index_is_zero(self):
        state = SearchState()
        assert state.selected_index == 0

    def test_is_searching_defaults_to_false(self):
        state = SearchState()
        assert state.is_searching is False


# ---------------------------------------------------------------------------
# handle_input — character typing
# ---------------------------------------------------------------------------

class TestHandleInputTyping:

    def test_printable_character_appended_to_query(self, tmp_path):
        rts = make_rts()
        rts.handle_input("h")
        rts.handle_input("i")
        assert rts.state.query == "hi"

    def test_cursor_advances_with_each_character(self, tmp_path):
        rts = make_rts()
        rts.handle_input("a")
        rts.handle_input("b")
        assert rts.state.cursor_pos == 2

    def test_backspace_removes_last_character(self, tmp_path):
        rts = make_rts()
        rts.handle_input("h")
        rts.handle_input("i")
        rts.handle_input("BACKSPACE")
        assert rts.state.query == "h"
        assert rts.state.cursor_pos == 1

    def test_backspace_on_empty_query_does_nothing(self, tmp_path):
        rts = make_rts()
        rts.handle_input("BACKSPACE")
        assert rts.state.query == ""
        assert rts.state.cursor_pos == 0

    def test_typing_returns_redraw_action(self, tmp_path):
        rts = make_rts()
        action = rts.handle_input("x")
        assert action == "redraw"


# ---------------------------------------------------------------------------
# handle_input — navigation
# ---------------------------------------------------------------------------

class TestHandleInputNavigation:

    def test_esc_returns_exit_action(self, tmp_path):
        rts = make_rts()
        assert rts.handle_input("ESC") == "exit"

    def test_enter_with_no_results_returns_none(self, tmp_path):
        rts = make_rts()
        assert rts.handle_input("ENTER") is None

    def test_enter_with_results_returns_select(self, tmp_path):
        rts = make_rts()
        rts.state.results = [make_result(tmp_path / "a.jsonl")]
        rts.state.selected_index = 0
        assert rts.handle_input("ENTER") == "select"

    def test_down_increments_selected_index(self, tmp_path):
        rts = make_rts()
        rts.state.results = [
            make_result(tmp_path / "a.jsonl"),
            make_result(tmp_path / "b.jsonl"),
        ]
        rts.state.selected_index = 0
        rts.handle_input("DOWN")
        assert rts.state.selected_index == 1

    def test_down_does_not_exceed_result_count(self, tmp_path):
        rts = make_rts()
        rts.state.results = [make_result(tmp_path / "a.jsonl")]
        rts.state.selected_index = 0
        rts.handle_input("DOWN")
        assert rts.state.selected_index == 0

    def test_up_decrements_selected_index(self, tmp_path):
        rts = make_rts()
        rts.state.results = [
            make_result(tmp_path / "a.jsonl"),
            make_result(tmp_path / "b.jsonl"),
        ]
        rts.state.selected_index = 1
        rts.handle_input("UP")
        assert rts.state.selected_index == 0

    def test_up_does_not_go_below_zero(self, tmp_path):
        rts = make_rts()
        rts.state.results = [make_result(tmp_path / "a.jsonl")]
        rts.state.selected_index = 0
        rts.handle_input("UP")
        assert rts.state.selected_index == 0

    def test_left_decrements_cursor(self, tmp_path):
        rts = make_rts()
        rts.handle_input("a")
        rts.handle_input("b")
        rts.handle_input("LEFT")
        assert rts.state.cursor_pos == 1

    def test_right_increments_cursor(self, tmp_path):
        rts = make_rts()
        rts.handle_input("a")
        rts.handle_input("b")
        rts.handle_input("LEFT")
        rts.handle_input("LEFT")
        rts.handle_input("RIGHT")
        assert rts.state.cursor_pos == 1

    def test_none_key_returns_none(self, tmp_path):
        rts = make_rts()
        assert rts.handle_input(None) is None


# ---------------------------------------------------------------------------
# trigger_search
# ---------------------------------------------------------------------------

class TestTriggerSearch:

    def test_sets_is_searching_flag(self, tmp_path):
        rts = make_rts()
        rts.trigger_search()
        assert rts.state.is_searching is True

    def test_updates_last_update_timestamp(self, tmp_path):
        rts = make_rts()
        before = time.time()
        rts.trigger_search()
        assert rts.state.last_update >= before

    def test_clears_stale_cache_entries(self, tmp_path):
        rts = make_rts()
        rts.results_cache["hello"] = ["some result"]
        rts.results_cache["world"] = ["other result"]
        rts.state.query = "hel"
        rts.trigger_search()
        # "hello" starts with "hel" so stays; "world" does not
        assert "hello" in rts.results_cache
        assert "world" not in rts.results_cache


# ---------------------------------------------------------------------------
# create_smart_searcher
# ---------------------------------------------------------------------------

class TestCreateSmartSearcher:

    def test_returns_the_same_searcher_object(self, tmp_path):
        searcher = ConversationSearcher(cache_dir=tmp_path / ".cache")
        result = create_smart_searcher(searcher)
        assert result is searcher

    def test_search_method_is_replaced(self, tmp_path):
        searcher = ConversationSearcher(cache_dir=tmp_path / ".cache")
        original = searcher.search
        create_smart_searcher(searcher)
        assert searcher.search is not original

    def test_smart_search_returns_list(self, tmp_path):
        searcher = ConversationSearcher(cache_dir=tmp_path / ".cache")
        create_smart_searcher(searcher)
        # Point at empty dir so no files are found — should return []
        results = searcher.search("python", search_dir=tmp_path)
        assert isinstance(results, list)

    def test_smart_search_respects_max_results(self, tmp_path):
        # Write several JSONL files with matching content
        for i in range(5):
            write_jsonl(tmp_path / f"s{i}.jsonl", [user_entry(f"python topic {i}")])
        searcher = ConversationSearcher(cache_dir=tmp_path / ".cache")
        create_smart_searcher(searcher)
        results = searcher.search("python", search_dir=tmp_path, max_results=2)
        assert len(results) <= 2
