"""
Tests for interactive_ui.py.

InteractiveUI is primarily a user-input-driven loop, so most of run(),
show_sessions_menu(), and get_folder_selection() are not unit-testable
without extensive stdin mocking. We test the parts that are purely
logic-driven: show_progress output, open_folder platform dispatch,
and extract_conversations delegation to the underlying extractor.
"""

import sys
import pytest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from interactive_ui import InteractiveUI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ui(tmp_path: Path) -> InteractiveUI:
    """Return an InteractiveUI with output_dir set to tmp_path."""
    return InteractiveUI(output_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# show_progress
# ---------------------------------------------------------------------------

class TestShowProgress:

    def test_progress_bar_contains_current_and_total(self, tmp_path, capsys):
        ui = make_ui(tmp_path)
        ui.show_progress(3, 10)
        out = capsys.readouterr().out
        assert "3" in out
        assert "10" in out

    def test_full_progress_bar_is_fully_filled(self, tmp_path, capsys):
        ui = make_ui(tmp_path)
        ui.show_progress(10, 10)
        out = capsys.readouterr().out
        # The unfilled character is defined in the source as "░"; if the terminal
        # can't represent it the source itself would be broken, so matching by
        # checking the filled character appears and nothing is left unfilled.
        assert "\u2591" not in out  # ░ — no unfilled segments

    def test_zero_progress_bar_is_fully_empty(self, tmp_path, capsys):
        ui = make_ui(tmp_path)
        ui.show_progress(0, 10)
        out = capsys.readouterr().out
        assert "\u2588" not in out  # █ — no filled segments

    def test_zero_total_does_not_raise(self, tmp_path):
        ui = make_ui(tmp_path)
        ui.show_progress(0, 0)  # Should not raise ZeroDivisionError


# ---------------------------------------------------------------------------
# open_folder — platform dispatch
# ---------------------------------------------------------------------------

class TestOpenFolder:

    def test_macos_calls_open_subprocess(self, tmp_path):
        ui = make_ui(tmp_path)
        with patch("platform.system", return_value="Darwin"):
            with patch("subprocess.run") as mock_run:
                ui.open_folder(tmp_path)
                mock_run.assert_called_once_with(["open", str(tmp_path)])

    def test_linux_calls_xdg_open_subprocess(self, tmp_path):
        ui = make_ui(tmp_path)
        with patch("platform.system", return_value="Linux"):
            with patch("subprocess.run") as mock_run:
                ui.open_folder(tmp_path)
                mock_run.assert_called_once_with(["xdg-open", str(tmp_path)])

    def test_windows_calls_os_startfile(self, tmp_path):
        ui = make_ui(tmp_path)
        with patch("platform.system", return_value="Windows"):
            # os.startfile only exists on Windows; create=True allows patching on other platforms
            with patch("os.startfile", create=True) as mock_startfile:
                ui.open_folder(tmp_path)
                mock_startfile.assert_called_once_with(str(tmp_path))

    def test_exception_in_open_is_silently_swallowed(self, tmp_path):
        ui = make_ui(tmp_path)
        with patch("platform.system", return_value="Darwin"):
            with patch("subprocess.run", side_effect=Exception("no subprocess")):
                ui.open_folder(tmp_path)  # Should not raise


# ---------------------------------------------------------------------------
# extract_conversations — delegation
# ---------------------------------------------------------------------------

class TestExtractConversations:

    def test_delegates_to_extractor_extract_multiple(self, tmp_path):
        ui = make_ui(tmp_path)
        ui.sessions = [tmp_path / "s0.jsonl", tmp_path / "s1.jsonl"]
        ui.extractor.extract_multiple = MagicMock(return_value=(2, 2))
        ui.extract_conversations([0, 1], tmp_path)
        ui.extractor.extract_multiple.assert_called_once_with(ui.sessions, [0, 1])

    def test_returns_success_count_from_extractor(self, tmp_path):
        ui = make_ui(tmp_path)
        ui.sessions = [tmp_path / "s0.jsonl"]
        ui.extractor.extract_multiple = MagicMock(return_value=(1, 1))
        result = ui.extract_conversations([0], tmp_path)
        assert result == 1

    def test_updates_extractor_output_dir(self, tmp_path):
        ui = make_ui(tmp_path)
        ui.sessions = []
        ui.extractor.extract_multiple = MagicMock(return_value=(0, 0))
        new_dir = tmp_path / "newout"
        ui.extract_conversations([], new_dir)
        assert ui.extractor.output_dir == new_dir
