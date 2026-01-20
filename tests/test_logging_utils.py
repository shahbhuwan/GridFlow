# tests/test_logging_utils.py

import logging
from unittest.mock import MagicMock, patch

import pytest

from gridflow.utils.logging_utils import MinimalFilter, SuppressWarnErrFilter, setup_logging

# ############################################################################
# Tests for MinimalFilter
# ############################################################################

@pytest.fixture
def minimal_filter():
    """Provides an instance of the MinimalFilter."""
    return MinimalFilter()

def test_minimal_filter_blocks_non_info_levels(minimal_filter):
    """
    Test that the filter BLOCKS all log levels other than INFO.
    Implementation detail: 'if record.levelno != logging.INFO: return False'
    """
    debug_record = logging.LogRecord('name', logging.DEBUG, 'path', 1, 'msg', (), None)
    warning_record = logging.LogRecord('name', logging.WARNING, 'path', 1, 'msg', (), None)
    error_record = logging.LogRecord('name', logging.ERROR, 'path', 1, 'msg', (), None)
    
    assert minimal_filter.filter(debug_record) is False
    assert minimal_filter.filter(warning_record) is False
    assert minimal_filter.filter(error_record) is False

def test_minimal_filter_blocks_general_info_messages(minimal_filter):
    """Test that the filter blocks generic INFO messages not in the whitelist."""
    blocked_record = logging.LogRecord('name', logging.INFO, 'path', 1, 'A generic info message.', (), None)
    assert minimal_filter.filter(blocked_record) is False

@pytest.mark.parametrize("allowed_message", [
    "Progress: 1/10",
    "Node: esgf-data.dkrz.de",
    "Parallel processing started",
    "Completed: 10/10 files processed.",
    "Downloaded file.zip",
    "Found 5 files to process.",
    "Querying node...",
    "Process finished.",
    "Running in demo mode",
    "Example Command:",
    "Stop signal received!",
    "Execution was interrupted",
    "Failed: checksum mismatch",
    "All nodes failed to respond"
])
def test_minimal_filter_allows_whitelisted_messages(minimal_filter, allowed_message):
    """Test that the filter allows specific, whitelisted INFO messages."""
    allowed_record = logging.LogRecord('name', logging.INFO, 'path', 1, allowed_message, (), None)
    assert minimal_filter.filter(allowed_record) is True

# ############################################################################
# Tests for SuppressWarnErrFilter
# ############################################################################

def test_suppress_warn_err_filter():
    """Test that SuppressWarnErrFilter allows levels < WARNING and blocks >= WARNING."""
    f = SuppressWarnErrFilter()
    
    info_rec = logging.LogRecord('name', logging.INFO, 'path', 1, 'msg', (), None)
    debug_rec = logging.LogRecord('name', logging.DEBUG, 'path', 1, 'msg', (), None)
    warn_rec = logging.LogRecord('name', logging.WARNING, 'path', 1, 'msg', (), None)
    err_rec = logging.LogRecord('name', logging.ERROR, 'path', 1, 'msg', (), None)
    crit_rec = logging.LogRecord('name', logging.CRITICAL, 'path', 1, 'msg', (), None)

    assert f.filter(info_rec) is True
    assert f.filter(debug_rec) is True
    assert f.filter(warn_rec) is False
    assert f.filter(err_rec) is False
    assert f.filter(crit_rec) is False

# ############################################################################
# Tests for setup_logging
# ############################################################################

@pytest.fixture(autouse=True)
def reset_logging():
    """Fixture to ensure the logging module is in a clean state for each test."""
    yield
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)


def test_setup_logging_creates_directory_and_file(tmp_path):
    """Test that the function creates the log directory and file."""
    log_dir = tmp_path / "logs"
    setup_logging(str(log_dir), "verbose", "test_prefix")
    
    assert log_dir.is_dir()
    log_files = list(log_dir.glob("test_prefix_*.log"))
    assert len(log_files) == 1

@patch('logging.getLogger')
def test_setup_logging_verbose_level(mock_getLogger, tmp_path):
    """Test the 'verbose' logging configuration."""
    mock_logger = MagicMock()
    mock_getLogger.return_value = mock_logger
    
    setup_logging(str(tmp_path), "verbose")

    mock_logger.setLevel.assert_called_with(logging.DEBUG)

    assert mock_logger.addHandler.call_count == 2
    
    file_handler = mock_logger.addHandler.call_args_list[0][0][0]
    console_handler = mock_logger.addHandler.call_args_list[1][0][0]

    assert file_handler.level == logging.DEBUG

    assert console_handler.level == logging.INFO

    assert not any(isinstance(f, MinimalFilter) for f in console_handler.filters)

@patch('logging.getLogger')
def test_setup_logging_minimal_level(mock_getLogger, tmp_path):
    """Test the 'minimal' logging configuration."""
    mock_logger = MagicMock()
    mock_getLogger.return_value = mock_logger
    
    setup_logging(str(tmp_path), "minimal")

    mock_logger.setLevel.assert_any_call(logging.DEBUG)
    
    file_handler = mock_logger.addHandler.call_args_list[0][0][0]
    console_handler = mock_logger.addHandler.call_args_list[1][0][0]

    assert file_handler.level == logging.DEBUG

    assert console_handler.level == logging.INFO

    filters = console_handler.filters
    assert any(isinstance(f, MinimalFilter) for f in filters)
    assert any(isinstance(f, SuppressWarnErrFilter) for f in filters)

@patch('logging.getLogger')
def test_setup_logging_debug_level(mock_getLogger, tmp_path):
    """Test the 'debug' logging configuration."""
    mock_logger = MagicMock()
    mock_getLogger.return_value = mock_logger
    
    setup_logging(str(tmp_path), "debug")

    mock_logger.setLevel.assert_called_with(logging.DEBUG)
    
    file_handler = mock_logger.addHandler.call_args_list[0][0][0]
    console_handler = mock_logger.addHandler.call_args_list[1][0][0]

    assert file_handler.level == logging.DEBUG
    assert console_handler.level == logging.DEBUG

    formatter = file_handler.formatter
    assert '%(threadName)s' in formatter._fmt

def test_setup_logging_fallback_on_exception(tmp_path, capsys, mocker):
    """Test that logging falls back to basicConfig if setup fails."""
    mocker.patch('pathlib.Path.mkdir', side_effect=OSError("Permission denied"))
    
    setup_logging(str(tmp_path), "verbose")
    
    captured = capsys.readouterr()
    assert "FATAL: Failed to initialize logging" in captured.err