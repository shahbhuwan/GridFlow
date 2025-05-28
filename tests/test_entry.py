import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture
def mock_sys_argv():
    """Fixture to restore sys.argv after each test."""
    original_argv = sys.argv
    yield
    sys.argv = original_argv

def test_main_no_args(mock_sys_argv):
    """Test main() with no arguments (launches GUI)."""
    sys.argv = ["entry.py"]
    with patch("gui.main.main", new=MagicMock(), create=True) as mock_gui_main:
        with patch("gridflow.__main__.main", new=MagicMock(), create=True) as mock_cli_main:
            from gridflow.entry import main
            main()
            mock_gui_main.assert_called_once()
            mock_cli_main.assert_not_called()

def test_main_with_args(mock_sys_argv):
    """Test main() with arguments (runs CLI)."""
    sys.argv = ["entry.py", "--help"]
    with patch("gui.main.main", new=MagicMock(), create=True) as mock_gui_main:
        with patch("gridflow.__main__.main", new=MagicMock(), create=True) as mock_cli_main:
            from gridflow.entry import main
            main()
            mock_gui_main.assert_not_called()
            mock_cli_main.assert_called_once()

def test_main_empty_args(mock_sys_argv):
    """Test main() with empty sys.argv (edge case, launches GUI)."""
    sys.argv = []
    with patch("gui.main.main", new=MagicMock(), create=True) as mock_gui_main:
        with patch("gridflow.__main__.main", new=MagicMock(), create=True) as mock_cli_main:
            from gridflow.entry import main
            main()
            mock_gui_main.assert_called_once()
            mock_cli_main.assert_not_called()

def test_main_gui_import_error(mock_sys_argv):
    """Test main() when gui.main import fails."""
    sys.argv = ["entry.py"]
    with patch("gui.main.main", side_effect=ImportError("cannot import name 'main' from 'gui'"), create=True):
        with patch("gridflow.__main__.main", new=MagicMock(), create=True) as mock_cli_main:
            with pytest.raises(ImportError):
                from gridflow.entry import main
                main()
            mock_cli_main.assert_not_called()