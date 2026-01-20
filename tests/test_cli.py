# tests/test_cli.py

import sys
import pytest
from unittest.mock import MagicMock, patch
from gridflow.cli import main

@pytest.fixture
def mock_args():
    """Mock standard CLI arguments with safe defaults."""
    with patch('argparse.ArgumentParser.parse_args') as mock_parse:
        args = MagicMock()
        args.config = None
        args.demo = False
        args.is_gui_mode = False
        args.log_dir = '.'
        args.log_level = 'info'
        args.stop_event.is_set.return_value = False
        
        mock_parse.return_value = args
        yield mock_parse

def test_cli_no_args_exits():
    """Ensure running without arguments prints help/usage (handled by argparse)."""
    with patch('sys.argv', ['gridflow']):
        with pytest.raises(SystemExit):
            main()

@patch('gridflow.cli.prism_downloader.main')
def test_cli_dispatch_prism(mock_prism_main, mock_args):
    """Test that 'gridflow prism' calls the prism downloader."""
    mock_args.return_value.command = 'prism'
    main()
    mock_prism_main.assert_called_once()

@patch('gridflow.cli.cmip6_downloader.main')
def test_cli_dispatch_cmip6(mock_cmip6_main, mock_args):
    """Test that 'gridflow cmip6' calls the cmip6 downloader."""
    mock_args.return_value.command = 'cmip6'
    main()
    mock_cmip6_main.assert_called_once()

@patch('gridflow.cli.cmip5_downloader.main')
def test_cli_dispatch_cmip5(mock_cmip5_main, mock_args):
    """Test that 'gridflow cmip5' calls the cmip5 downloader."""
    mock_args.return_value.command = 'cmip5'
    main()
    mock_cmip5_main.assert_called_once()

@patch('gridflow.cli.era5_downloader.main')
def test_cli_dispatch_era5(mock_era5_main, mock_args):
    """Test that 'gridflow era5' calls the era5 downloader."""
    mock_args.return_value.command = 'era5'
    main()
    mock_era5_main.assert_called_once()

@patch('gridflow.cli.dem_downloader.main')
def test_cli_dispatch_dem(mock_dem_main, mock_args):
    """Test that 'gridflow dem' calls the dem downloader."""
    mock_args.return_value.command = 'dem'
    main()
    mock_dem_main.assert_called_once()

@patch('gridflow.cli.crop_netcdf.main')
def test_cli_dispatch_crop(mock_crop_main, mock_args):
    """Test that 'gridflow crop' calls the crop script."""
    mock_args.return_value.command = 'crop'
    main()
    mock_crop_main.assert_called_once()

@patch('gridflow.cli.clip_netcdf.main')
def test_cli_dispatch_clip(mock_clip_main, mock_args):
    """Test that 'gridflow clip' calls the clip script."""
    mock_args.return_value.command = 'clip'
    main()
    mock_clip_main.assert_called_once()

@patch('gridflow.cli.unit_convert.main')
def test_cli_dispatch_convert(mock_convert_main, mock_args):
    """Test that 'gridflow convert' calls the unit converter."""
    mock_args.return_value.command = 'convert'
    main()
    mock_convert_main.assert_called_once()

@patch('gridflow.cli.temporal_aggregate.main')
def test_cli_dispatch_aggregate(mock_agg_main, mock_args):
    """Test that 'gridflow aggregate' calls the aggregator."""
    mock_args.return_value.command = 'aggregate'
    main()
    mock_agg_main.assert_called_once()

@patch('gridflow.cli.catalog_generator.main')
def test_cli_dispatch_catalog(mock_catalog_main, mock_args):
    """Test that 'gridflow catalog' calls the catalog generator."""
    mock_args.return_value.command = 'catalog'
    main()
    mock_catalog_main.assert_called_once()

def test_cli_keyboard_interrupt(mock_args):
    """Test that Ctrl+C is handled gracefully (exit code 130)."""
    mock_args.return_value.command = 'prism'
    with patch('gridflow.cli.prism_downloader.main', side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 130

def test_cli_general_exception(mock_args, capsys):
    """Test that generic exceptions are caught and printed to stderr."""
    mock_args.return_value.command = 'prism'
    with patch('gridflow.cli.prism_downloader.main', side_effect=ValueError("Test Error")):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1

    captured = capsys.readouterr()
    assert "Error: Test Error" in captured.err