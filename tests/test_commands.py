import pytest
import argparse
import logging
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path
from gridflow.commands import setup_backend_logging, download_command, download_cmip5_command, download_prism_command, crop_command, clip_command, catalog_command, main

@pytest.fixture
def mock_args():
    """Fixture to create a mock argparse.Namespace with default attributes."""
    return argparse.Namespace(
        metadata_dir='./metadata',
        log_level='minimal',
        output_dir=None,
        workers=None,
        demo=False
    )

@pytest.fixture
def mock_logging():
    """Fixture to mock logging setup and capture logs."""
    with patch("gridflow.commands.setup_logging") as mock_setup_logging:
        with patch("logging.error") as mock_error:
            with patch("logging.info") as mock_info:
                yield mock_setup_logging, mock_error, mock_info

@pytest.fixture
def mock_path_mkdir():
    """Fixture to mock Path.mkdir."""
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        yield mock_mkdir

@pytest.fixture
def reset_logger():
    """Fixture to reset logging handlers before each test."""
    logger = logging.getLogger()
    logger.handlers = []
    yield
    logger.handlers = []

def test_setup_backend_logging_success(mock_args, mock_logging, mock_path_mkdir, reset_logger):
    """Test setup_backend_logging with successful log directory creation."""
    mock_args.metadata_dir = "./metadata"
    mock_args.log_level = "debug"
    mock_setup_logging, _, _ = mock_logging
    setup_backend_logging(mock_args, project_prefix="test")
    mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_setup_logging.assert_called_once_with(Path("./metadata"), "debug", prefix="gridflow_test_")

def test_setup_backend_logging_no_metadata_dir(mock_args, mock_logging, mock_path_mkdir, reset_logger):
    """Test setup_backend_logging with default log directory."""
    mock_args.metadata_dir = None
    mock_args.log_level = "debug"
    mock_setup_logging, _, _ = mock_logging
    setup_backend_logging(mock_args, project_prefix="test")
    mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_setup_logging.assert_called_once_with(Path("logs"), "debug", prefix="gridflow_test_")

def test_setup_backend_logging_failure(mock_args, mock_logging, mock_path_mkdir, reset_logger):
    """Test setup_backend_logging when log directory creation fails."""
    mock_args.metadata_dir = "./metadata"
    mock_args.log_level = "debug"
    mock_setup_logging, mock_error, _ = mock_logging
    mock_path_mkdir.side_effect = PermissionError("Access denied")
    with pytest.raises(SystemExit) as exc:
        setup_backend_logging(mock_args, project_prefix="test")
    assert exc.value.code == 1
    mock_error.assert_called_once_with("Failed to create log directory metadata: Access denied")

def test_download_command(mock_args, mock_logging, reset_logger):
    """Test download_command calls cmip6_run_download."""
    mock_args.project = "CMIP6"
    with patch("gridflow.commands.cmip6_run_download") as mock_download:
        download_command(mock_args)
        mock_download.assert_called_once_with(mock_args)

def test_download_cmip5_command(mock_args, mock_logging, reset_logger):
    """Test download_cmip5_command maps time_frequency and calls cmip5_run_download."""
    mock_args.time_frequency = "mon"
    with patch("gridflow.commands.cmip5_run_download") as mock_download:
        download_cmip5_command(mock_args)
        assert mock_args.frequency == "mon"
        mock_download.assert_called_once_with(mock_args)

def test_download_prism_command(mock_args, mock_logging, reset_logger):
    """Test download_prism_command with regular arguments."""
    mock_args.variable = "tmean"
    mock_args.resolution = "4km"
    mock_args.time_step = "monthly"
    mock_args.start_date = "2020-01"
    mock_args.end_date = "2020-03"
    mock_args.output_dir = "./prism_data"
    mock_args.retries = 3
    mock_args.timeout = 30
    mock_args.demo = False
    mock_args.workers = 4
    with patch("gridflow.commands.download_prism") as mock_download:
        download_prism_command(mock_args)
        mock_download.assert_called_once_with(
            variable="tmean",
            resolution="4km",
            time_step="monthly",
            start_date="2020-01",
            end_date="2020-03",
            output_dir="./prism_data",
            metadata_dir="./metadata",
            log_level="minimal",
            retries=3,
            timeout=30,
            demo=False,
            workers=4
        )

def test_download_prism_command_demo(mock_args, mock_logging, reset_logger):
    """Test download_prism_command in demo mode."""
    mock_args.demo = True
    mock_args.output_dir = "./prism_data"  # Match argparse default
    with patch("gridflow.commands.download_prism") as mock_download:
        download_prism_command(mock_args)
        assert mock_args.variable == "tmean"
        assert mock_args.resolution == "4km"
        assert mock_args.time_step == "monthly"
        assert mock_args.start_date == "2020-01"
        assert mock_args.end_date == "2020-03"
        assert mock_args.workers == 4
        mock_download.assert_called_once_with(
            variable="tmean",
            resolution="4km",
            time_step="monthly",
            start_date="2020-01",
            end_date="2020-03",
            output_dir="./prism_data",
            metadata_dir="./metadata",
            log_level="minimal",
            retries=3,
            timeout=30,
            demo=True,
            workers=4
        )

def test_crop_command_valid_args(mock_args, mock_logging, reset_logger):
    """Test crop_command with valid spatial bounds."""
    mock_args.input_dir = "./cmip6_data"
    mock_args.output_dir = "./cropped_data"
    mock_args.min_lat = 40.0
    mock_args.max_lat = 45.0
    mock_args.min_lon = -100.0
    mock_args.max_lon = -95.0
    mock_args.buffer_km = 10.0
    mock_args.workers = 4
    mock_args.demo = False
    with patch("gridflow.commands.crop_netcdf") as mock_crop:
        crop_command(mock_args)
        mock_crop.assert_called_once_with(
            input_dir="./cmip6_data",
            output_dir="./cropped_data",
            min_lat=40.0,
            max_lat=45.0,
            min_lon=-100.0,
            max_lon=-95.0,
            buffer_km=10.0,
            workers=4,
            demo=False
        )

def test_crop_command_missing_args(mock_args, mock_logging, reset_logger):
    """Test crop_command with missing spatial bounds."""
    mock_args.demo = False
    mock_args.min_lat = None
    mock_args.max_lat = 45.0
    mock_args.min_lon = -100.0
    mock_args.max_lon = -95.0
    mock_setup_logging, mock_error, _ = mock_logging
    with pytest.raises(SystemExit) as exc:
        crop_command(mock_args)
    assert exc.value.code == 1
    mock_error.assert_called_once_with(
        "All spatial bounds (--min-lat, --max-lat, --min-lon, --max-lon) must be provided unless --demo"
    )

def test_crop_command_demo(mock_args, mock_logging, reset_logger):
    """Test crop_command in demo mode."""
    mock_args.demo = True
    mock_args.input_dir = "./cmip6_data"
    mock_args.output_dir = "./cropped_data"
    mock_args.min_lat = None
    mock_args.max_lat = None
    mock_args.min_lon = None
    mock_args.max_lon = None
    mock_args.buffer_km = 10.0
    mock_args.workers = 4
    with patch("gridflow.commands.crop_netcdf") as mock_crop:
        crop_command(mock_args)
        mock_crop.assert_called_once()

def test_clip_command(mock_args, mock_logging, reset_logger):
    """Test clip_command with shapefile path."""
    mock_args.input_dir = "./cmip6_data"
    mock_args.shapefile_path = "./iowa_border.shp"
    mock_args.buffer_km = 20.0
    mock_args.output_dir = "./clipped_data"
    mock_args.workers = 4
    with patch("gridflow.commands.clip_netcdf") as mock_clip:
        clip_command(mock_args)
        mock_clip.assert_called_once_with(
            input_dir="./cmip6_data",
            shapefile_path="./iowa_border.shp",
            buffer_km=20.0,
            output_dir="./clipped_data",
            workers=4
        )

def test_catalog_command(mock_args, mock_logging, reset_logger):
    """Test catalog_command with regular arguments."""
    mock_args.input_dir = "./cmip6_data"
    mock_args.output_dir = "./catalog"
    mock_args.workers = 4
    mock_args.demo = False
    with patch("gridflow.commands.generate_catalog") as mock_generate:
        mock_generate.return_value = {"catalog": "data"}
        catalog_command(mock_args)
        mock_generate.assert_called_once_with(
            input_dir="./cmip6_data",
            output_dir="./catalog",
            demo_mode=False,
            workers=4
        )

def test_catalog_command_demo_failure(mock_args, mock_logging, reset_logger):
    """Test catalog_command in demo mode with failed catalog generation."""
    mock_args.input_dir = "./cmip6_data"
    mock_args.output_dir = "./catalog"
    mock_args.demo = True
    mock_args.workers = None  # Match argparse default
    mock_setup_logging, _, mock_info = mock_logging
    with patch("gridflow.commands.generate_catalog") as mock_generate:
        mock_generate.return_value = {}
        with pytest.raises(SystemExit) as exc:
            catalog_command(mock_args)
        assert exc.value.code == 0
        mock_info.assert_called_once_with("Catalog generation failed in demo mode, exiting")

def test_main_download(mock_args):
    """Test main() with download command."""
    with patch("gridflow.commands.argparse.ArgumentParser.parse_args") as mock_parse:
        with patch("gridflow.commands.download_command") as mock_download:
            mock_args.command = "download"
            mock_parse.return_value = mock_args
            main()
            mock_download.assert_called_once_with(mock_args)

def test_main_invalid_command(mock_args):
    """Test main() with an invalid command."""
    with patch("gridflow.commands.argparse.ArgumentParser.parse_args") as mock_parse:
        with patch("gridflow.commands.logging.error") as mock_error:
            mock_args.command = "invalid"
            mock_parse.return_value = mock_args
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
            mock_error.assert_called_once_with("Command invalid failed: 'invalid'")

def test_main_command_exception(mock_args):
    """Test main() when a command raises an exception."""
    with patch("gridflow.commands.argparse.ArgumentParser.parse_args") as mock_parse:
        with patch("gridflow.commands.download_command") as mock_download:
            with patch("gridflow.commands.logging.error") as mock_error:
                mock_args.command = "download"
                mock_parse.return_value = mock_args
                mock_download.side_effect = ValueError("Test error")
                with pytest.raises(SystemExit) as exc:
                    main()
                assert exc.value.code == 1
                mock_error.assert_called_once_with("Command download failed: Test error")