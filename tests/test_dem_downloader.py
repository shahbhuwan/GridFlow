# tests/test_dem_downloader.py

import argparse
import logging
import threading
from unittest.mock import MagicMock, patch, call
from pathlib import Path

import pytest
import boto3
from botocore.exceptions import ClientError

# FIX 1: Rename module import to avoid shadowing the 'dem_downloader' fixture below
import gridflow.download.dem_downloader as dem_module
from gridflow.download.dem_downloader import (
    Downloader, 
    QueryHandler, 
    FileManager, 
    create_download_session, 
    signal_handler, 
    stop_event, 
    add_arguments, 
    main
)

# ############################################################################
# Fixtures
# ############################################################################

@pytest.fixture
def mock_stop_event():
    event = MagicMock(spec=threading.Event)
    event.is_set.return_value = False
    return event

@pytest.fixture
def mock_s3_client(mocker):
    """Mocks the boto3 S3 client."""
    mock_client = MagicMock()
    mocker.patch('boto3.client', return_value=mock_client)
    return mock_client

@pytest.fixture
def file_manager(tmp_path):
    return FileManager(str(tmp_path / "downloads"), str(tmp_path / "metadata"))

@pytest.fixture
def query_handler(mock_stop_event, mock_s3_client):
    handler = QueryHandler(mock_stop_event)
    handler.s3 = mock_s3_client # Inject mock
    return handler

@pytest.fixture
def downloader(file_manager, mock_stop_event, mock_s3_client):
    settings = {'workers': 1, 'is_gui_mode': False}
    dl = Downloader(file_manager, mock_stop_event, **settings)
    dl.s3 = mock_s3_client # Inject mock
    return dl

# ############################################################################
# Tests for Signal Handler
# ############################################################################

def test_signal_handler(caplog):
    caplog.set_level(logging.INFO)
    stop_event.clear()
    signal_handler(None, None)
    assert stop_event.is_set()
    assert "Stop signal received" in caplog.text

# ############################################################################
# Tests for FileManager
# ############################################################################

def test_file_manager_paths(file_manager):
    file_info = {'filename': 'test_tile.tif', 'source': 'COP30'}
    path = file_manager.get_output_path(file_info)
    assert path == file_manager.download_dir / 'COP30' / 'test_tile.tif'

def test_save_metadata(file_manager):
    files = [{'filename': 'test.json', 'output_path': Path('/tmp/test.tif')}]
    file_manager.save_metadata(files, "meta.json")
    assert (file_manager.metadata_dir / "meta.json").exists()

# ############################################################################
# Tests for QueryHandler (Tile Generation logic)
# ############################################################################

def test_format_coord(query_handler):
    # Test internal helper method for N/S E/W formatting
    assert query_handler._format_cop_coord(45.5, True) == "N45"
    
    assert query_handler._format_cop_coord(-10.1, True) == "S11"
    
    assert query_handler._format_cop_coord(5.0, False) == "E005"
    assert query_handler._format_cop_coord(-93.2, False) == "W094" # floor(-93.2) -> -94

def test_generate_potential_files(query_handler):
    # Test generating tiles for a 2x2 degree area
    # Bounds: North 41.9, South 40.1, West -91.9, East -90.1
    # Lat range: floor(40.1)=40 to floor(41.9)=41 -> [40, 41]
    # Lon range: floor(-91.9)=-92 to floor(-90.1)=-91 -> [-92, -91]
    bounds = {'north': 41.9, 'south': 40.1, 'west': -91.9, 'east': -90.1}
    
    files = query_handler.generate_potential_files(bounds)
    
    assert len(files) == 4
    filenames = [f['filename'] for f in files]
    
    # FIX 3: Updated expected filenames based on correct floor logic
    # -92 -> W092, -91 -> W091
    assert "Copernicus_DSM_COG_10_N40_00_W091_00_DEM.tif" in filenames
    assert "Copernicus_DSM_COG_10_N41_00_W091_00_DEM.tif" in filenames
    # W090 is NOT included because -90.1 falls in the -91 bin (interval -91 to -90)

def test_validate_files_on_s3_exists(query_handler, mock_s3_client):
    # Mock HEAD object success
    mock_s3_client.head_object.return_value = {}

    potential = [{'s3_bucket': 'b', 's3_key': 'k', 'filename': 'f', 'source': 'COP30', 'title': 'f'}]
    valid = query_handler.validate_files_on_s3(potential)
    
    assert len(valid) == 1
    assert valid[0]['filename'] == 'f'

def test_validate_files_on_s3_missing(query_handler, mock_s3_client):
    # Mock HEAD object 404 (ClientError)
    error_response = {'Error': {'Code': '404', 'Message': 'Not Found'}}
    mock_s3_client.head_object.side_effect = ClientError(error_response, 'HeadObject')
    
    potential = [{'s3_bucket': 'b', 's3_key': 'k', 'filename': 'f'}]
    valid = query_handler.validate_files_on_s3(potential)
    
    assert len(valid) == 0

# ############################################################################
# Tests for Downloader
# ############################################################################

def test_download_file_success(downloader, tmp_path, mock_s3_client):
    output = tmp_path / "downloads" / "COP30" / "file.tif"
    
    file_info = {'filename': 'file.tif', 's3_bucket': 'b', 's3_key': 'k', 'title': 'file.tif', 'source': 'COP30'}
    
    # Mock S3 download side effect
    def side_effect(bucket, key, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f: f.write("data")
    mock_s3_client.download_file.side_effect = side_effect

    path, error = downloader.download_file(file_info)
    
    assert path == str(output)
    assert error is None
    assert output.exists()
    mock_s3_client.download_file.assert_called_with('b', 'k', str(output))

def test_download_file_skip_existing(downloader, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)

    output = tmp_path / "downloads" / "COP30" / "file.tif"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'wb') as f: f.write(b'data')

    file_info = {'filename': 'file.tif', 's3_bucket': 'b', 's3_key': 'k', 'source': 'COP30', 'title': 'file.tif'}
    
    path, error = downloader.download_file(file_info)
    
    assert path == str(output)
    assert "Skipping" in caplog.text
    downloader.s3_cop.download_file.assert_not_called()

def test_download_file_failure(downloader, mock_s3_client):
    mock_s3_client.download_file.side_effect = Exception("Connection Lost")
    
    file_info = {'filename': 'fail.tif', 's3_bucket': 'b', 's3_key': 'k', 'title': 'fail', 'source': 'COP30'}
    
    path, error = downloader.download_file(file_info)
    
    assert path is None
    assert error is not None
    assert "Connection Lost" in error['error']

def test_download_all_flow(downloader):
    # Mock download_file to avoid actual threading complexity
    with patch.object(downloader, 'download_file', return_value=('path', None)):
        files = [{'f': 1}, {'f': 2}]
        downloaded, failed = downloader.download_all(files)
        assert len(downloaded) == 2
        assert len(failed) == 0

# ############################################################################
# Tests for create_download_session
# ############################################################################

@patch('gridflow.download.dem_downloader.QueryHandler')
@patch('gridflow.download.dem_downloader.Downloader')
def test_create_download_session_success(mock_dl_cls, mock_qh_cls, mock_stop_event):
    settings = {
        'bounds': {'north': 40, 'south': 30, 'east': -10, 'west': -20},
        'output_dir': '.', 'metadata_dir': '.', 'workers': 1
    }
    
    # Mock QueryHandler responses
    mock_qh = mock_qh_cls.return_value
    mock_qh.generate_potential_files.return_value = [{'p': 1}]
    mock_qh.validate_files_on_s3.return_value = [{'valid': 1}]
    
    # Mock Downloader responses
    mock_dl = mock_dl_cls.return_value
    mock_dl.download_all.return_value = (['path'], [])
    
    create_download_session(settings, mock_stop_event)
    
    mock_qh.generate_potential_files.assert_called()
    mock_qh.validate_files_on_s3.assert_called()
    mock_dl.download_all.assert_called()

def test_create_download_session_missing_bounds(mock_stop_event, caplog):
    settings = {'bounds': None, 'is_gui_mode': False}
    with pytest.raises(SystemExit):
        create_download_session(settings, mock_stop_event)
    assert "Bounds (North, South, East, West) are required" in caplog.text

def test_create_download_session_no_files_found(mock_stop_event, caplog):
    caplog.set_level(logging.INFO)
    settings = {
        'bounds': {'north': 0, 'south': 0, 'east': 0, 'west': 0}, # Point in ocean?
        'output_dir': '.', 'metadata_dir': '.', 'is_gui_mode': False
    }
    with patch('gridflow.download.dem_downloader.QueryHandler') as mock_qh_cls:
        mock_qh = mock_qh_cls.return_value
        mock_qh.generate_potential_files.return_value = [{'p': 1}]
        mock_qh.validate_files_on_s3.return_value = [] # Found nothing (ocean)
        
        with pytest.raises(SystemExit):
            create_download_session(settings, mock_stop_event)
        
        assert "No available tiles found on S3" in caplog.text

def test_generate_potential_files_usgs(query_handler):
    """Verifies USGS 10m tile generation naming convention (nXXwYYY)."""
    # Bounds for a small area in Iowa
    bounds = {'north': 42.5, 'south': 41.5, 'west': -93.5, 'east': -92.5}
    
    files = query_handler.generate_potential_files(bounds, dem_type='USGS10m')
    
    # Expected lat range: ceil(41.5)=42 to ceil(42.5)=43 -> [42, 43]
    # Expected lon range: floor(-93.5)=-94 to floor(-92.5)=-93 -> [-94, -93]
    assert len(files) == 4
    filenames = [f['filename'] for f in files]
    
    assert "USGS_13_n42w093.tif" in filenames
    assert "USGS_13_n43w094.tif" in filenames

# ############################################################################
# Tests for main
# ############################################################################

def test_add_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--bounds', '40', '30', '-10', '-20'])
    assert args.bounds == [40.0, 30.0, -10.0, -20.0]

@patch('argparse.ArgumentParser.parse_args')
@patch('gridflow.download.dem_downloader.create_download_session')
@patch('gridflow.download.dem_downloader.setup_logging')
def test_main_demo(mock_logging, mock_session, mock_args, caplog, capsys):
    """Verifies demo mode triggers dual sessions with independent settings."""
    caplog.set_level(logging.INFO)

    # Configure the mock namespace
    mock_namespace = MagicMock()
    mock_namespace.demo = True
    mock_namespace.config = None
    mock_namespace.bounds = None
    mock_namespace.is_gui_mode = False
    mock_namespace.log_dir = "./gridflow_logs"
    mock_namespace.log_level = "minimal"

    mock_stop_event = MagicMock()
    mock_stop_event.is_set.return_value = False
    mock_namespace.stop_event = mock_stop_event
    
    mock_args.return_value = mock_namespace
    
    main()

    captured = capsys.readouterr()
    assert "Running in demo mode" in captured.out

    assert "Demo complete" in caplog.text

    assert mock_session.call_count == 2

    assert mock_session.call_args_list[0][0][0]['dem_type'] == 'COP30'
    assert mock_session.call_args_list[0][0][0]['bounds']['north'] == 42.9

    assert mock_session.call_args_list[1][0][0]['dem_type'] == 'USGS10m'

@patch('argparse.ArgumentParser.parse_args')
@patch('gridflow.download.dem_downloader.setup_logging')
def test_main_missing_args(mock_logging, mock_args, caplog):
    # Setup mock to fail validation
    mock_namespace = MagicMock()
    mock_namespace.demo = False
    mock_namespace.config = None
    mock_namespace.bounds = None # Missing
    mock_namespace.is_gui_mode = False
    mock_namespace.stop_event.is_set.return_value = False
    
    mock_args.return_value = mock_namespace
    
    with pytest.raises(SystemExit):
        main()
    
    assert "Bounds are required" in caplog.text

@patch('argparse.ArgumentParser.parse_args')
@patch('gridflow.download.dem_downloader.setup_logging')
def test_main_invalid_bounds(mock_logging, mock_args, caplog):
    # Setup mock with invalid bounds (North < South)
    mock_namespace = MagicMock()
    mock_namespace.demo = False
    mock_namespace.config = None
    mock_namespace.bounds = [30, 40, -10, -20] # North(30) < South(40)
    mock_namespace.is_gui_mode = False
    
    mock_args.return_value = mock_namespace
    
    with pytest.raises(SystemExit):
        main()
    
    assert "Invalid bounds: North must be greater than South" in caplog.text