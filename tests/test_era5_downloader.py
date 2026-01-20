# tests/test_era5_downloader.py

import argparse
import logging
import signal
import json
import threading
from concurrent.futures import Future
from importlib import reload
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from datetime import datetime

import pytest
import boto3
from botocore.exceptions import ClientError

import gridflow.download.era5_downloader as era5_downloader
from gridflow.download.era5_downloader import (
    Downloader, 
    QueryHandler, 
    FileManager, 
    create_download_session, 
    signal_handler, 
    stop_event, 
    add_arguments, 
    main,
    VARIABLE_MAP
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
    handler.s3 = mock_s3_client 
    return handler

@pytest.fixture
def downloader(file_manager, mock_stop_event, mock_s3_client):
    settings = {'workers': 1, 'is_gui_mode': False}
    dl = Downloader(file_manager, mock_stop_event, **settings)
    dl.s3 = mock_s3_client 
    return dl

# ############################################################################
# Tests for Import Dependencies
# ############################################################################

def test_missing_dependency_boto3(capsys):
    with patch.dict('sys.modules', {'boto3': None}):
        with pytest.raises(SystemExit) as exc:
            reload(era5_downloader)
        assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Missing required library: boto3" in captured.out

# ############################################################################
# Tests for Signal Handler
# ############################################################################

def test_signal_handler(caplog):
    caplog.set_level(logging.INFO)
    era5_downloader.stop_event.clear()
    signal_handler(None, None)
    assert era5_downloader.stop_event.is_set()
    assert "Stop signal received" in caplog.text

# ############################################################################
# Tests for FileManager
# ############################################################################

def test_file_manager_paths(file_manager):
    file_info = {'filename': 'test_file.nc'}
    path = file_manager.get_output_path(file_info)
    assert path == file_manager.download_dir / 'test_file.nc'

def test_save_metadata(file_manager):
    files = [{'filename': 'test.nc', 'output_path': Path('/tmp/test.nc')}]
    file_manager.save_metadata(files, "meta.json")
    assert (file_manager.metadata_dir / "meta.json").exists()

# ############################################################################
# Tests for QueryHandler
# ############################################################################

def test_generate_potential_files_valid(query_handler):
    # Test mapping logic (t2m -> e5.oper.an.sfc)
    files = query_handler.generate_potential_files(['t2m'], "2021-01-01", "2021-02-01")
    assert len(files) == 2 # Jan an Febd
    assert files[0]['variable_user'] == 't2m'
    assert 'e5.oper.an.sfc' in files[0]['prefix']
    assert files[0]['year'] == 2021
    assert files[0]['month'] == 1

def test_generate_potential_files_composite(query_handler):
    files = query_handler.generate_potential_files(['precip'], "2021-01-01", "2021-01-31")
    assert len(files) == 2 # CP and LSP
    codes = [f['prefix'] for f in files]
    assert any("128_143_cp" in c for c in codes)
    assert any("128_142_lsp" in c for c in codes)

def test_generate_potential_files_invalid(query_handler, caplog):
    files = query_handler.generate_potential_files(['invalid_var'], "2021-01-01", "2021-01-01")
    assert len(files) == 0
    assert "not found in mapping" in caplog.text

def test_validate_files_on_s3_success(query_handler, mock_s3_client):
    # Mock S3 response
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [{'Key': 'bucket/path/file.nc', 'Size': 12345}]
    }
    
    potential = [{'prefix': 'p', 's3_bucket': 'b', 'variable_user': 'v', 'year': 2021, 'month': 1}]
    valid = query_handler.validate_files_on_s3(potential)
    
    assert len(valid) == 1
    assert valid[0]['s3_key'] == 'bucket/path/file.nc'
    assert valid[0]['size_bytes'] == 12345

def test_validate_files_on_s3_empty(query_handler, mock_s3_client):
    mock_s3_client.list_objects_v2.return_value = {} # No Contents
    potential = [{'prefix': 'p', 's3_bucket': 'b', 'variable_user': 'v', 'year': 2021, 'month': 1}]
    valid = query_handler.validate_files_on_s3(potential)
    assert len(valid) == 0

# ############################################################################
# Tests for Downloader
# ############################################################################

def test_download_file_success(downloader, tmp_path, mock_s3_client):
    output = tmp_path / "downloads" / "file.nc"
    file_info = {'filename': 'file.nc', 's3_bucket': 'b', 's3_key': 'k', 'size_bytes': 100, 'title': 'file.nc'}
    
    def side_effect(bucket, key, path):
        with open(path, 'w') as f: f.write("data")
    mock_s3_client.download_file.side_effect = side_effect

    path, error = downloader.download_file(file_info)
    
    assert path == str(output)
    assert error is None
    assert output.exists()
    mock_s3_client.download_file.assert_called_with('b', 'k', str(output))

def test_download_file_skip_existing(downloader, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    output = tmp_path / "downloads" / "file.nc"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'wb') as f: f.write(b'\0' * 100) 
    
    file_info = {'filename': 'file.nc', 's3_bucket': 'b', 's3_key': 'k', 'size_bytes': 100}
    
    path, error = downloader.download_file(file_info)
    
    assert path == str(output)
    assert "Skipping" in caplog.text
    downloader.s3.download_file.assert_not_called()

def test_download_file_failure(downloader, mock_s3_client):
    mock_s3_client.download_file.side_effect = Exception("S3 Error")
    file_info = {'filename': 'fail.nc', 's3_bucket': 'b', 's3_key': 'k', 'title': 'fail'}
    
    path, error = downloader.download_file(file_info)
    
    assert path is None
    assert error is not None
    assert "S3 Error" in error['error']

def test_download_all_success(downloader):
    with patch.object(downloader, 'download_file', return_value=('path/to/file', None)):
        files = [{'f': 1}, {'f': 2}]
        downloaded, failed = downloader.download_all(files)
        assert len(downloaded) == 2
        assert len(failed) == 0

def test_download_all_interrupted(downloader, mock_stop_event):
    files = [{'f': 1}, {'f': 2}, {'f': 3}]
    
    mock_futures = [MagicMock(spec=Future) for _ in files]
    mock_futures[0].result.return_value = ('path1', None)
    
    with patch('gridflow.download.era5_downloader.ThreadPoolExecutor') as mock_tpe:
        mock_executor = mock_tpe.return_value
        mock_executor.submit.side_effect = mock_futures
        
        def as_completed_gen(fs):
            yield mock_futures[0]
            mock_stop_event.is_set.return_value = True
            
        with patch('gridflow.download.era5_downloader.as_completed', side_effect=as_completed_gen):
            downloaded, failed = downloader.download_all(files)

            assert len(downloaded) == 1
            mock_executor.shutdown.assert_called()

# ############################################################################
# Tests for create_download_session
# ############################################################################

@patch('gridflow.download.era5_downloader.QueryHandler')
@patch('gridflow.download.era5_downloader.Downloader')
def test_create_download_session_success(mock_dl_cls, mock_qh_cls, mock_stop_event):
    settings = {
        'variables': 't2m', 'start_date': '2021-01-01', 'end_date': '2021-01-01',
        'output_dir': '.', 'metadata_dir': '.', 'workers': 1
    }
    
    mock_qh = mock_qh_cls.return_value
    mock_qh.generate_potential_files.return_value = [{'p': 1}]
    mock_qh.validate_files_on_s3.return_value = [{'valid': 1}]
    
    mock_dl = mock_dl_cls.return_value
    mock_dl.download_all.return_value = (['path'], [])
    
    create_download_session(settings, mock_stop_event)
    
    mock_qh.generate_potential_files.assert_called()
    mock_qh.validate_files_on_s3.assert_called()
    mock_dl.download_all.assert_called()

def test_create_download_session_no_vars(mock_stop_event, caplog):
    settings = {'variables': '', 'is_gui_mode': False}
    with pytest.raises(SystemExit):
        create_download_session(settings, mock_stop_event)
    assert "No variables specified" in caplog.text

def test_create_download_session_invalid_vars(mock_stop_event, caplog):
    settings = {
        'variables': 'bad_var', 'start_date': '2021-01-01', 'end_date': '2021-01-01',
        'is_gui_mode': False
    }
    with patch('gridflow.download.era5_downloader.QueryHandler') as mock_qh_cls:
        mock_qh = mock_qh_cls.return_value
        mock_qh.generate_potential_files.return_value = [] 
        
        with pytest.raises(SystemExit):
            create_download_session(settings, mock_stop_event)
        
        assert "No valid variables" in caplog.text

def test_create_download_session_no_files_found(mock_stop_event, caplog):
    caplog.set_level(logging.INFO)
    settings = {
        'variables': 't2m', 'start_date': '2021-01-01', 'end_date': '2021-01-01',
        'output_dir': '.', 'metadata_dir': '.', 'is_gui_mode': False
    }
    with patch('gridflow.download.era5_downloader.QueryHandler') as mock_qh_cls:
        mock_qh = mock_qh_cls.return_value
        mock_qh.generate_potential_files.return_value = [{'p': 1}]
        mock_qh.validate_files_on_s3.return_value = [] # S3 found nothing
        
        with pytest.raises(SystemExit):
            create_download_session(settings, mock_stop_event)
        
        assert "No available files" in caplog.text

# ############################################################################
# Tests for main
# ############################################################################

def test_add_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--variables', 't2m', '-sd', '2021-01-01', '-ed', '2021-01-02'])
    assert args.variables == 't2m'
    assert args.start_date == '2021-01-01'
    assert args.end_date == '2021-01-02'

@patch('argparse.ArgumentParser.parse_args')
@patch('gridflow.download.era5_downloader.create_download_session')
@patch('gridflow.download.era5_downloader.setup_logging')
def test_main_demo(mock_logging, mock_session, mock_args, caplog, capsys):
    """Verify demo mode defaults and output."""
    caplog.set_level(logging.INFO)
    
    mock_namespace = MagicMock()
    mock_namespace.demo = True
    mock_namespace.config = None
    mock_namespace.variables = None
    mock_namespace.is_gui_mode = False
    mock_namespace.list_variables = False

    mock_namespace.stop_event.is_set.return_value = False
    
    mock_args.return_value = mock_namespace
    
    main()
    
    from gridflow.download.era5_downloader import HAS_UI_LIBS
    
    if HAS_UI_LIBS:
        captured = capsys.readouterr()
        assert "Running in demo mode" in captured.out
    else:
        assert "Running in demo mode" in caplog.text
        
    args, _ = mock_session.call_args
    assert args[0]['variables'] == ['t2m', 'precip']

@patch('argparse.ArgumentParser.parse_args')
@patch('gridflow.download.era5_downloader.setup_logging')
def test_main_missing_args(mock_logging, mock_args, caplog):
    mock_namespace = MagicMock()
    mock_namespace.demo = False
    mock_namespace.config = None
    mock_namespace.variables = 't2m'
    mock_namespace.start_date = None 
    mock_namespace.is_gui_mode = False
    mock_namespace.list_variables = False
    mock_namespace.stop_event.is_set.return_value = False

    mock_args.return_value = mock_namespace
    
    with pytest.raises(SystemExit):
        main()
    
    assert "Required arguments missing" in caplog.text

@patch('argparse.ArgumentParser.parse_args')
def test_main_list_variables(mock_args, capsys):
    mock_args.return_value = MagicMock(list_variables=True)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Short Code" in captured.out or "Available ERA5 Variables" in captured.out