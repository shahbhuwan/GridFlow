# tests/test_dem_downloader.py

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from requests.exceptions import HTTPError, RequestException

import gridflow.download.dem_downloader as dem_downloader
from gridflow.download.dem_downloader import (
    DEMDownloader,
    run_dem_download_session,
    signal_handler,
    stop_event,
    add_arguments,
    main,
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
def dem_downloader(mock_stop_event):
    settings = {'api_key': 'key', 'bounds': {'north': 40, 'south': 30, 'east': -100, 'west': -110}, 'output_file': 'dem.tif', 'dem_type': 'COP30'}
    return DEMDownloader(settings, mock_stop_event)

# ############################################################################
# Tests for signal_handler
# ############################################################################

def test_signal_handler(caplog, mocker):
    caplog.set_level(logging.INFO)
    mock_event = mocker.patch('gridflow.download.dem_downloader.stop_event')
    signal_handler(None, None)
    assert mock_event.set.called
    assert "Stop signal received" in caplog.text
    assert "Please wait for ongoing tasks" in caplog.text

# ############################################################################
# Tests for DEMDownloader
# ############################################################################

def test_dem_downloader_init(mock_stop_event):
    settings = {'api_key': 'key', 'bounds': {'north': 40, 'south': 30, 'east': -100, 'west': -110}, 'output_file': 'dem.tif', 'dem_type': 'COP30'}
    dl = DEMDownloader(settings, mock_stop_event)
    assert dl.settings == settings
    assert dl._stop_event == mock_stop_event

def test_download_success(mocker, dem_downloader, tmp_path):
    output_file = tmp_path / "dem.tif"
    dem_downloader.settings['output_file'] = str(output_file)
    mock_get = mocker.patch('requests.get')
    mock_get.return_value.iter_content.return_value = [b'data']
    mock_get.return_value.raise_for_status = MagicMock()
    assert dem_downloader.download() is True
    assert output_file.exists()

def test_download_interrupted_before(mock_stop_event, dem_downloader):
    mock_stop_event.is_set.return_value = True
    assert dem_downloader.download() is False

def test_download_http_error(mocker, dem_downloader, caplog):
    mock_get = mocker.patch('requests.get', side_effect=HTTPError(response=MagicMock(status_code=401, text="Unauthorized")))
    assert dem_downloader.download() is False
    assert "HTTP Error 401" in caplog.text

def test_download_bad_request(mocker, dem_downloader, caplog):
    mock_get = mocker.patch('requests.get', side_effect=HTTPError(response=MagicMock(status_code=400, text="Invalid bounds")))
    assert dem_downloader.download() is False
    assert "Bad Request" in caplog.text

def test_download_request_exception(mocker, dem_downloader, caplog):
    mocker.patch('requests.get', side_effect=RequestException("Network error"))
    assert dem_downloader.download() is False
    assert "A network error occurred" in caplog.text

def test_download_unexpected_error(mocker, dem_downloader, caplog):
    mocker.patch('requests.get', side_effect=Exception("Unexpected"))
    assert dem_downloader.download() is False
    assert "An unexpected error occurred" in caplog.text

# ############################################################################
# Tests for run_dem_download_session
# ############################################################################

def test_run_dem_download_session_success(mocker, mock_stop_event):
    mocker.patch.object(DEMDownloader, 'download', return_value=True)
    settings = {'api_key': 'key', 'bounds': {'north': 40, 'south': 30, 'east': -100, 'west': -110}, 'output_file': 'dem.tif', 'dem_type': 'COP30'}
    run_dem_download_session(settings, mock_stop_event)

def test_run_dem_download_session_exception(mocker, mock_stop_event, caplog):
    mocker.patch.object(DEMDownloader, 'download', side_effect=Exception("Critical"))
    settings = {'api_key': 'key', 'bounds': {'north': 40, 'south': 30, 'east': -100, 'west': -110}, 'output_file': 'dem.tif', 'dem_type': 'COP30'}
    run_dem_download_session(settings, mock_stop_event)
    assert "critical error" in caplog.text
    assert mock_stop_event.set.called

# ############################################################################
# Tests for add_arguments
# ############################################################################

def test_add_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--api_key', 'key', '--bounds', '40', '30', '-100', '-110', '--output_file', 'dem.tif'])
    assert args.api_key == 'key'
    assert args.bounds == [40.0, 30.0, -100.0, -110.0]
    assert args.output_file == 'dem.tif'
    assert args.dem_type == 'COP30'

# ############################################################################
# Tests for main
# ############################################################################

@patch('argparse.ArgumentParser')
@patch('gridflow.download.dem_downloader.setup_logging')
@patch('gridflow.download.dem_downloader.run_dem_download_session')
@patch('signal.signal')
def test_main_success(mock_signal, mock_session, mock_logging, mock_parser):
    mock_args = MagicMock(api_key='key', bounds=[40, 30, -100, -110], output_file='dem.tif', log_dir='./logs', log_level='info')
    mock_parser.return_value.parse_args.return_value = mock_args
    main()
    mock_session.assert_called()
    mock_signal.assert_called_with(signal.SIGINT, signal_handler)

@patch('argparse.ArgumentParser')
@patch('gridflow.download.dem_downloader.setup_logging')
@patch('gridflow.download.dem_downloader.run_dem_download_session')
def test_main_demo(mock_session, mock_logging, mock_parser, caplog):
    caplog.set_level(logging.INFO)
    mock_args = MagicMock(demo=True, api_key='key', log_dir='./logs', log_level='info')
    mock_parser.return_value.parse_args.return_value = mock_args
    main()
    assert "Running in demo mode" in caplog.text

@patch('argparse.ArgumentParser')
@patch('gridflow.download.dem_downloader.setup_logging')
def test_main_no_params(mock_logging, mock_parser, caplog):
    caplog.set_level(logging.INFO)
    mock_args = MagicMock(demo=False, api_key=None, bounds=None, output_file=None)
    mock_parser.return_value.parse_args.return_value = mock_args
    with pytest.raises(SystemExit):
        main()
    assert "--api_key, --bounds, and --output_file are required" in caplog.text

@patch('argparse.ArgumentParser')
@patch('gridflow.download.dem_downloader.setup_logging')
@patch('gridflow.download.dem_downloader.stop_event')
def test_main_interrupted(mock_stop, mock_logging, mock_parser, caplog):
    mock_stop.is_set.return_value = True
    mock_parser.return_value.parse_args.return_value = MagicMock()
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 130
    assert "Execution was interrupted" in caplog.text