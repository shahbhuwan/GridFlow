# tests/test_prism_downloader.py

import argparse
import logging
import signal
import json
import threading
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import requests
from requests.exceptions import RequestException

import gridflow.download.prism_downloader as prism_downloader
from gridflow.download.prism_downloader import (
    FileManager,
    AvailabilityChecker,
    Downloader,
    create_download_session,
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
def file_manager(tmp_path):
    return FileManager(str(tmp_path / "downloads"), str(tmp_path / "metadata"))

@pytest.fixture
def availability_checker(mock_stop_event):
    return AvailabilityChecker(mock_stop_event, workers=1, timeout=10)

@pytest.fixture
def downloader(mock_stop_event):
    return Downloader(mock_stop_event, workers=1, timeout=10)

# ############################################################################
# Tests for signal_handler
# ############################################################################

def test_signal_handler(caplog, mocker):
    caplog.set_level(logging.INFO)
    mock_event = mocker.patch('gridflow.download.prism_downloader.stop_event')
    signal_handler(None, None)
    assert mock_event.set.called
    assert "Stop signal received" in caplog.text
    assert "Please wait for ongoing tasks" in caplog.text

# ############################################################################
# Tests for FileManager
# ############################################################################

def test_file_manager_init(tmp_path):
    fm = FileManager(str(tmp_path / "downloads"), str(tmp_path / "metadata"), metadata_prefix="meta_")
    assert fm.download_dir.exists()
    assert fm.metadata_dir.exists()
    assert fm.metadata_prefix == "meta_"

def test_file_manager_init_error(mocker, caplog):
    mocker.patch('pathlib.Path.mkdir', side_effect=OSError("Permission denied"))
    with pytest.raises(SystemExit):
        FileManager("/invalid", "/invalid")
    assert "Failed to create directories" in caplog.text

def test_get_output_path(file_manager):
    path = file_manager.get_output_path("ppt", "4km", "file.zip")
    expected = file_manager.download_dir / 'ppt' / '4km' / 'file.zip'
    assert path == expected
    assert path.parent.exists()

def test_save_metadata(file_manager, tmp_path):
    files = [{'url': 'http://url'}]
    file_manager.save_metadata(files, "test.json")
    metadata_path = file_manager.metadata_dir / "test.json"
    assert metadata_path.exists()
    with open(metadata_path, 'r') as f:
        assert json.load(f) == [{'url': 'http://url'}]

def test_save_metadata_error(mocker, file_manager, caplog):
    mocker.patch('builtins.open', side_effect=IOError("Write error"))
    file_manager.save_metadata([], "test.json")
    assert "Failed to save metadata" in caplog.text

# ############################################################################
# Tests for AvailabilityChecker
# ############################################################################

def test_availability_checker_init(mock_stop_event):
    checker = AvailabilityChecker(mock_stop_event, workers=2, timeout=20)
    assert checker._stop_event == mock_stop_event
    assert checker.workers == 2
    assert checker.timeout == 20

def test_check_url_success(mocker, availability_checker):
    mock_head = mocker.patch('requests.head', return_value=MagicMock(status_code=200))
    assert availability_checker.check_url("http://url") is True
    mock_head.assert_called_with("http://url", timeout=availability_checker.timeout, allow_redirects=True)

def test_check_url_failure(mocker, availability_checker):
    mock_head = mocker.patch('requests.head', return_value=MagicMock(status_code=404))
    assert availability_checker.check_url("http://url") is False

def test_check_url_exception(mocker, availability_checker):
    mocker.patch('requests.head', side_effect=RequestException("Network error"))
    assert availability_checker.check_url("http://url") is False

def test_check_url_interrupted(mock_stop_event, availability_checker):
    mock_stop_event.is_set.return_value = True
    assert availability_checker.check_url("http://url") is False

def test_find_available_files_success(mocker, availability_checker):
    mocker.patch.object(availability_checker, 'check_url', return_value=True)
    potential_files = [{'url': 'http://url1'}, {'url': 'http://url2'}]
    available = availability_checker.find_available_files(potential_files)
    assert len(available) == 2

def test_find_available_files_partial(mocker, availability_checker):
    mocker.patch.object(availability_checker, 'check_url', side_effect=[True, False])
    potential_files = [{'url': 'http://url1'}, {'url': 'http://url2'}]
    available = availability_checker.find_available_files(potential_files)
    assert len(available) == 1
    assert available[0]['url'] == 'http://url1'

def test_find_available_files_interrupted(mock_stop_event, availability_checker):
    mock_stop_event.is_set.return_value = True
    available = availability_checker.find_available_files([{'url': 'http://url'}])
    assert available == []

# ############################################################################
# Tests for Downloader
# ############################################################################

def test_downloader_init(mock_stop_event):
    dl = Downloader(mock_stop_event, workers=2, timeout=20)
    assert dl._stop_event == mock_stop_event
    assert dl.workers == 2
    assert dl.timeout == 20
    assert dl.successful_downloads == 0
    assert dl.executor is None

def test_shutdown(downloader):
    downloader.executor = MagicMock()
    downloader.shutdown()
    downloader.executor.shutdown.assert_called_with(wait=True, cancel_futures=True)

def test_download_file_success(mocker, downloader, tmp_path):
    output_path = tmp_path / "file.zip"
    file_info = {'url': 'http://url', 'output_path': output_path}
    mock_get = mocker.patch('requests.get')
    mock_get.return_value.iter_content.return_value = [b'data']
    mock_get.return_value.raise_for_status = MagicMock()
    result = downloader.download_file(file_info)
    assert result == file_info
    assert output_path.exists()

def test_download_file_interrupted_before(mock_stop_event, downloader):
    mock_stop_event.is_set.return_value = True
    result = downloader.download_file({'url': 'http://url', 'output_path': Path('file')})
    assert result is None

def test_download_file_interrupt_during(mocker, downloader, mock_stop_event, tmp_path):
    output_path = tmp_path / "file.zip"
    file_info = {'url': 'http://url', 'output_path': output_path}
    mock_get = mocker.patch('requests.get')
    def iter_content(chunk_size):
        mock_stop_event.is_set.return_value = True
        yield b'data'
    mock_get.return_value.iter_content = iter_content
    mock_get.return_value.raise_for_status = MagicMock()
    result = downloader.download_file(file_info)
    assert result is None
    assert not output_path.exists()

def test_download_file_failure(mocker, downloader, caplog, tmp_path):
    output_path = tmp_path / "file.zip"
    file_info = {'url': 'http://url', 'output_path': output_path}
    mocker.patch('requests.get', side_effect=RequestException("Fail"))
    result = downloader.download_file(file_info)
    assert result is None
    assert "Download failed" in caplog.text
    assert not output_path.exists()

def test_download_all_no_files(downloader):
    downloaded, failed = downloader.download_all([])
    assert downloaded == []
    assert failed == []

def test_download_all_success(mocker, downloader):
    file_info = {'file': 'info', 'output_path': Path('dummy/path.zip')}
    mocker.patch.object(downloader, 'download_file', return_value=file_info)
    mocker.patch('concurrent.futures.ThreadPoolExecutor')
    mocker.patch('concurrent.futures.as_completed', return_value=[MagicMock(result=lambda: file_info)])
    
    downloaded, failed = downloader.download_all([file_info])
    assert len(downloaded) == 1
    assert len(failed) == 0

def test_download_all_failure(mocker, downloader):
    file_info = {'file': 'info', 'output_path': Path('dummy/path.zip')}
    mocker.patch.object(downloader, 'download_file', return_value=None)
    mocker.patch('concurrent.futures.ThreadPoolExecutor')
    mocker.patch('concurrent.futures.as_completed', return_value=[MagicMock(result=lambda: None)])
    
    downloaded, failed = downloader.download_all([file_info])
    assert len(downloaded) == 0
    assert len(failed) == 1

def test_download_all_interrupted(mocker, downloader, mock_stop_event):
    mock_futures = [MagicMock(spec=Future) for _ in range(3)]
    mock_futures[0].result.return_value = {'path1'}
    mock_futures[1].result.return_value = {'path2'}
    mock_futures[2].result.return_value = {'path3'}

    mock_tpe = mocker.patch('gridflow.download.prism_downloader.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    mock_executor.submit.side_effect = mock_futures

    def mock_as_completed(fs):
        yield mock_futures[0]
        yield mock_futures[1]
        mock_stop_event.is_set.return_value = True
        yield mock_futures[2]

    mocker.patch('gridflow.download.prism_downloader.as_completed', side_effect=mock_as_completed)

    downloaded, failed = downloader.download_all([{}, {}, {}])
    assert len(downloaded) == 2
    assert len(failed) == 0

# ############################################################################
# Tests for create_download_session
# ############################################################################

def test_create_download_session_no_files(mocker, mock_stop_event, caplog):
    caplog.set_level(logging.INFO)
    settings = {'variable': 'ppt', 'resolution': '4km', 'time_step': 'daily', 'start_date': '2020-01-01', 'end_date': '2020-01-01', 'output_dir': './downloads', 'metadata_dir': './metadata', 'workers': 1, 'timeout': 30}
    mocker.patch.object(AvailabilityChecker, 'find_available_files', return_value=[])
    with pytest.raises(SystemExit):
        create_download_session(settings, mock_stop_event)
    assert "No files were found on the server" in caplog.text

def test_create_download_session_existing_files(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    settings = {'variable': 'ppt', 'resolution': '4km', 'time_step': 'daily', 'start_date': '2020-01-01', 'end_date': '2020-01-01', 'output_dir': str(tmp_path / "downloads"), 'metadata_dir': str(tmp_path / "metadata"), 'workers': 1, 'timeout': 30}
    available_files = [{'url': 'http://url', 'output_path': tmp_path / "downloads" / "ppt" / "4km" / "file.zip"}]
    (available_files[0]['output_path'].parent).mkdir(parents=True)
    available_files[0]['output_path'].touch()
    mocker.patch.object(AvailabilityChecker, 'find_available_files', return_value=available_files)
    create_download_session(settings, mock_stop_event)
    assert "1 files already exist" in caplog.text

def test_create_download_session_dry_run(mocker, mock_stop_event, caplog):
    caplog.set_level(logging.INFO)
    settings = {'dry_run': True, 'variable': 'ppt', 'resolution': '4km', 'time_step': 'daily', 'start_date': '2020-01-01', 'end_date': '2020-01-01', 'output_dir': './downloads', 'metadata_dir': './metadata', 'workers': 1, 'timeout': 30}
    mocker.patch.object(AvailabilityChecker, 'find_available_files', return_value=[{'url': 'http://url', 'output_path': Path('dummy')}])
    create_download_session(settings, mock_stop_event)
    assert "Dry run: Would attempt to download" in caplog.text

def test_create_download_session_success(mocker, mock_stop_event):
    settings = {'variable': 'ppt', 'resolution': '4km', 'time_step': 'daily', 'start_date': '2020-01-01', 'end_date': '2020-01-01', 'output_dir': './downloads', 'metadata_dir': './metadata', 'workers': 1, 'timeout': 30}
    available_files = [{'url': 'http://url', 'output_path': Path('./downloads/ppt/4km/file.zip')}]
    mocker.patch.object(AvailabilityChecker, 'find_available_files', return_value=available_files)
    mock_dl = mocker.patch.object(Downloader, 'download_all', return_value=([available_files[0]], []))
    create_download_session(settings, mock_stop_event)
    assert mock_dl.called

def test_create_download_session_exception(mocker, mock_stop_event, caplog):
    mocker.patch.object(AvailabilityChecker, 'find_available_files', side_effect=Exception("Critical"))
    settings = {'variable': 'ppt', 'resolution': '4km', 'time_step': 'daily', 'start_date': '2020-01-01', 'end_date': '2020-01-01', 'output_dir': './downloads', 'metadata_dir': './metadata', 'workers': 1, 'timeout': 30}
    create_download_session(settings, mock_stop_event)
    assert "critical error" in caplog.text
    assert mock_stop_event.set.called

# ############################################################################
# Tests for add_arguments
# ############################################################################

def test_add_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--variable', 'ppt', '--start-date', '2020-01-01', '--end-date', '2020-01-31', '--workers', '2'])
    assert args.variable == ['ppt'] 
    assert args.start_date == '2020-01-01'
    assert args.end_date == '2020-01-31'
    assert args.workers == 2
    assert args.resolution == '4km'
    assert args.output_dir == './downloads_prism'

# ############################################################################
# Tests for main
# ############################################################################

@patch('argparse.ArgumentParser')
@patch('gridflow.download.prism_downloader.setup_logging')
@patch('gridflow.download.prism_downloader.create_download_session')
@patch('signal.signal')
def test_main_success(mock_signal, mock_session, mock_logging, mock_parser):
    evt = threading.Event()

    mock_args = MagicMock(
        variable='ppt',
        start_date='2020-01-01',
        end_date='2020-01-31',
        log_dir='./logs',
        log_level='info',
    )
    mock_args.is_gui_mode = False
    mock_args.stop_event  = evt
    mock_args.config = None
    mock_args.demo = False

    mock_parser.return_value.parse_args.return_value = mock_args

    main()

    mock_session.assert_called_once()
    mock_signal.assert_called_with(signal.SIGINT, signal_handler)
    args_passed, kwargs_passed = mock_session.call_args
    assert args_passed[1] is evt

@patch('argparse.ArgumentParser')
@patch('gridflow.download.prism_downloader.setup_logging')
@patch('gridflow.download.prism_downloader.create_download_session')
def test_main_demo(mock_session, mock_logging, mock_parser):
    mock_args = MagicMock(demo=True, log_dir='./logs', log_level='info', config=None)
    mock_parser.return_value.parse_args.return_value = mock_args
    main()
    mock_session.assert_called()

@patch('argparse.ArgumentParser')
@patch('gridflow.download.prism_downloader.setup_logging')
def test_main_no_params(mock_logging, mock_parser, caplog):
    mock_args            = MagicMock()
    mock_args.demo       = False
    mock_args.variable   = None
    mock_args.start_date = None
    mock_args.end_date   = None
    mock_args.is_gui_mode = False
    mock_args.config     = None

    mock_parser.return_value.parse_args.return_value = mock_args

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "required when not in --demo mode" in caplog.text

@patch('argparse.ArgumentParser')
@patch('gridflow.download.prism_downloader.setup_logging')
@patch('gridflow.download.prism_downloader.create_download_session')
def test_main_interrupted(mock_session, mock_logging, mock_parser, caplog):
    evt = threading.Event()
    evt.set()

    mock_args = MagicMock(
        variable='ppt',
        start_date='2020-01-01',
        end_date='2020-01-31',
        log_dir='./logs',
        log_level='info',
    )
    mock_args.is_gui_mode = False
    mock_args.stop_event  = evt
    mock_args.config = None
    mock_args.demo = False

    mock_parser.return_value.parse_args.return_value = mock_args

    mock_session.side_effect = lambda *a, **k: None

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 130
    assert "Execution was interrupted." in caplog.text