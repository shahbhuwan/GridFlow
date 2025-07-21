# tests/test_cmip5_downloader.py

import argparse
import json
import logging
import signal
import sys
import threading
from urllib.parse import parse_qs
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import requests
from requests.exceptions import RequestException

import gridflow.download.cmip5_downloader as cmip5_downloader
from gridflow.download.cmip5_downloader import (
    ESGF_NODES,
    FileManager,
    InterruptibleSession,
    QueryHandler,
    Downloader,
    create_download_session,
    load_config,
    signal_handler,
    stop_event,
    add_arguments,
    main,
)

# ############################################################################
# Fixtures
# ############################################################################

@pytest.fixture
def mock_session(mocker):
    return mocker.patch('requests.Session')

@pytest.fixture
def mock_stop_event():
    event = MagicMock(spec=threading.Event)
    event.is_set.return_value = False
    return event

@pytest.fixture
def file_manager(tmp_path):
    return FileManager(str(tmp_path / "downloads"), str(tmp_path / "metadata"), "structured")

@pytest.fixture
def query_handler(mock_stop_event):
    return QueryHandler(ESGF_NODES, mock_stop_event)

@pytest.fixture
def downloader(file_manager, mock_stop_event):
    return Downloader(file_manager, mock_stop_event, workers=1, no_verify_ssl=False)

# ############################################################################
# Tests for signal_handler
# ############################################################################

def test_signal_handler(caplog, mocker):
    """
    Tests the signal_handler function.
    """
    caplog.set_level(logging.INFO)
    mock_event = mocker.patch('gridflow.download.cmip5_downloader.stop_event')
    signal_handler(None, None)
    assert mock_event.set.called
    assert "Stop signal received" in caplog.text
    assert "Please wait for ongoing tasks" in caplog.text

# ############################################################################
# Tests for InterruptibleSession
# ############################################################################

def test_interruptible_session_init(mock_session, mock_stop_event):
    session = InterruptibleSession(mock_stop_event, cert_path="/path/to/cert")
    assert session.stop_event == mock_stop_event
    assert session.cert == "/path/to/cert"
    assert isinstance(session.adapters['http://'], requests.adapters.HTTPAdapter)
    assert isinstance(session.adapters['https://'], requests.adapters.HTTPAdapter)

def test_interruptible_session_get_interrupted(mock_stop_event):
    mock_stop_event.is_set.return_value = True
    session = InterruptibleSession(mock_stop_event)
    with pytest.raises(RequestException, match="Download interrupted by user."):
        session.get("http://example.com")

def test_interruptible_session_get_success(mock_stop_event, mocker):
    mock_super_get = mocker.patch('requests.Session.get')
    session = InterruptibleSession(mock_stop_event, cert_path="/cert")
    session.get("http://example.com", timeout=(5, 5))
    mock_super_get.assert_called_with("http://example.com", timeout=(5, 5), cert="/cert")

# ############################################################################
# Tests for FileManager
# ############################################################################

def test_file_manager_init(tmp_path):
    fm = FileManager(str(tmp_path / "downloads"), str(tmp_path / "metadata"), "flat", prefix="test_", metadata_prefix="meta_")
    assert fm.download_dir.exists()
    assert fm.metadata_dir.exists()
    assert fm.save_mode == "flat"
    assert fm.prefix == "test_"
    assert fm.metadata_prefix == "meta_"

def test_file_manager_init_error(mocker, caplog):
    mocker.patch('pathlib.Path.mkdir', side_effect=OSError("Permission denied"))
    with pytest.raises(SystemExit):
        FileManager("/invalid", "/invalid", "structured")
    assert "Failed to create directories" in caplog.text

def test_get_output_path_structured(file_manager):
    file_info = {
        'title': 'file.nc',
        'variable': ['tas'],
        'model': ['CanCM4'],
        'experiment': ['historical'],
        'ensemble': ['r1i1p1']
    }
    path = file_manager.get_output_path(file_info)
    expected = file_manager.download_dir / 'tas' / 'CanCM4' / 'historical' / 'r1i1p1' / 'file.nc'
    assert path == expected
    assert path.parent.exists()

def test_get_output_path_flat(file_manager):
    file_manager.save_mode = "flat"
    file_manager.prefix = "prefix_"
    file_info = {
        'title': 'file.nc',
        'model': ['CanCM4'],
        'experiment': ['historical']
    }
    path = file_manager.get_output_path(file_info)
    expected = file_manager.download_dir / 'prefix_CanCM4_historical_file.nc'
    assert path == expected

def test_get_output_path_defaults(file_manager):
    file_info = {'title': 'file.nc'}
    path = file_manager.get_output_path(file_info)
    expected = file_manager.download_dir / 'unknown' / 'unknown' / 'unknown' / 'unknown' / 'file.nc'
    assert path == expected

def test_save_metadata(file_manager, tmp_path):
    files = [{'id': 1}]
    file_manager.save_metadata(files, "test.json")
    metadata_path = file_manager.metadata_dir / "test.json"
    assert metadata_path.exists()
    with open(metadata_path, 'r') as f:
        assert json.load(f) == files

def test_save_metadata_error(mocker, file_manager, caplog):
    mocker.patch('builtins.open', side_effect=IOError("Write error"))
    file_manager.save_metadata([], "test.json")
    assert "Failed to save metadata" in caplog.text

# ############################################################################
# Tests for QueryHandler
# ############################################################################

def test_query_handler_init(mock_stop_event):
    qh = QueryHandler(["node1"], mock_stop_event)
    assert qh.nodes == ["node1"]
    assert qh._stop_event == mock_stop_event
    assert isinstance(qh.session, InterruptibleSession)

def test_build_query(query_handler):
    """
    Tests the build_query method.
    """
    url = query_handler.build_query("https://base", {'model': 'CanCM4'})
    base, query_string = url.split('?')
    assert base == "https://base"
    parsed_query = parse_qs(query_string)
    expected = {
        'type': ['File'],
        'project': ['CMIP5'],
        'format': ['application/solr+json'],
        'limit': ['1000'],
        'distrib': ['true'],
        'model': ['CanCM4']
    }
    assert parsed_query == expected

def test_fetch_datasets_success(mocker, query_handler):
    mock_fetch_node = mocker.patch.object(query_handler, '_fetch_from_node', return_value=[{'id': '1'}])
    files = query_handler.fetch_datasets({'model': 'CanCM4'}, 30)
    assert files == [{'id': '1'}]
    mock_fetch_node.assert_called_once()

def test_fetch_datasets_multiple_nodes_unique(mocker, query_handler):
    mock_fetch_node = mocker.patch.object(query_handler, '_fetch_from_node')
    mock_fetch_node.side_effect = [[{'id': '1'}], [{'id': '1'}, {'id': '2'}]]
    files = query_handler.fetch_datasets({}, 30)
    assert len(files) == 1  # Only from first node since all_files not empty
    assert files[0]['id'] == '1'

def test_fetch_datasets_interrupted(mock_stop_event, query_handler):
    mock_stop_event.is_set.return_value = True
    files = query_handler.fetch_datasets({}, 30)
    assert files == []

def test_fetch_datasets_timeout(mocker, query_handler, caplog):
    mocker.patch.object(query_handler, '_fetch_from_node', side_effect=requests.Timeout)
    files = query_handler.fetch_datasets({}, 30)
    assert "timed out" in caplog.text
    assert files == []

def test_fetch_datasets_connection_error(mocker, query_handler, caplog):
    mocker.patch.object(query_handler, '_fetch_from_node', side_effect=requests.ConnectionError)
    files = query_handler.fetch_datasets({}, 30)
    assert "Could not connect" in caplog.text
    assert files == []

def test_fetch_datasets_request_exception(mocker, query_handler, caplog):
    mocker.patch.object(query_handler, '_fetch_from_node', side_effect=requests.RequestException("Error"))
    files = query_handler.fetch_datasets({}, 30)
    assert "An error occurred" in caplog.text
    assert files == []

def test_fetch_datasets_unexpected_error(mocker, query_handler, caplog):
    mocker.patch.object(query_handler, '_fetch_from_node', side_effect=Exception("Unexpected"))
    files = query_handler.fetch_datasets({}, 30)
    assert "unexpected error" in caplog.text
    assert files == []

def test_fetch_datasets_all_fail(query_handler, caplog):
    query_handler.nodes = []
    files = query_handler.fetch_datasets({}, 30)
    assert "All nodes failed" in caplog.text
    assert files == []

def test__fetch_from_node_success(mocker, query_handler):
    mock_get = mocker.patch.object(query_handler.session, 'get')
    mock_get.return_value.json.return_value = {
        'response': {'docs': [{'id': '1'}], 'numFound': 1}
    }
    files = query_handler._fetch_from_node("node", {}, 30)
    assert files == [{'id': '1'}]

def test__fetch_from_node_pagination(mocker, query_handler):
    mock_get = mocker.patch.object(query_handler.session, 'get')
    mock_get.side_effect = [
        MagicMock(json=lambda: {'response': {'docs': [{'id': '1'}], 'numFound': 2}}),
        MagicMock(json=lambda: {'response': {'docs': [{'id': '2'}], 'numFound': 2}})
    ]
    files = query_handler._fetch_from_node("node", {}, 30)
    assert len(files) == 2

def test__fetch_from_node_no_docs(mocker, query_handler):
    mock_get = mocker.patch.object(query_handler.session, 'get')
    mock_get.return_value.json.return_value = {'response': {'docs': [], 'numFound': 0}}
    files = query_handler._fetch_from_node("node", {}, 30)
    assert files == []

def test__fetch_from_node_interrupted(mock_stop_event, query_handler):
    mock_stop_event.is_set.return_value = True
    files = query_handler._fetch_from_node("node", {}, 30)
    assert files == []

def test__fetch_from_node_http_error(mocker, query_handler):
    mock_get = mocker.patch.object(query_handler.session, 'get')
    mock_get.return_value.raise_for_status.side_effect = requests.HTTPError("404")
    with pytest.raises(requests.HTTPError):
        query_handler._fetch_from_node("node", {}, 30)

# ############################################################################
# Tests for Downloader
# ############################################################################

def test_downloader_init_no_auth(file_manager, mock_stop_event):
    dl = Downloader(file_manager, mock_stop_event, workers=2)
    assert dl.file_manager == file_manager
    assert dl._stop_event == mock_stop_event
    assert dl.settings['workers'] == 2
    assert dl.cert_path is None
    assert isinstance(dl.session, InterruptibleSession)
    assert dl.session.auth is None

def test_downloader_init_with_basic_auth(file_manager, mock_stop_event):
    dl = Downloader(file_manager, mock_stop_event, username="user", password="pass")
    assert dl.session.auth == ("user", "pass")

def test_downloader_init_with_openid(mocker, file_manager, mock_stop_event):
    mock_fetch_cert = mocker.patch.object(Downloader, '_fetch_esgf_certificate')
    Downloader(file_manager, mock_stop_event, openid="https://openid", username="user", password="pass")
    mock_fetch_cert.assert_called_with("https://openid", "user", "pass")

def test__fetch_esgf_certificate_existing(mocker, downloader, caplog, tmp_path):
    """
    Tests fetching certificate when existing valid one is present.
    """
    caplog.set_level(logging.DEBUG)
    mocker.patch('pathlib.Path.home', return_value=tmp_path)
    cert_dir = tmp_path / ".esg"
    cert_dir.mkdir()
    cert_path = cert_dir / "credentials.pem"
    cert_path.write_text("BEGIN CERTIFICATE")
    downloader._fetch_esgf_certificate("openid", "user", "pass")
    assert downloader.cert_path == cert_path
    assert "Using existing certificate" in caplog.text

def test__fetch_esgf_certificate_invalid_existing(mocker, downloader, tmp_path):
    """
    Tests fetching new certificate when existing is invalid.
    """
    mocker.patch('pathlib.Path.home', return_value=tmp_path)
    cert_dir = tmp_path / ".esg"
    cert_dir.mkdir()
    cert_path = cert_dir / "credentials.pem"
    cert_path.write_text("INVALID")
    mock_post = mocker.patch('requests.post')
    downloader._fetch_esgf_certificate("openid", "user", "pass")
    mock_post.assert_called()

def test__fetch_esgf_certificate_success(mocker, downloader, tmp_path):
    """
    Tests successful fetching of ESGF certificate.
    """
    mocker.patch('pathlib.Path.home', return_value=tmp_path)
    cert_dir = tmp_path / ".esg"
    cert_dir.mkdir(exist_ok=True)
    cert_path = cert_dir / "credentials.pem"
    # Simulate no existing cert
    mock_post = mocker.patch('requests.post')
    mock_post.return_value.text = "BEGIN CERTIFICATE content"
    mock_post.return_value.raise_for_status = MagicMock()
    mock_open = mocker.mock_open()
    mocker.patch('builtins.open', mock_open)
    downloader._fetch_esgf_certificate("https://base/openid", "user", "pass")
    mock_post.assert_called_with("https://base/esgf-idp/openid/", data={"openid": "https://base/openid", "username": "user", "password": "pass"}, timeout=30)
    mock_open.assert_called_with(cert_path, 'w')

def test__fetch_esgf_certificate_no_cert_in_response(mocker, downloader, caplog, tmp_path):
    """
    Tests when no valid cert in response.
    """
    mocker.patch('pathlib.Path.home', return_value=tmp_path)
    cert_dir = tmp_path / ".esg"
    cert_dir.mkdir(exist_ok=True)
    mock_post = mocker.patch('requests.post')
    mock_post.return_value.text = "No cert"
    mock_post.return_value.raise_for_status = MagicMock()
    downloader._fetch_esgf_certificate("openid", "user", "pass")
    assert downloader.cert_path is None
    assert "No valid certificate" in caplog.text

def test__fetch_esgf_certificate_request_error(mocker, downloader, caplog, tmp_path):
    """
    Tests request exception during cert fetch.
    """
    mocker.patch('pathlib.Path.home', return_value=tmp_path)
    cert_dir = tmp_path / ".esg"
    cert_dir.mkdir(exist_ok=True)
    mocker.patch('requests.post', side_effect=requests.RequestException("Network error"))
    downloader._fetch_esgf_certificate("openid", "user", "pass")
    assert downloader.cert_path is None
    assert "Failed to fetch ESGF certificate" in caplog.text

def test__fetch_esgf_certificate_unexpected_error(mocker, downloader, caplog, tmp_path):
    """
    Tests unexpected exception during cert fetch.
    """
    mocker.patch('pathlib.Path.home', return_value=tmp_path)
    cert_dir = tmp_path / ".esg"
    cert_dir.mkdir(exist_ok=True)
    mocker.patch('requests.post', side_effect=Exception("Unexpected"))
    downloader._fetch_esgf_certificate("openid", "user", "pass")
    assert downloader.cert_path is None
    assert "Unexpected error during certificate fetch" in caplog.text

def test_shutdown(mocker, downloader):
    mock_executor = mocker.Mock()
    downloader.executor = mock_executor
    mock_session_close = mocker.patch.object(downloader.session, 'close')
    downloader.shutdown()
    mock_executor.shutdown.assert_called_with(wait=True, cancel_futures=True)
    mock_session_close.assert_called()
    assert downloader.executor is None

def test_verify_checksum_no_checksum(downloader):
    assert downloader.verify_checksum(Path("file"), {}) is True

def test_verify_checksum_md5_success(mocker, downloader):
    file_info = {'checksum': ['8d777f385d3dfec8815d20f7496026dc'], 'checksum_type': ['MD5']}
    mock_open = mocker.mock_open(read_data=b'data')
    mocker.patch('builtins.open', mock_open)
    mock_md5 = mocker.patch('gridflow.download.cmip5_downloader.md5')
    mock_hasher = mocker.Mock()
    mock_hasher.update = mocker.Mock()
    mock_hasher.hexdigest = lambda: '8d777f385d3dfec8815d20f7496026dc'
    mock_md5.return_value = mock_hasher
    assert downloader.verify_checksum(Path("file"), file_info) is True

def test_verify_checksum_sha256_mismatch(mocker, downloader, caplog):
    file_info = {'checksum': ['abc'], 'checksum_type': ['SHA256']}
    mock_open = mocker.mock_open(read_data=b'data')
    mocker.patch('builtins.open', mock_open)
    mocker.patch('hashlib.sha256', return_value=MagicMock(hexdigest=lambda: 'def'))
    assert downloader.verify_checksum(Path("file"), file_info) is False
    assert "Checksum mismatch" in caplog.text

def test_verify_checksum_error(mocker, downloader, caplog):
    file_info = {'checksum': ['abc'], 'checksum_type': ['SHA256']}
    mocker.patch('builtins.open', side_effect=Exception("Read error"))
    assert downloader.verify_checksum(Path("file"), file_info) is False
    assert "Checksum verification failed" in caplog.text

def test_download_file_interrupted_before(mock_stop_event, downloader):
    mock_stop_event.is_set.return_value = True
    path, failed_info = downloader.download_file({'title': 'file.nc'})
    assert path is None
    assert failed_info['title'] == 'file.nc'

def test_download_file_exists_checksum_ok(mocker, downloader):
    mock_path = mocker.Mock(exists=lambda: True)
    mocker.patch.object(downloader.file_manager, 'get_output_path', return_value=mock_path)
    mocker.patch.object(downloader, 'verify_checksum', return_value=True)
    path, failed = downloader.download_file({'title': 'file.nc'})
    assert path == str(mock_path)
    assert failed is None

def test_download_file_no_httpserver_url(downloader, caplog):
    path, failed = downloader.download_file({'title': 'file.nc', 'url': []})
    assert path is None
    assert failed['title'] == 'file.nc'
    assert "No 'HTTPServer' URL" in caplog.text

def test_download_file_success(mocker, downloader, tmp_path):
    output_path = tmp_path / "file.nc"
    mocker.patch.object(downloader.file_manager, 'get_output_path', return_value=output_path)
    mocker.patch.object(downloader, 'verify_checksum', return_value=True)
    mock_get = mocker.patch.object(downloader.session, 'get')
    mock_get.return_value.iter_content.return_value = [b'data']
    mock_get.return_value.raise_for_status = MagicMock()
    path, failed = downloader.download_file({'title': 'file.nc', 'url': ['http://url|HTTP|HTTPServer']})
    assert path == str(output_path)
    assert failed is None
    assert output_path.exists()

def test_download_file_interrupt_during_download(mocker, downloader, mock_stop_event, tmp_path, caplog):
    output_path = tmp_path / "file.nc"
    mocker.patch.object(downloader.file_manager, 'get_output_path', return_value=output_path)
    mock_get = mocker.patch.object(downloader.session, 'get')
    def iter_content(chunk_size):
        yield b'data1'
        mock_stop_event.is_set.return_value = True
        yield b'data2'
    mock_get.return_value.iter_content = iter_content
    mock_get.return_value.raise_for_status = MagicMock()
    path, failed = downloader.download_file({'title': 'file.nc', 'url': ['http://url|HTTP|HTTPServer']})
    assert path is None
    assert failed is not None
    assert "interrupted by user" in caplog.text
    assert not (tmp_path / "file.nc.tmp").exists()

def test_download_file_failure(mocker, downloader, caplog, tmp_path):
    output_path = tmp_path / "file.nc"
    mocker.patch.object(downloader.file_manager, 'get_output_path', return_value=output_path)
    mock_get = mocker.patch.object(downloader.session, 'get', side_effect=RequestException("Fail"))
    path, failed = downloader.download_file({'title': 'file.nc', 'url': ['http://url|HTTP|HTTPServer']})
    assert path is None
    assert failed['title'] == 'file.nc'
    assert "Download failed" in caplog.text

def test_download_all_no_files(downloader):
    downloaded, failed = downloader.download_all([])
    assert downloaded == []
    assert failed == []

def test_download_all_success(mocker, downloader):
    mocker.patch.object(downloader, 'download_file', return_value=("path", None))
    mocker.patch('concurrent.futures.ThreadPoolExecutor')
    mocker.patch('concurrent.futures.as_completed', return_value=[MagicMock(result=lambda: ("path", None))])
    downloaded, failed = downloader.download_all([{'title': 'file.nc', 'url': ['http://url|HTTP|HTTPServer']}])
    assert len(downloaded) == 1
    assert len(failed) == 0

def test_download_all_failure(mocker, downloader):
    mock_future = MagicMock()
    mock_future.result.side_effect = Exception("Error")
    mocker.patch('concurrent.futures.ThreadPoolExecutor', return_value=MagicMock(submit=lambda f, *a: mock_future))
    mocker.patch('concurrent.futures.as_completed', return_value=[mock_future])
    downloaded, failed = downloader.download_all([{'title': 'file'}])
    assert len(downloaded) == 0
    assert len(failed) == 1

def test_download_all_interrupted(mocker, downloader, mock_stop_event):
    mock_futures = [MagicMock(spec=Future) for _ in range(3)]
    mock_futures[0].result.return_value = ("path1", None)
    mock_futures[1].result.return_value = ("path2", None)
    mock_futures[2].result.return_value = ("path3", None)

    mock_tpe = mocker.patch('gridflow.download.cmip5_downloader.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    mock_executor.submit.side_effect = mock_futures

    def mock_as_completed(fs):
        yield mock_futures[0]
        yield mock_futures[1]
        mock_stop_event.is_set.return_value = True
        yield mock_futures[2]

    mocker.patch('gridflow.download.cmip5_downloader.as_completed', side_effect=mock_as_completed)

    downloaded, failed = downloader.download_all([{}, {}, {}])
    assert len(downloaded) == 2
    assert len(failed) == 0

# ############################################################################
# Tests for load_config
# ############################################################################

def test_load_config_success(tmp_path):
    config_path = tmp_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump({'key': 'value'}, f)
    config = load_config(str(config_path))
    assert config == {'key': 'value'}

def test_load_config_not_found(caplog):
    with pytest.raises(SystemExit):
        load_config("nonexistent.json")
    assert "Failed to load config" in caplog.text

def test_load_config_json_error(mocker, caplog):
    mocker.patch('builtins.open', mocker.mock_open(read_data="invalid json"))
    with pytest.raises(SystemExit):
        load_config("file.json")
    assert "Failed to load config" in caplog.text

# ############################################################################
# Tests for create_download_session
# ############################################################################

def test_create_download_session_retry_mode(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    retry_path = tmp_path / "failed.json"
    with open(retry_path, 'w') as f:
        json.dump([{'title': 'file'}], f)
    mocker.patch('gridflow.download.cmip5_downloader.load_config', return_value=[{'title': 'file'}])
    mock_fm = mocker.patch('gridflow.download.cmip5_downloader.FileManager')
    mock_dl = mocker.patch('gridflow.download.cmip5_downloader.Downloader')
    mock_dl.return_value.download_all.return_value = (["path"], [])
    settings = {'retry_failed_path': str(retry_path), 'output_dir': 'dir', 'metadata_dir': 'meta', 'save_mode': 'flat'}
    create_download_session({}, settings, mock_stop_event)
    assert "Retrying 1 files" in caplog.text

def test_create_download_session_retry_empty(mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('gridflow.download.cmip5_downloader.load_config', return_value=[])
    with pytest.raises(SystemExit):
        create_download_session({}, {'retry_failed_path': 'empty.json'}, stop_event)
    assert "Retry file is empty" in caplog.text

def test_create_download_session_no_params(caplog):
    with pytest.raises(SystemExit):
        create_download_session({}, {}, stop_event)
    assert "No search parameters" in caplog.text

def test_create_download_session_demo(mocker, mock_stop_event):
    settings = {'demo': True}
    mocker.patch.object(QueryHandler, 'fetch_datasets', return_value=[{'title': 'file'}])
    mocker.patch('gridflow.download.cmip5_downloader.FileManager')
    create_download_session({}, settings, mock_stop_event)

def test_create_download_session_dry_run(mocker, mock_stop_event, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch.object(QueryHandler, 'fetch_datasets', return_value=[{'title': 'file'}])
    mocker.patch('gridflow.download.cmip5_downloader.FileManager')
    settings = {'dry_run': True, 'timeout': 30, 'output_dir': './downloads', 'metadata_dir': './metadata', 'save_mode': 'structured'}
    create_download_session({'project': 'CMIP5'}, settings, mock_stop_event)
    assert "Dry run: Would attempt to download" in caplog.text

def test_create_download_session_success(mocker, mock_stop_event):
    mocker.patch.object(QueryHandler, 'fetch_datasets', return_value=[{'title': 'file1'}, {'title': 'file1'}, {'title': 'file2'}])
    mock_fm = mocker.patch('gridflow.download.cmip5_downloader.FileManager')
    mock_dl = mocker.patch('gridflow.download.cmip5_downloader.Downloader')
    mock_dl.return_value.download_all.return_value = (["path"], [{'title': 'failed'}])
    settings = {'max_downloads': 1, 'output_dir': 'dir', 'metadata_dir': 'meta', 'save_mode': 'flat', 'timeout': 30}
    create_download_session({'project': 'CMIP5'}, settings, mock_stop_event)
    assert mock_fm.return_value.save_metadata.call_count == 2

def test_create_download_session_exception(mocker, mock_stop_event, caplog):
    mocker.patch.object(QueryHandler, 'fetch_datasets', side_effect=Exception("Critical"))
    create_download_session({'project': 'CMIP5'}, {}, mock_stop_event)
    assert "critical error" in caplog.text
    assert mock_stop_event.set.called

# ############################################################################
# Tests for add_arguments
# ############################################################################

def test_add_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--model', 'CanCM4', '--variable', 'tas', '--workers', '2'])
    assert args.model == 'CanCM4'
    assert args.variable == 'tas'
    assert args.workers == 2
    assert args.project == 'CMIP5'
    assert args.output_dir == './downloads_cmip5'

# ############################################################################
# Tests for main
# ############################################################################

@patch('argparse.ArgumentParser')
@patch('gridflow.download.cmip5_downloader.setup_logging')
@patch('gridflow.download.cmip5_downloader.create_download_session')
@patch('signal.signal')
def test_main_success(mock_signal, mock_session, mock_logging, mock_parser):
    mock_args = MagicMock(config=None, demo=False, project='CMIP5', model='CanCM4', log_dir='./logs', log_level='info')
    mock_parser.return_value.parse_args.return_value = mock_args
    main()
    mock_session.assert_called()
    mock_signal.assert_called_with(signal.SIGINT, signal_handler)

# Replacement for test_main_demo
@patch('argparse.ArgumentParser')
@patch('gridflow.download.cmip5_downloader.setup_logging')
@patch('gridflow.download.cmip5_downloader.create_download_session')
def test_main_demo(mock_session, mock_logging, mock_parser):
    mock_args = MagicMock(demo=True, config=None, log_dir='./logs', log_level='info')
    mock_parser.return_value.parse_args.return_value = mock_args
    main()
    mock_session.assert_called()

@patch('argparse.ArgumentParser')
@patch('gridflow.download.cmip5_downloader.setup_logging')
@patch('gridflow.download.cmip5_downloader.load_config', return_value={})
def test_main_config(mock_load, mock_logging, mock_parser, caplog):
    mock_parser.return_value.parse_args.return_value = MagicMock(config='config.json', demo=False)
    with pytest.raises(SystemExit):
        main()
    assert "No search parameters" in caplog.text

@patch('argparse.ArgumentParser')
@patch('gridflow.download.cmip5_downloader.setup_logging')
def test_main_openid_validation(mock_logging, mock_parser, caplog):
    mock_parser.return_value.parse_args.return_value = MagicMock(openid='openid', id=None, password='pass', config=None, demo=False)
    with pytest.raises(SystemExit):
        main()
    assert "Both --id and --password are required" in caplog.text

@patch('argparse.ArgumentParser')
@patch('gridflow.download.cmip5_downloader.setup_logging')
@patch('gridflow.download.cmip5_downloader.stop_event')
@patch('gridflow.download.cmip5_downloader.create_download_session')
def test_main_interrupted(mock_session, mock_stop, mock_logging, mock_parser, caplog):
    caplog.set_level(logging.WARNING)
    mock_stop.is_set.return_value = True
    mock_args = MagicMock(
        config=None,
        demo=False,
        project='CMIP5',
        model='CanCM4',
        log_dir='./logs',
        log_level='info',
        timeout=30,
        output_dir='./downloads',
        metadata_dir='./metadata',
        save_mode='structured'
    )
    mock_parser.return_value.parse_args.return_value = mock_args
    mock_session.side_effect = lambda params, settings, stop_event: None  # Simulate session stopping early
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 130
    assert "Execution was interrupted" in caplog.text