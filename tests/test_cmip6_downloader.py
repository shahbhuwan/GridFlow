# tests/test_cmip6_downloader.py

import argparse
import json
import logging
import signal
import threading
from pathlib import Path
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest
import requests
from requests.exceptions import RequestException

from gridflow.download.cmip6_downloader import (
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
def query_handler(mock_stop_event, mocker):
    mocker.patch.object(QueryHandler, '_get_available_nodes', return_value=["node1"])
    qh = QueryHandler(stop_event=mock_stop_event)
    return qh


@pytest.fixture
def downloader(file_manager, mock_stop_event):
    return Downloader(file_manager, mock_stop_event, workers=1, no_verify_ssl=False)

# ############################################################################
# Tests for signal_handler
# ############################################################################

def test_signal_handler(caplog, mocker):
    caplog.set_level(logging.INFO)
    mock_event = mocker.patch('gridflow.download.cmip6_downloader.stop_event')
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
        'activity_id': ['HighResMIP'],
        'variable_id': ['tas'],
        'nominal_resolution': ['250 km']
    }
    path = file_manager.get_output_path(file_info)
    expected = file_manager.download_dir / 'tas' / '250km' / 'HighResMIP' / 'file.nc'
    assert path == expected
    assert path.parent.exists()

def test_get_output_path_flat(file_manager):
    file_manager.save_mode = "flat"
    file_manager.prefix = "prefix_"
    file_info = {
        'title': 'file.nc',
        'activity_id': ['HighResMIP'],
        'nominal_resolution': ['250 km']
    }
    path = file_manager.get_output_path(file_info)
    expected = file_manager.download_dir / 'prefix_HighResMIP_250km_file.nc'
    assert path == expected

def test_get_output_path_defaults(file_manager):
    file_info = {'title': 'file.nc'}
    path = file_manager.get_output_path(file_info)
    expected = file_manager.download_dir / 'unknown' / 'unknown_resolution' / 'unknown' / 'file.nc'
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

def test_query_handler_init(mock_stop_event, mocker):
    mocker.patch.object(QueryHandler, '_get_available_nodes', return_value=["node1"])
    qh = QueryHandler(stop_event=mock_stop_event)
    assert qh.nodes == ["node1"]
    assert qh._stop_event == mock_stop_event
    assert isinstance(qh.session, InterruptibleSession)

def test_build_query(query_handler):
    url = query_handler.build_query("https://base", {'variable_id': 'tas'})
    expected = "https://base?type=File&project=CMIP6&format=application/solr%2Bjson&limit=1000&distrib=true&variable_id=tas"
    assert url == expected

def test_fetch_datasets_success(mocker, query_handler):
    import threading
    query_handler._stop_event = threading.Event()
    query_handler.nodes = ["node1"]

    mocker.patch.object(
        query_handler,
        '_fetch_from_node',
        return_value=[{'instance_id': 'X.1', 'url': ['http://n1/file1']}]
    )
    files = query_handler.fetch_datasets({'variable_id': 'tas'}, 30)
    assert files == [{'instance_id': 'X.1', 'url': ['http://n1/file1']}]


def test_fetch_datasets_multiple_nodes_unique(mocker, query_handler):
    import threading
    query_handler._stop_event = threading.Event()
    query_handler.nodes = ["node1", "node2"]

    mock_fetch_node = mocker.patch.object(query_handler, '_fetch_from_node')
    mock_fetch_node.side_effect = [
        [{'instance_id': 'X.1', 'url': ['http://n1/file1']}],
        [{'instance_id': 'X.1', 'url': ['http://n2/file1']},
         {'instance_id': 'X.2', 'url': ['http://n2/file2']}]
    ]

    files = query_handler.fetch_datasets({}, 30)
    assert {'instance_id': 'X.1', 'url': ['http://n1/file1', 'http://n2/file1']} in files or \
           {'instance_id': 'X.1', 'url': ['http://n2/file1', 'http://n1/file1']} in files
    assert {'instance_id': 'X.2', 'url': ['http://n2/file2']} in files
    assert len(files) == 2

def test_fetch_datasets_interrupted(mock_stop_event, query_handler):
    mock_stop_event.is_set.return_value = True
    files = query_handler.fetch_datasets({}, 30)
    assert files == []

def test_fetch_datasets_timeout(mocker, query_handler, caplog):
    caplog.set_level(logging.DEBUG)  
    query_handler._stop_event = threading.Event()
    query_handler.nodes = ["node1"]

    mocker.patch.object(query_handler, '_fetch_from_node', side_effect=requests.Timeout)
    files = query_handler.fetch_datasets({}, 30)
    assert files == []
    assert "Error querying node" in caplog.text

def test_fetch_datasets_connection_error(mocker, query_handler, caplog):
    caplog.set_level(logging.DEBUG)  
    query_handler._stop_event = threading.Event()
    query_handler.nodes = ["node1"]

    mocker.patch.object(query_handler, '_fetch_from_node', side_effect=requests.ConnectionError)
    files = query_handler.fetch_datasets({}, 30)
    assert files == []
    assert "Error querying node" in caplog.text

def test_fetch_datasets_request_exception(mocker, query_handler, caplog):
    caplog.set_level(logging.DEBUG) 
    mocker.patch.object(query_handler, '_fetch_from_node', side_effect=requests.RequestException("Error"))
    files = query_handler.fetch_datasets({}, 30)
    assert "Error querying node" in caplog.text
    assert files == []

def test_fetch_datasets_unexpected_error(mocker, query_handler, caplog):
    caplog.set_level(logging.DEBUG) 
    import threading
    query_handler._stop_event = threading.Event()
    query_handler.nodes = ["node1"]

    mocker.patch.object(query_handler, '_fetch_from_node', side_effect=Exception("Unexpected"))
    files = query_handler.fetch_datasets({}, 30)
    assert files == []
    assert "Error querying node" in caplog.text

def test_fetch_datasets_all_fail(query_handler, caplog):
    import threading
    query_handler._stop_event = threading.Event()
    query_handler.nodes = []
    files = query_handler.fetch_datasets({}, 30)
    assert files == []
    assert "No ESGF nodes available to query." in caplog.text

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
    mocker.patch('pathlib.Path.home', return_value=tmp_path)
    cert_dir = tmp_path / ".esg"
    cert_dir.mkdir()
    cert_path = cert_dir / "credentials.pem"
    cert_path.write_text("INVALID")
    mock_post = mocker.patch('requests.post')
    downloader._fetch_esgf_certificate("openid", "user", "pass")
    mock_post.assert_called()

def test__fetch_esgf_certificate_success(mocker, downloader, tmp_path):
    from unittest.mock import MagicMock
    # Don't touch real home
    mocker.patch('pathlib.Path.home', return_value=tmp_path)
    cert_dir = tmp_path / ".esg"
    cert_dir.mkdir(exist_ok=True)
    cert_path = cert_dir / "credentials.pem"

    # Simulate no existing cert
    mock_post = mocker.patch('requests.post')
    mock_post.return_value.text = "BEGIN CERTIFICATE content"
    mock_post.return_value.raise_for_status = MagicMock()

    mopen = mocker.mock_open()
    mocker.patch('builtins.open', mopen)

    downloader._fetch_esgf_certificate("https://base/openid", "user", "pass")

    # New endpoint + payload
    mock_post.assert_called_with(
        "https://base/esgf-idp/openid/login",
        data={"openid_identifier": "https://base/openid", "username": "user", "password": "pass"},
        timeout=30
    )
    # NEW: Writer is opened without explicit encoding in the updated code
    mopen.assert_called_with(cert_path, "w")
    handle = mopen()
    handle.write.assert_called_once_with("BEGIN CERTIFICATE content")

def test__fetch_esgf_certificate_no_cert_in_response(mocker, downloader, caplog, tmp_path):
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
    mocker.patch('pathlib.Path.home', return_value=tmp_path)
    cert_dir = tmp_path / ".esg"
    cert_dir.mkdir(exist_ok=True)
    mocker.patch('requests.post', side_effect=requests.RequestException("Network error"))
    downloader._fetch_esgf_certificate("openid", "user", "pass")
    assert downloader.cert_path is None
    assert "Failed to fetch ESGF certificate" in caplog.text

def test__fetch_esgf_certificate_unexpected_error(mocker, downloader, caplog, tmp_path):
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
    mock_md5 = mocker.patch('hashlib.md5')
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
    caplog.set_level(logging.DEBUG)
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
    assert "Stop detected" in caplog.text

def test_download_file_failure(mocker, downloader, caplog, tmp_path):
    output_path = tmp_path / "file.nc"
    mocker.patch.object(downloader.file_manager, 'get_output_path', return_value=output_path)
    mock_get = mocker.patch.object(downloader.session, 'get', side_effect=RequestException("Fail"))
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('pathlib.Path.unlink')
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

    mock_tpe = mocker.patch('gridflow.download.cmip6_downloader.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    mock_executor.submit.side_effect = mock_futures

    def mock_as_completed(fs):
        yield mock_futures[0]
        yield mock_futures[1]
        mock_stop_event.is_set.return_value = True
        yield mock_futures[2]

    mocker.patch('gridflow.download.cmip6_downloader.as_completed', side_effect=mock_as_completed)

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
    config = load_config("nonexistent.json")
    assert config == {}
    assert "Failed to load config" in caplog.text

def test_load_config_json_error(mocker, caplog):
    mocker.patch('builtins.open', mocker.mock_open(read_data="invalid json"))
    config = load_config("file.json")
    assert config == {}
    assert "Failed to load config" in caplog.text

# ############################################################################
# Tests for create_download_session
# ############################################################################

def test_create_download_session_retry_mode(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    retry_path = tmp_path / "failed.json"
    with open(retry_path, 'w') as f:
        json.dump([{'title': 'file'}], f)
    mocker.patch('gridflow.download.cmip6_downloader.load_config', return_value=[{'title': 'file'}])
    mock_fm = mocker.patch('gridflow.download.cmip6_downloader.FileManager')
    mock_dl = mocker.patch('gridflow.download.cmip6_downloader.Downloader')
    mock_dl.return_value.download_all.return_value = (["path"], [])
    settings = {'retry_failed_path': str(retry_path), 'output_dir': 'dir', 'metadata_dir': 'meta', 'save_mode': 'flat'}
    create_download_session({}, settings, mock_stop_event)
    assert "Retrying 1 files" in caplog.text

def test_create_download_session_retry_empty(mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('gridflow.download.cmip6_downloader.load_config', return_value=[])
    with pytest.raises(SystemExit):
        create_download_session({}, {'retry_failed_path': 'empty.json'}, stop_event)
    assert "Retry file is empty" in caplog.text

def test_create_download_session_no_params(caplog):
    with pytest.raises(SystemExit):
        create_download_session({}, {}, stop_event)
    assert "No specific search parameters provided" in caplog.text

def test_create_download_session_demo(mocker, mock_stop_event):
    settings = {'demo': True}
    mocker.patch.object(QueryHandler, 'fetch_datasets', return_value=[{'title': 'file'}])
    mocker.patch('gridflow.download.cmip6_downloader.FileManager')
    create_download_session({}, settings, mock_stop_event)

def test_create_download_session_dry_run(mocker, mock_stop_event, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch.object(QueryHandler, 'fetch_datasets', return_value=[{'title': 'file'}])
    mocker.patch('gridflow.download.cmip6_downloader.FileManager')
    settings = {'dry_run': True, 'timeout': 30, 'output_dir': './downloads', 'metadata_dir': './metadata', 'save_mode': 'structured'}
    
    create_download_session({'project': 'CMIP6', 'variable': 'tas'}, settings, mock_stop_event)
    
    assert "Dry run: Would attempt to download" in caplog.text

def test_create_download_session_success(mocker, mock_stop_event):
    mocker.patch.object(QueryHandler, 'fetch_datasets', return_value=[{'title': 'file1'}, {'title': 'file1'}, {'title': 'file2'}])
    mock_fm = mocker.patch('gridflow.download.cmip6_downloader.FileManager')
    mock_dl = mocker.patch('gridflow.download.cmip6_downloader.Downloader')
    mock_dl.return_value.download_all.return_value = (["path"], [{'title': 'failed'}])
    settings = {'max_downloads': 1, 'output_dir': 'dir', 'metadata_dir': 'meta', 'save_mode': 'flat', 'timeout': 30}

    create_download_session({'project': 'CMIP6', 'variable': 'tas'}, settings, mock_stop_event)
    
    assert mock_fm.return_value.save_metadata.call_count == 2 

def test_create_download_session_exception(mocker, mock_stop_event, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch.object(QueryHandler, 'fetch_datasets', side_effect=Exception("Critical"))
    
    create_download_session({'project': 'CMIP6', 'variable': 'tas'}, {}, mock_stop_event)
    
    assert "critical error" in caplog.text
    assert mock_stop_event.set.called

# ############################################################################
# Tests for add_arguments
# ############################################################################

def test_add_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--model', 'HadGEM3-GC31-LL', '--variable', 'tas', '--workers', '2'])
    assert args.model == 'HadGEM3-GC31-LL'
    assert args.variable == 'tas'
    assert args.workers == 2
    assert args.project == 'CMIP6'
    assert args.output_dir == './downloads_cmip6'

# ############################################################################
# Tests for main
# ############################################################################

@patch('argparse.ArgumentParser')
@patch('gridflow.download.cmip6_downloader.setup_logging')
@patch('gridflow.download.cmip6_downloader.create_download_session')
@patch('signal.signal')
def test_main_success(mock_signal, mock_session, mock_logging, mock_parser, caplog):
    caplog.set_level(logging.INFO)
    mock_args = MagicMock(
        config=None,
        demo=False,
        project='CMIP6',
        model='HadGEM3-GC31-LL',
        log_dir='./logs',
        log_level='info',
        is_gui_mode=False,
        stop_event=threading.Event(),
    )
    mock_parser.return_value.parse_args.return_value = mock_args

    from gridflow.download.cmip6_downloader import main, signal_handler
    main()

    mock_session.assert_called_once()
    mock_signal.assert_called_with(signal.SIGINT, signal_handler)
    assert "Execution was interrupted." not in caplog.text


@patch('argparse.ArgumentParser')
@patch('gridflow.download.cmip6_downloader.setup_logging')
@patch('gridflow.download.cmip6_downloader.create_download_session')
def test_main_demo(mock_session, mock_logging, mock_parser):
    mock_args = MagicMock(demo=True, config=None, log_dir='./logs', log_level='info')
    mock_parser.return_value.parse_args.return_value = mock_args
    main()
    mock_session.assert_called()

@patch('argparse.ArgumentParser')
@patch('gridflow.download.cmip6_downloader.setup_logging')
@patch('gridflow.download.cmip6_downloader.load_config', return_value={})
def test_main_config(mock_load, mock_logging, mock_parser, caplog):
    caplog.set_level(logging.ERROR)
    
    from argparse import Namespace
    mock_parser.return_value.parse_args.return_value = Namespace(
        config='config.json', demo=False, is_gui_mode=False,
        project=None, activity=None, experiment=None, variable=None, 
        frequency=None, model=None, ensemble=None, grid_label=None, resolution=None,
        latest=None, replica=None, data_node=None, openid=None,
        log_dir='./logs', log_level='info',
        stop_event=threading.Event()
    )
    
    with pytest.raises(SystemExit):
        main()
    
    assert "No specific search parameters provided" in caplog.text

@patch('argparse.ArgumentParser')
@patch('gridflow.download.cmip6_downloader.setup_logging')
def test_main_openid_validation(mock_logging, mock_parser, caplog):
    caplog.set_level(logging.INFO)
    
    from argparse import Namespace
    mock_parser.return_value.parse_args.return_value = Namespace(
        openid='openid', id=None, password='pass', 
        config=None, demo=False, is_gui_mode=False,
        log_dir='./logs', log_level='info',
        stop_event=threading.Event()
    )
    
    from gridflow.download.cmip6_downloader import main
    with pytest.raises(SystemExit):
        main()
        
    assert "Both --id and --password are required" in caplog.text

@patch('argparse.ArgumentParser')
@patch('gridflow.download.cmip6_downloader.setup_logging')
@patch('gridflow.download.cmip6_downloader.create_download_session')
def test_main_interrupted(mock_session, mock_logging, mock_parser, caplog):
    caplog.set_level(logging.INFO)

    evt = threading.Event()
    evt.set()  

    from argparse import Namespace
    mock_parser.return_value.parse_args.return_value = Namespace(
        config=None, demo=False, 
        project='CMIP6', model='HadGEM3-GC31-LL',
        activity=None, experiment=None, variable=None, frequency=None,
        ensemble=None, grid_label=None, resolution=None,
        latest=None, replica=None, data_node=None, openid=None,
        log_dir='./logs', log_level='info',
        output_dir='./downloads', metadata_dir='./metadata', save_mode='structured',
        is_gui_mode=False,
        stop_event=evt,
    )

    mock_session.side_effect = lambda params, settings, stop_event: None

    from gridflow.download.cmip6_downloader import main
    with pytest.raises(SystemExit):
        main()
        
    assert "Execution was interrupted." in caplog.text
