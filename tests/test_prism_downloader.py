import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from gridflow.prism_downloader import FileManager, Downloader, compute_sha256, check_data_availability, validate_date, download_prism

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Mock PRISM file info
MOCK_FILE_INFO = {
    "url": "https://data.prism.oregonstate.edu/time_series/us/an/4km/tmean/monthly/2020/prism_tmean_us_25m_202001.zip",
    "filename": "prism_tmean_us_25m_202001.zip",
    "date": "202001",
    "output_path": None  # Will be set in tests
}

@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directories for downloads and metadata."""
    download_dir = tmp_path / "downloads"
    metadata_dir = tmp_path / "metadata"
    download_dir.mkdir()
    metadata_dir.mkdir()
    return download_dir, metadata_dir

@pytest.fixture
def file_manager(temp_dir):
    """Create a FileManager instance with temporary directories."""
    download_dir, metadata_dir = temp_dir
    return FileManager(str(download_dir), str(metadata_dir), metadata_prefix="test_")

@pytest.fixture
def downloader(file_manager):
    """Create a Downloader instance."""
    return Downloader(file_manager, retries=1, timeout=5, workers=2)

@pytest.fixture
def mock_requests_head():
    """Mock requests.head for availability checks."""
    with patch("requests.head") as mock_head:
        mock_head.return_value.status_code = 200
        yield mock_head

@pytest.fixture
def mock_requests_get():
    """Mock requests.get for downloads."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Length": "9"}  # Match the size of b"mock_data"
        mock_response.iter_content.return_value = [b"mock_data"]  # 9 bytes
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        yield mock_get

@pytest.fixture
def mock_thread_manager():
    """Mock ThreadManager to prevent shutdowns during tests."""
    with patch("gridflow.prism_downloader.ThreadManager") as mock_tm:
        mock_tm_instance = MagicMock()
        mock_tm_instance.is_shutdown.return_value = False
        mock_tm_instance.is_running.side_effect = [True, True, False]  # Simulate worker completion
        mock_tm_instance.add_worker.side_effect = lambda func, name: func(MagicMock(is_set=lambda: False))
        mock_tm_instance.stop.return_value = None
        mock_tm.return_value = mock_tm_instance
        yield mock_tm

# Test FileManager
def test_file_manager_create_directories(temp_dir, caplog):
    """Test directory creation in FileManager."""
    caplog.set_level(logging.ERROR)
    download_dir, metadata_dir = temp_dir
    fm = FileManager(str(download_dir / "new"), str(metadata_dir / "new"), metadata_prefix="test_")
    assert (download_dir / "new").exists()
    assert (metadata_dir / "new").exists()
    # Test directory creation failure
    with patch("pathlib.Path.mkdir", side_effect=Exception("Permission denied")):
        with pytest.raises(Exception, match="Permission denied"):
            FileManager(str(download_dir / "fail"), str(metadata_dir / "fail"), "")
    assert "Failed to create directories" in caplog.text

def test_file_manager_get_output_path(file_manager):
    """Test output path generation in FileManager."""
    output_path = file_manager.get_output_path("tmean", "4km", "202001")
    expected_path = file_manager.download_dir / "prism_tmean_us_4km_202001.zip"
    assert output_path == expected_path
    # Test with different resolution
    output_path = file_manager.get_output_path("ppt", "800m", "202002")
    expected_path = file_manager.download_dir / "prism_ppt_us_800m_202002.zip"
    assert output_path == expected_path

def test_file_manager_save_metadata(file_manager, caplog):
    """Test metadata saving in FileManager."""
    caplog.set_level(logging.DEBUG)
    files = [{"url": "http://example.com", "output_path": Path("test.zip"), "date": "202001"}]
    file_manager.save_metadata(files, "test_metadata.json")
    metadata_path = file_manager.metadata_dir / "test_test_metadata.json"
    assert metadata_path.exists()
    with open(metadata_path, "r") as f:
        saved_data = json.load(f)
        assert saved_data[0]["url"] == "http://example.com"
        assert saved_data[0]["output_path"] == "test.zip"
        assert saved_data[0]["date"] == "202001"
    assert "Saved metadata to" in caplog.text
    # Test save failure
    with patch("builtins.open", side_effect=Exception("Write error")):
        with pytest.raises(Exception, match="Write error"):
            file_manager.save_metadata(files, "fail.json")
    assert "Failed to save metadata" in caplog.text

# Test compute_sha256
def test_compute_sha256(temp_dir):
    """Test SHA256 checksum calculation."""
    download_dir, _ = temp_dir
    test_file = download_dir / "test.txt"
    test_file.write_bytes(b"mock_data")
    expected_checksum = "66132d93b22825cf80f48b8f149c60777b2fd069e4f07190f3f0b4052538545f"
    assert compute_sha256(test_file) == expected_checksum
    # Test with empty file
    empty_file = download_dir / "empty.txt"
    empty_file.write_bytes(b"")
    assert compute_sha256(empty_file) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        compute_sha256(download_dir / "nonexistent.txt")

# Test check_data_availability
def test_check_data_availability_success(mock_requests_head, caplog):
    """Test successful data availability check with HEAD request."""
    caplog.set_level(logging.INFO)
    result = check_data_availability("tmean", "4km", "monthly", 2020, "202001")
    assert result == {
        "url": "https://data.prism.oregonstate.edu/time_series/us/an/4km/tmean/monthly/2020/prism_tmean_us_25m_202001.zip",
        "filename": "prism_tmean_us_25m_202001.zip",
        "date": "202001"
    }
    assert "Data available for tmean at 4km (monthly) on 202001" in caplog.text

def test_check_data_availability_head_fallback_to_get(mock_requests_head, mock_requests_get, caplog):
    """Test fallback to GET request when HEAD fails."""
    caplog.set_level(logging.DEBUG)
    mock_requests_head.side_effect = requests.RequestException("HEAD failed")
    result = check_data_availability("tmean", "4km", "monthly", 2020, "202001")
    assert result == {
        "url": "https://data.prism.oregonstate.edu/time_series/us/an/4km/tmean/monthly/2020/prism_tmean_us_25m_202001.zip",
        "filename": "prism_tmean_us_25m_202001.zip",
        "date": "202001"
    }
    assert "HEAD request failed" in caplog.text
    assert "Data available for tmean at 4km (monthly) on 202001" in caplog.text

def test_check_data_availability_unavailable(mock_requests_head, mock_requests_get, caplog):
    """Test unavailable data with failed HTTP requests."""
    caplog.set_level(logging.WARNING)
    mock_requests_head.return_value.status_code = 404
    mock_requests_get.return_value.status_code = 404
    result = check_data_availability("tmean", "4km", "monthly", 2020, "202001")
    assert result is None
    assert "Data unavailable for tmean on 202001 (monthly, 4km): HTTP 404, skipping" in caplog.text
    # Test with request exception
    mock_requests_head.side_effect = requests.RequestException("Network error")
    mock_requests_get.side_effect = requests.RequestException("Network error")
    result = check_data_availability("tmean", "4km", "monthly", 2020, "202001")
    assert result is None
    assert "Data unavailable for tmean on 202001 (monthly, 4km): Network error, skipping" in caplog.text

def test_check_data_availability_800m_resolution(mock_requests_head, caplog):
    """Test data availability with 800m resolution."""
    caplog.set_level(logging.INFO)
    result = check_data_availability("ppt", "800m", "daily", 2020, "20200101")
    assert result == {
        "url": "https://data.prism.oregonstate.edu/time_series/us/an/800m/ppt/daily/2020/prism_ppt_us_30s_20200101.zip",
        "filename": "prism_ppt_us_30s_20200101.zip",
        "date": "20200101"
    }
    assert "Data available for ppt at 800m (daily) on 20200101" in caplog.text

# Test validate_date
def test_validate_date_valid_daily(caplog):
    """Test valid daily date formats in validate_date."""
    caplog.set_level(logging.ERROR)
    assert validate_date("2020-01-01", "daily") is True
    assert validate_date("20200101", "daily") is True
    assert "Invalid" not in caplog.text

def test_validate_date_valid_monthly(caplog):
    """Test valid monthly date formats in validate_date."""
    caplog.set_level(logging.ERROR)
    assert validate_date("2020-01", "monthly") is True
    assert validate_date("202001", "monthly") is True
    assert "Invalid" not in caplog.text

def test_validate_date_invalid_time_step(caplog):
    """Test invalid time_step in validate_date."""
    caplog.set_level(logging.ERROR)
    assert validate_date("2020-01-01", "yearly") is False
    assert "Invalid time_step: yearly. Must be 'daily' or 'monthly'" in caplog.text

def test_validate_date_invalid_format(caplog):
    """Test invalid date format in validate_date."""
    caplog.set_level(logging.ERROR)
    assert validate_date("2020-13-01", "daily") is False
    assert "Invalid date format: 2020-13-01. Expected YYYY-MM-DD or YYYYMMDD for daily data" in caplog.text
    assert validate_date("invalid", "daily") is False
    assert "Invalid date format: invalid. Expected YYYY-MM-DD or YYYYMMDD for daily data" in caplog.text
    assert validate_date("2020-13", "monthly") is False
    assert "Invalid date format: 2020-13. Expected YYYY-MM or YYYYMM for monthly data" in caplog.text

def test_validate_date_out_of_range(caplog):
    """Test out-of-range dates in validate_date."""
    caplog.set_level(logging.ERROR)
    assert validate_date("1980-01-01", "daily") is False
    assert "Date 1980-01-01 is too old for daily data (starts 1981)" in caplog.text
    assert validate_date("1894-01", "monthly") is False
    assert "Date 1894-01 is too old for monthly data (starts 1895)" in caplog.text
    assert validate_date("2026-01-01", "daily") is False
    assert "Date 2026-01-01 exceeds current year 2025 for daily data" in caplog.text

# Test Downloader
def test_downloader_init(file_manager):
    """Test Downloader initialization."""
    downloader = Downloader(file_manager, retries=2, timeout=10, workers=4)
    assert downloader.file_manager == file_manager
    assert downloader.retries == 2
    assert downloader.timeout == 10
    assert downloader.workers == 4
    assert downloader.successful_downloads == 0

def test_downloader_download_file_success(downloader, mock_requests_get, mock_thread_manager, caplog):
    """Test successful file download in Downloader."""
    caplog.set_level(logging.INFO)
    file_info = MOCK_FILE_INFO.copy()
    file_info["output_path"] = downloader.file_manager.download_dir / file_info["filename"]
    path = downloader.download_file(file_info)
    assert path == str(file_info["output_path"])
    assert file_info["output_path"].exists()
    assert downloader.successful_downloads == 1
    assert "Downloaded prism_tmean_us_25m_202001.zip" in caplog.text
    assert "SHA256 checksum for prism_tmean_us_25m_202001.zip" in caplog.text

def test_downloader_download_file_size_mismatch(downloader, mock_requests_get, mock_thread_manager, caplog):
    """Test file size mismatch in Downloader."""
    caplog.set_level(logging.ERROR)
    mock_response = mock_requests_get.return_value
    mock_response.headers = {"Content-Length": "2048"}  # Expected size larger than downloaded
    file_info = MOCK_FILE_INFO.copy()
    file_info["output_path"] = downloader.file_manager.download_dir / file_info["filename"]
    path = downloader.download_file(file_info)
    assert path is None
    assert file_info["output_path"].exists()  # File is created but download fails due to size mismatch
    assert "File size mismatch for prism_tmean_us_25m_202001.zip: expected 2048 bytes, got 9 bytes" in caplog.text

def test_downloader_download_file_failure(downloader, mock_requests_get, mock_thread_manager, caplog):
    """Test download failure after retries in Downloader."""
    caplog.set_level(logging.ERROR)
    mock_requests_get.side_effect = requests.RequestException("Download failed")
    file_info = MOCK_FILE_INFO.copy()
    file_info["output_path"] = downloader.file_manager.download_dir / file_info["filename"]
    path = downloader.download_file(file_info)
    assert path is None
    assert not file_info["output_path"].exists()
    assert "Failed to download" in caplog.text

def test_downloader_download_all(downloader, mock_requests_get, mock_thread_manager, caplog):
    """Test downloading multiple files in Downloader."""
    caplog.set_level(logging.INFO)
    files = [
        {**MOCK_FILE_INFO, "output_path": downloader.file_manager.download_dir / "prism_tmean_us_25m_202001.zip"},
        {**MOCK_FILE_INFO, "filename": "prism_tmean_us_25m_202002.zip", "date": "202002", 
         "output_path": downloader.file_manager.download_dir / "prism_tmean_us_25m_202002.zip",
         "url": "https://data.prism.oregonstate.edu/time_series/us/an/4km/tmean/monthly/2020/prism_tmean_us_25m_202002.zip"}
    ]
    downloaded = downloader.download_all(files)
    assert len(downloaded) == 2
    assert downloader.successful_downloads == 2
    assert (downloader.file_manager.download_dir / "prism_tmean_us_25m_202001.zip").exists()
    assert (downloader.file_manager.download_dir / "prism_tmean_us_25m_202002.zip").exists()
    assert "Final Progress: 2/2 files" in caplog.text

def test_downloader_download_all_empty(downloader, mock_thread_manager, caplog):
    """Test download_all with empty file list."""
    caplog.set_level(logging.INFO)
    downloaded = downloader.download_all([])
    assert len(downloaded) == 0
    assert downloader.successful_downloads == 0
    assert "No files to download" in caplog.text

def test_downloader_download_all_failure(downloader, mock_requests_get, mock_thread_manager, caplog):
    """Test download_all with some failed downloads."""
    caplog.set_level(logging.ERROR)
    mock_response_success = MagicMock(
        status_code=200,
        headers={"Content-Length": "9"},
        iter_content=MagicMock(return_value=[b"mock_data"]),  # Mock iter_content as a method
        raise_for_status=lambda: None
    )
    mock_requests_get.side_effect = [requests.RequestException("Download failed"), mock_response_success]
    files = [
        {**MOCK_FILE_INFO, "output_path": downloader.file_manager.download_dir / "prism_tmean_us_25m_202001.zip"},
        {**MOCK_FILE_INFO, "filename": "prism_tmean_us_25m_202002.zip", "date": "202002", 
         "output_path": downloader.file_manager.download_dir / "prism_tmean_us_25m_202002.zip",
         "url": "https://data.prism.oregonstate.edu/time_series/us/an/4km/tmean/monthly/2020/prism_tmean_us_25m_202002.zip"}
    ]
    downloaded = downloader.download_all(files)
    assert len(downloaded) == 1
    assert downloader.successful_downloads == 1
    assert not (downloader.file_manager.download_dir / "prism_tmean_us_25m_202001.zip").exists()
    assert (downloader.file_manager.download_dir / "prism_tmean_us_25m_202002.zip").exists()
    assert "Failed to download" in caplog.text

# Test download_prism
def test_download_prism_demo(mock_requests_head, mock_requests_get, mock_thread_manager, temp_dir, caplog):
    """Test download_prism in demo mode."""
    caplog.set_level(logging.DEBUG)
    download_dir, metadata_dir = temp_dir
    success = download_prism(
        variable="ppt",  # Will be overridden by demo mode
        resolution="4km",
        time_step="daily",
        start_date="2020-01-01",
        end_date="2020-01-01",
        output_dir=str(download_dir),
        metadata_dir=str(metadata_dir),
        log_level="debug",
        retries=1,
        timeout=5,
        demo=True,
        workers=4  # Match demo mode workers
    )
    assert success is True
    assert (download_dir / "prism_tmean_us_4km_202001.zip").exists()
    assert (download_dir / "prism_tmean_us_4km_202002.zip").exists()
    assert (download_dir / "prism_tmean_us_4km_202003.zip").exists()
    metadata_path = metadata_dir / "gridflow_prism_query_results.json"
    assert metadata_path.exists()
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        assert len(metadata) == 3
        dates = {item["date"] for item in metadata}
        assert {"202001", "202002", "202003"} == dates
    assert "Running in demo mode: downloading tmean for January–March 2020 (monthly, 4km)" in caplog.text
    assert "Checking availability for 3 monthly files" in caplog.text
    assert "Found 3 monthly files in chunk" in caplog.text  # Check all files processed
    assert "Completed: 3/3 files processed successfully" in caplog.text
    assert "Worker function completed" in caplog.text  # Ensure worker finishes

def test_download_prism_daily(mock_requests_head, mock_requests_get, mock_thread_manager, temp_dir, caplog):
    """Test download_prism for daily data."""
    caplog.set_level(logging.INFO)
    download_dir, metadata_dir = temp_dir
    success = download_prism(
        variable="tmean",
        resolution="4km",
        time_step="daily",
        start_date="2020-01-01",
        end_date="2020-01-02",
        output_dir=str(download_dir),
        metadata_dir=str(metadata_dir),
        log_level="debug",
        retries=1,
        timeout=5,
        demo=False,
        workers=2
    )
    assert success is True
    assert (download_dir / "prism_tmean_us_4km_20200101.zip").exists()
    assert (download_dir / "prism_tmean_us_4km_20200102.zip").exists()
    metadata_path = metadata_dir / "gridflow_prism_query_results.json"
    assert metadata_path.exists()
    assert "Completed: 2/2 files processed successfully" in caplog.text

def test_download_prism_monthly(mock_requests_head, mock_requests_get, mock_thread_manager, temp_dir, caplog):
    """Test download_prism for monthly data."""
    caplog.set_level(logging.INFO)
    download_dir, metadata_dir = temp_dir
    success = download_prism(
        variable="ppt",
        resolution="800m",
        time_step="monthly",
        start_date="2020-01",
        end_date="2020-02",
        output_dir=str(download_dir),
        metadata_dir=str(metadata_dir),
        log_level="debug",
        retries=1,
        timeout=5,
        demo=False,
        workers=2
    )
    assert success is True
    assert (download_dir / "prism_ppt_us_800m_202001.zip").exists()
    assert (download_dir / "prism_ppt_us_800m_202002.zip").exists()
    metadata_path = metadata_dir / "gridflow_prism_query_results.json"
    assert metadata_path.exists()
    assert "Completed: 2/2 files processed successfully" in caplog.text

def test_download_prism_invalid_variable(temp_dir, caplog):
    """Test download_prism with invalid variable."""
    caplog.set_level(logging.ERROR)
    download_dir, metadata_dir = temp_dir
    with pytest.raises(ValueError, match="Invalid variable: invalid"):
        download_prism(
            variable="invalid",
            resolution="4km",
            time_step="daily",
            start_date="2020-01-01",
            end_date="2020-01-01",
            output_dir=str(download_dir),
            metadata_dir=str(metadata_dir),
            log_level="debug",
            retries=1,
            timeout=5,
            demo=False
        )
    assert "Invalid variable 'invalid'" in caplog.text

def test_download_prism_invalid_date_range(temp_dir, caplog):
    """Test download_prism with invalid date range."""
    caplog.set_level(logging.ERROR)
    download_dir, metadata_dir = temp_dir
    with pytest.raises(ValueError, match="Start date must be before or equal to end date"):
        download_prism(
            variable="tmean",
            resolution="4km",
            time_step="daily",
            start_date="2020-01-02",
            end_date="2020-01-01",
            output_dir=str(download_dir),
            metadata_dir=str(metadata_dir),
            log_level="debug",
            retries=1,
            timeout=5,
            demo=False
        )
    assert "start_date must be before or equal to end_date" in caplog.text

def test_download_prism_invalid_date_format(temp_dir, caplog):
    """Test download_prism with invalid date format."""
    caplog.set_level(logging.ERROR)
    download_dir, metadata_dir = temp_dir
    with pytest.raises(ValueError, match="Invalid start date: invalid"):
        download_prism(
            variable="tmean",
            resolution="4km",
            time_step="daily",
            start_date="invalid",
            end_date="2020-01-01",
            output_dir=str(download_dir),
            metadata_dir=str(metadata_dir),
            log_level="debug",
            retries=1,
            timeout=5,
            demo=False
        )
    assert "Invalid date format: invalid" in caplog.text

def test_download_prism_no_files_available(mock_requests_head, mock_requests_get, mock_thread_manager, temp_dir, caplog):
    """Test download_prism when no files are available."""
    caplog.set_level(logging.WARNING)
    mock_requests_head.return_value.status_code = 404
    mock_requests_get.return_value.status_code = 404
    download_dir, metadata_dir = temp_dir
    with pytest.raises(ValueError, match="No PRISM files available for download"):
        download_prism(
            variable="tmean",
            resolution="4km",
            time_step="daily",
            start_date="2020-01-01",
            end_date="2020-01-01",
            output_dir=str(download_dir),
            metadata_dir=str(metadata_dir),
            log_level="debug",
            retries=1,
            timeout=5,
            demo=False
        )
    assert "Data unavailable for tmean on 20200101 (daily, 4km): HTTP 404" in caplog.text
    assert "No PRISM files to download" in caplog.text

def test_download_prism_all_existing(downloader, mock_requests_head, mock_requests_get, mock_thread_manager, temp_dir, caplog):
    """Test download_prism when all files already exist."""
    caplog.set_level(logging.INFO)
    download_dir, metadata_dir = temp_dir
    # Pre-create files
    (download_dir / "prism_tmean_us_4km_20200101.zip").write_bytes(b"mock_data")
    (download_dir / "prism_tmean_us_4km_20200102.zip").write_bytes(b"mock_data")
    success = download_prism(
        variable="tmean",
        resolution="4km",
        time_step="daily",
        start_date="2020-01-01",
        end_date="2020-01-02",
        output_dir=str(download_dir),
        metadata_dir=str(metadata_dir),
        log_level="debug",
        retries=1,
        timeout=5,
        demo=False,
        workers=2
    )
    assert success is True
    assert "File prism_tmean_us_4km_20200101.zip already exists" in caplog.text
    assert "File prism_tmean_us_4km_20200102.zip already exists" in caplog.text
    assert "Completed: 2/2 files processed successfully" in caplog.text
    metadata_path = metadata_dir / "gridflow_prism_query_results.json"
    assert metadata_path.exists()

def test_download_prism_default_workers(mock_requests_head, mock_requests_get, mock_thread_manager, temp_dir, caplog):
    """Test download_prism with default workers (None)."""
    caplog.set_level(logging.INFO)
    download_dir, metadata_dir = temp_dir
    with patch("os.cpu_count", return_value=4):
        success = download_prism(
            variable="tmean",
            resolution="4km",
            time_step="daily",
            start_date="2020-01-01",
            end_date="2020-01-01",
            output_dir=str(download_dir),
            metadata_dir=str(metadata_dir),
            log_level="info",
            retries=1,
            timeout=5,
            demo=False,
            workers=None
        )
    assert success is True
    assert (download_dir / "prism_tmean_us_4km_20200101.zip").exists()
    assert "Completed: 1/1 files processed successfully" in caplog.text