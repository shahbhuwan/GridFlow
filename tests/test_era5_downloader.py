import argparse
import logging
import signal
from concurrent.futures import Future
from importlib import reload
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from pathlib import Path

import gridflow.download.era5_downloader as era5_downloader
from gridflow.download.era5_downloader import Downloader, stop_event, signal_handler, generate_tasks, run_download_session, add_arguments, main

# ############################################################################
# Unit Tests for Pure Functions
# ############################################################################

def test_generate_tasks_single_month():
    """
    Tests task generation for a date range within a single month.
    """
    tasks = generate_tasks("2023-01-01", "2023-01-31", ["2m_temperature"])
    assert len(tasks) == 1, "Should generate one task for a single month"
    assert tasks[0] == (2023, 1, "2m_temperature"), "The generated task is incorrect"

def test_generate_tasks_multiple_months():
    """
    Tests task generation across multiple months with multiple variables.
    """
    start = "2023-11-01"
    end = "2024-02-15"  # Spans a year boundary
    variables = ["total_precipitation", "snowfall"]
    
    tasks = generate_tasks(start, end, variables)
    
    # Expected: (Nov, Dec, Jan, Feb) x 2 variables = 8 tasks
    assert len(tasks) == 8, "Should generate correct number of tasks across years"

    # Check for specific expected tasks
    expected_tasks = [
        (2023, 11, "total_precipitation"),
        (2023, 12, "total_precipitation"),
        (2024, 1, "total_precipitation"),
        (2024, 2, "total_precipitation"),
        (2023, 11, "snowfall"),
        (2023, 12, "snowfall"),
        (2024, 1, "snowfall"),
        (2024, 2, "snowfall"),
    ]
    for task in expected_tasks:
        assert task in tasks, f"Expected task {task} not found in generated list"

def test_generate_tasks_no_tasks():
    """
    Tests task generation when end date is before start date (no tasks).
    """
    tasks = generate_tasks("2023-02-01", "2023-01-31", ["2m_temperature"])
    assert len(tasks) == 0, "Should generate no tasks if end < start"

# ############################################################################
# Fixtures for Testing the Downloader Class
# ############################################################################

@pytest.fixture
def mock_cdsapi_client(mocker):
    """
    A pytest fixture that mocks the `cdsapi.Client`.
    This prevents any real network calls from being made during tests.
    """
    mock_client = MagicMock()
    # Default simulate success: touch file
    def simulate_download(dataset, request, output_path):
        Path(output_path).touch()  # Creates a dummy file
    
    mock_client.retrieve.side_effect = simulate_download
    mocker.patch('gridflow.download.era5_downloader.cdsapi.Client', return_value=mock_client)
    return mock_client

@pytest.fixture
def downloader_instance(tmp_path, mock_cdsapi_client):
    """
    A fixture to create a reusable instance of the Downloader class for tests.
    """
    settings = {
        'api_key': 'DUMMY_UID:DUMMY_KEY',
        'output_dir': str(tmp_path),
        'aoi': 'corn_belt',
        'workers': 1
    }
    stop_event = MagicMock()
    stop_event.is_set.return_value = False
    
    downloader = Downloader(settings, stop_event)
    return downloader

# ############################################################################
# Unit Tests for Import Dependency Checks
# ############################################################################

def test_missing_dependency_cdsapi(capsys):
    """
    Tests the except block for missing cdsapi library.
    """
    with patch.dict('sys.modules', {'cdsapi': None}):
        with pytest.raises(SystemExit) as exc:
            reload(era5_downloader)
        assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Missing required library: cdsapi" in captured.out

def test_missing_local_import_logging_utils(capsys):
    """
    Tests the except block for missing local import.
    """
    with patch.dict('sys.modules', {'gridflow.utils.logging_utils': None}):
        with pytest.raises(SystemExit) as exc:
            reload(era5_downloader)
        assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "gridflow/utils/logging_utils.py not found" in captured.out

# ############################################################################
# Unit Tests for Signal Handler
# ############################################################################

def test_signal_handler(caplog):
    """
    Tests the signal_handler function directly.
    """
    # FIX: Set level to INFO to capture all messages from the handler.
    caplog.set_level(logging.INFO)
    stop_event.clear()  # Ensure not set
    signal_handler(None, None)
    assert stop_event.is_set(), "Stop event should be set"
    assert "Stop signal received" in caplog.text, "Warning log not captured"
    assert "Please wait for ongoing tasks" in caplog.text, "Info log not captured"

# ############################################################################
# Unit Tests for the Downloader Class Methods
# ############################################################################

def test_downloader_init(mocker):
    """
    Tests Downloader initialization.
    """
    mock_Client = mocker.patch('gridflow.download.era5_downloader.cdsapi.Client')
    settings = {'api_key': 'uid:key', 'output_dir': '/tmp'}
    dl = Downloader(settings, stop_event=MagicMock())
    assert dl.settings == settings
    mock_Client.assert_called_with(url="https://cds.climate.copernicus.eu/api/v2", key='uid:key')

def test_download_month_success(downloader_instance):
    """
    Tests the `download_month` method for a successful download scenario.
    """
    area_bounds = [50.0, -125.0, 24.0, -66.0]
    
    result = downloader_instance.download_month(2023, 5, "2m_temperature", area_bounds)
    
    assert result is True, "The method should return True on success"
    
    mock_client = downloader_instance.cds_client
    mock_client.retrieve.assert_called_once()
    
    args, _ = mock_client.retrieve.call_args
    assert args[0] == "reanalysis-era5-land"
    
    output_path = args[2]
    assert "era5_land_2023_05_corn_belt_2m_temperature.nc" in str(output_path)
    assert Path(output_path).exists()

def test_download_month_already_exists(downloader_instance):
    """
    Tests that the download is skipped if the file already exists.
    """
    output_file = Path(downloader_instance.settings['output_dir']) / "era5_land_2023_06_corn_belt_snowfall.nc"
    output_file.touch()

    result = downloader_instance.download_month(2023, 6, "snowfall", [])
    
    assert result is True, "Should return True even if file is skipped"
    
    downloader_instance.cds_client.retrieve.assert_not_called()

def test_download_month_api_failure(downloader_instance, tmp_path):
    """
    Tests the behavior when the cdsapi client raises an error, including file cleanup.
    """
    def simulate_failure(dataset, request, output_path):
        Path(output_path).touch()  # Simulate partial file
        raise Exception("API failure")

    downloader_instance.cds_client.retrieve.side_effect = simulate_failure

    output_dir = Path(downloader_instance.settings['output_dir'])
    output_file = output_dir / "era5_land_2023_07_corn_belt_runoff.nc"
    
    result = downloader_instance.download_month(2023, 7, "runoff", [])
    
    assert result is False, "Should return False when the API call fails"
    assert not output_file.exists(), "Partial file should be deleted on failure"

def test_download_month_interrupted_during_download(downloader_instance, caplog):
    """
    Tests interruption during the download (stop_event set in except block).
    """
    caplog.set_level(logging.WARNING)
    
    def simulate_interrupt(dataset, request, output_path):
        downloader_instance._stop_event.is_set.return_value = True
        raise Exception("Interrupted")

    downloader_instance.cds_client.retrieve.side_effect = simulate_interrupt

    result = downloader_instance.download_month(2023, 8, "soil_type", [])
    
    assert result is False, "Should return False when interrupted"
    assert "interrupted by user" in caplog.text, "Interrupted log not captured"

def test_download_all_success(downloader_instance):
    """
    Tests download_all with multiple successful tasks.
    """
    tasks = [(2023, 1, "var1"), (2023, 2, "var2")]
    area = []
    
    successful, total = downloader_instance.download_all(tasks, area)
    
    assert total == 2
    assert successful == 2
    assert downloader_instance.cds_client.retrieve.call_count == 2

def test_download_all_partial_failure(downloader_instance):
    """
    Tests download_all with one success and one failure.
    """
    calls = 0
    def mixed_result(dataset, request, output_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            Path(output_path).touch()
            return
        raise Exception("fail")

    downloader_instance.cds_client.retrieve.side_effect = mixed_result

    tasks = [(2023, 1, "var1"), (2023, 2, "var2")]
    area = []
    
    successful, total = downloader_instance.download_all(tasks, area)
    
    assert total == 2
    assert successful == 1

def test_download_all_interrupted(mocker, downloader_instance):
    """
    Tests interruption during download_all loop.
    """
    tasks = [(2023, 1, "var1"), (2023, 2, "var2"), (2023, 3, "var3")]
    area = []

    mock_futures = [MagicMock(spec=Future) for _ in tasks]
    for mf in mock_futures:
        mf.result.return_value = True

    mock_tpe = mocker.patch('gridflow.download.era5_downloader.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    mock_executor.submit.side_effect = mock_futures

    def mock_as_completed(fs):
        def gen():
            yield mock_futures[0]
            yield mock_futures[1]
            downloader_instance._stop_event.is_set.return_value = True
            yield mock_futures[2]
        return gen()

    mocker.patch('gridflow.download.era5_downloader.as_completed', side_effect=mock_as_completed)

    successful, total = downloader_instance.download_all(tasks, area)

    assert total == 3
    assert successful == 2

def test_downloader_shutdown(downloader_instance):
    """
    Tests the shutdown method.
    """
    # Simulate executor created
    downloader_instance.executor = MagicMock()
    downloader_instance.shutdown()
    downloader_instance.executor.shutdown.assert_called_with(wait=True, cancel_futures=True)

# ############################################################################
# Unit Tests for run_download_session
# ############################################################################

@patch('gridflow.download.era5_downloader.Downloader')
def test_run_download_session_custom_bounds(mock_downloader_class, caplog):
    """
    Tests run_download_session with custom bounds.
    """
    caplog.set_level(logging.INFO)
    mock_instance = mock_downloader_class.return_value
    mock_instance.download_all.return_value = (1, 1)
    
    settings = {
        'bounds': [90, -180, -90, 180],
        'variables': '2m_temperature',
        'start_date': '2023-01-01',
        'end_date': '2023-01-31',
        'output_dir': '/tmp',
        'aoi': None  # Not used
    }
    mock_stop = MagicMock(is_set=False)
    
    run_download_session(settings, mock_stop)
    
    assert "Using custom bounding box" in caplog.text
    mock_downloader_class.assert_called()
    mock_instance.download_all.assert_called()

@patch('gridflow.download.era5_downloader.Downloader')
def test_run_download_session_predefined_aoi(mock_downloader_class, caplog):
    """
    Tests run_download_session with predefined AOI.
    """
    caplog.set_level(logging.INFO)
    mock_instance = mock_downloader_class.return_value
    mock_instance.download_all.return_value = (1, 1)
    
    settings = {
        'aoi': 'global',
        'variables': 'total_precipitation',
        'start_date': '2023-01-01',
        'end_date': '2023-01-01',
        'output_dir': '/tmp',
        'bounds': None
    }
    mock_stop = MagicMock()
    mock_stop.is_set.return_value = False # Correct mock setup
    
    run_download_session(settings, mock_stop)
    
    assert "Using predefined AOI 'global'" in caplog.text
    # FIX: The test should expect the one task that is generated.
    expected_tasks = [(2023, 1, 'total_precipitation')]
    mock_instance.download_all.assert_called_with(expected_tasks, era5_downloader.AOI_BOUNDS['global'])

@patch('gridflow.download.era5_downloader.Downloader')
def test_run_download_session_no_tasks(mock_downloader_class, caplog):
    """
    Tests run_download_session when no tasks are generated.
    """
    caplog.set_level(logging.INFO)
    settings = {
        'aoi': 'conus',
        'variables': 'snowfall',
        'start_date': '2023-02-01',
        'end_date': '2023-01-31',  # Invalid range
        'output_dir': '/tmp'
    }
    mock_stop = MagicMock()
    
    run_download_session(settings, mock_stop)
    
    assert "No tasks to process" in caplog.text
    mock_downloader_class.assert_not_called()

@patch('gridflow.download.era5_downloader.Downloader')
def test_run_download_session_exception(mock_downloader_class, caplog):
    """
    Tests exception handling in run_download_session.
    """
    caplog.set_level(logging.CRITICAL)
    mock_downloader_class.side_effect = Exception("Critical error")
    
    settings = {'aoi': 'conus', 'variables': 'var', 'start_date': '2023-01-01', 'end_date': '2023-01-31'}
    mock_stop = MagicMock(is_set=False)
    
    run_download_session(settings, mock_stop)
    
    assert "A critical error occurred" in caplog.text
    assert mock_stop.set.called

@patch('gridflow.download.era5_downloader.Downloader')
def test_run_download_session_stopped(mock_downloader_class, caplog):
    """
    Tests when stop_event is set after downloads.
    """
    caplog.set_level(logging.WARNING)
    mock_instance = mock_downloader_class.return_value
    mock_instance.download_all.return_value = (1, 2)
    
    settings = {'aoi': 'conus', 'variables': 'var', 'start_date': '2023-01-01', 'end_date': '2023-01-31'}
    # FIX: Correctly mock the is_set method to be callable.
    mock_stop = MagicMock()
    mock_stop.is_set.return_value = True
    
    run_download_session(settings, mock_stop)
    
    assert "Process was stopped" in caplog.text

# ############################################################################
# Unit Tests for add_arguments
# ############################################################################

def test_add_arguments():
    """
    Tests that arguments are added correctly to the parser.
    """
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    
    # Test parsing with valid args
    args = parser.parse_args(['--api_key', 'uid:key', '--output_dir', '/custom', '--aoi', 'global', '--workers', '2'])
    assert args.api_key == 'uid:key'
    assert args.output_dir == '/custom'
    assert args.aoi == 'global'
    assert args.workers == 2
    assert args.demo is False

    # Test defaults
    args = parser.parse_args(['--api_key', 'key'])
    assert args.start_date == "2020-01-01"
    assert args.end_date == "2020-01-31"
    assert args.variables == "2m_temperature,total_precipitation"
    assert args.aoi == "corn_belt"
    assert args.log_dir == "./gridflow_logs"
    assert args.log_level == "verbose"

    # Test mutually exclusive: can't use both aoi and bounds
    with pytest.raises(SystemExit):
        parser.parse_args(['--api_key', 'key', '--aoi', 'global', '--bounds', '90', '-180', '-90', '180'])

    # Test bounds requires 4 floats
    with pytest.raises(SystemExit):
        parser.parse_args(['--api_key', 'key', '--bounds', '90', '-180', '-90'])

# ############################################################################
# Unit Tests for main
# ############################################################################

@patch('argparse.ArgumentParser')
@patch('gridflow.download.era5_downloader.setup_logging')
@patch('gridflow.download.era5_downloader.run_download_session')
@patch('signal.signal')
def test_main_success(mock_signal, mock_run, mock_logging, mock_parser_class):
    """
    Tests main with valid args.
    """
    # FIX: Clear the global stop_event to prevent interference from other tests.
    era5_downloader.stop_event.clear()
    mock_parser = mock_parser_class.return_value
    # Add all required attributes to the mock to avoid TypeErrors
    mock_args = MagicMock(api_key='key', demo=False, log_dir='./logs', log_level='verbose')
    mock_parser.parse_args.return_value = mock_args

    main()

    mock_logging.assert_called()
    mock_run.assert_called_with(vars(mock_args), era5_downloader.stop_event)
    mock_signal.assert_called_with(signal.SIGINT, era5_downloader.signal_handler)

@patch('argparse.ArgumentParser')
@patch('gridflow.download.era5_downloader.setup_logging')
@patch('gridflow.download.era5_downloader.run_download_session')
def test_main_demo_mode(mock_run, mock_logging, mock_parser_class, caplog):
    """
    Tests main in demo mode.
    """
    # FIX: Clear the global stop_event.
    era5_downloader.stop_event.clear()
    caplog.set_level(logging.INFO)
    mock_parser = mock_parser_class.return_value
    mock_args = MagicMock(demo=True, api_key='key', log_dir='./logs', log_level='verbose')
    mock_parser.parse_args.return_value = mock_args
    
    main()
    
    assert "Running in demo mode" in caplog.text
    assert "Demo will download" in caplog.text
    call_settings = mock_run.call_args[0][0]
    assert call_settings['start_date'] == '2023-01-01'

@patch('argparse.ArgumentParser')
def test_main_no_api_key(mock_parser_class, capsys):
    """
    Tests main when no API key is provided.
    """
    mock_parser = mock_parser_class.return_value
    mock_args = MagicMock(api_key=None, demo=False, log_dir='./logs', log_level='verbose')
    mock_parser.parse_args.return_value = mock_args
    
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "An API key is required" in captured.out

@patch('argparse.ArgumentParser')
def test_main_demo_no_api_key(mock_parser_class, capsys):
    """
    Tests demo mode without API key.
    """
    mock_parser = mock_parser_class.return_value
    mock_args = MagicMock(demo=True, api_key=None, log_dir='./logs', log_level='verbose')
    mock_parser.parse_args.return_value = mock_args
    
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "API key is required for demo mode" in captured.out

@patch('argparse.ArgumentParser')
@patch('gridflow.download.era5_downloader.setup_logging')
@patch('gridflow.download.era5_downloader.run_download_session')
def test_main_interrupted(mock_run, mock_logging, mock_parser_class, caplog):
    """
    Tests main when stop_event is set (interrupted).
    """
    caplog.set_level(logging.WARNING)
    mock_parser = mock_parser_class.return_value
    mock_args = MagicMock(api_key='key', demo=False)
    mock_parser.parse_args.return_value = mock_args
    
    era5_downloader.stop_event.set()  # Simulate interrupt
    
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 130
    assert "Execution was interrupted" in caplog.text
    
    era5_downloader.stop_event.clear()  # Reset for other tests