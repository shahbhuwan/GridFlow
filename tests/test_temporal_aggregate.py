# tests/test_temporal_aggregate.py

import argparse
import logging
import signal
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
from datetime import datetime

import gridflow.processing.temporal_aggregate as temporal_aggregate
from gridflow.processing.temporal_aggregate import (
    signal_handler,
    find_time_variable,
    aggregate_single_file,
    TemporalAggregator,
    run_aggregator_session,
    add_arguments,
    main,
)

# ############################################################################
# Fixtures
# ############################################################################

# FIX: Define a helper class to properly mock a NetCDF variable.
# This class is subscriptable and has a clean __dict__ for iteration.
class MockNetCDFVariable:
    def __init__(self):
        self.dtype = np.float32
        self.dimensions = ('time', 'lat', 'lon')
        self._FillValue = -999.0
        self.units = 'K'
        self.long_name = 'temperature'

    def __getitem__(self, key):
        # This makes the object subscriptable, e.g., var[:]
        # FIX: Data must be 3D to match dimensions ('time', 'lat', 'lon')
        return np.ma.array([[[1]], [[2]], [[3]]])

@pytest.fixture
def mock_stop_event():
    event = MagicMock(spec=threading.Event)
    event.is_set.return_value = False
    return event

@pytest.fixture
def aggregator(mock_stop_event):
    settings = {'workers': 1, 'variable': 'tas', 'output_frequency': 'monthly', 'method': 'mean'}
    return TemporalAggregator(settings, mock_stop_event)

@pytest.fixture
def mock_nc_dataset(mocker):
    mock_src = MagicMock()
    mock_dst = MagicMock()
    mocker.patch('gridflow.processing.temporal_aggregate.nc.Dataset', side_effect=[mock_src, mock_dst])
    mock_src.__enter__.return_value = mock_src
    mock_dst.__enter__.return_value = mock_dst

    # --- Mocks for variables and dimensions ---
    mock_time = MagicMock()
    mock_time.units = 'days since 2000-01-01'
    mock_time.calendar = 'standard'
    mock_time.axis = 'T'
    mock_time.__getitem__.return_value = np.array([0, 31, 59])  # Jan 1, Feb 1, Mar 1
    
    # Use the custom helper class for the variable mock
    mock_var = MockNetCDFVariable()
    
    dim_time = MagicMock()
    dim_time.isunlimited.return_value = True
    dim_lat = MagicMock()
    dim_lat.isunlimited.return_value = False
    dim_lat.__len__.return_value = 1
    dim_lon = MagicMock()
    dim_lon.isunlimited.return_value = False
    dim_lon.__len__.return_value = 1
    
    # Set attributes directly and avoid touching __dict__ to prevent mock corruption.
    mock_src.file_format = 'NETCDF4'
    mock_src.dimensions = {'time': dim_time, 'lat': dim_lat, 'lon': dim_lon}
    mock_src.variables = {'time': mock_time, 'tas': mock_var}

    mock_create_var = MagicMock()
    mock_dst.createVariable.return_value = mock_create_var
    
    return mock_src, mock_dst, mock_create_var

# ############################################################################
# Tests for signal_handler
# ############################################################################

def test_signal_handler(caplog, mocker):
    caplog.set_level(logging.INFO)
    mock_event = mocker.patch('gridflow.processing.temporal_aggregate.stop_event')
    signal_handler(None, None)
    assert mock_event.set.called
    assert "Stop signal received" in caplog.text
    assert "Please wait for ongoing tasks" in caplog.text

# ############################################################################
# Tests for find_time_variable
# ############################################################################

def test_find_time_variable_standard_name():
    mock_ds = MagicMock()
    mock_var = MagicMock(standard_name='time', spec=['standard_name', 'axis'])
    mock_ds.variables = {'tvar': mock_var}
    assert find_time_variable(mock_ds) == 'tvar'

def test_find_time_variable_axis():
    mock_ds = MagicMock()
    mock_var = MagicMock(axis='T', spec=['standard_name', 'axis'])
    mock_ds.variables = {'tvar': mock_var}
    assert find_time_variable(mock_ds) == 'tvar'

def test_find_time_variable_fallback():
    mock_ds = MagicMock()
    mock_ds.variables = {'time': MagicMock(spec=[])}
    assert find_time_variable(mock_ds) == 'time'

def test_find_time_variable_none():
    mock_ds = MagicMock()
    mock_ds.variables = {}
    assert find_time_variable(mock_ds) is None

# ############################################################################
# Tests for aggregate_single_file
# ############################################################################

def test_aggregate_single_file_success(mocker, mock_nc_dataset, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"
    
    mocker.patch('gridflow.processing.temporal_aggregate.nc.num2date', return_value=[
        datetime(2000, 1, 1), datetime(2000, 2, 1), datetime(2000, 3, 1)
    ])
    mocker.patch('gridflow.processing.temporal_aggregate.nc.date2num', side_effect=[15, 45, 75])
    
    result = aggregate_single_file(input_path, output_path, 'tas', 'monthly', 'mean', mock_stop_event)
    
    assert result is True
    assert "Successfully aggregated" in caplog.text

def test_aggregate_single_file_no_time_var(mock_nc_dataset, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"
    mock_src, _, _ = mock_nc_dataset
    mock_src.variables = {'tas': mock_src.variables['tas']} # Remove time
    
    assert not aggregate_single_file(input_path, output_path, 'tas', 'monthly', 'mean', mock_stop_event)
    assert "Required time or variable" in caplog.text

def test_aggregate_single_file_no_variable(mock_nc_dataset, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"
    mock_src, _, _ = mock_nc_dataset
    mock_src.variables = {'time': mock_src.variables['time']} # Remove tas
    
    assert not aggregate_single_file(input_path, output_path, 'tas', 'monthly', 'mean', mock_stop_event)
    assert "Required time or variable" in caplog.text

def test_aggregate_single_file_unsupported_frequency(mock_nc_dataset, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"
    
    assert not aggregate_single_file(input_path, output_path, 'tas', 'daily', 'mean', mock_stop_event)
    assert "Unsupported frequency" in caplog.text

def test_aggregate_single_file_unsupported_method(mock_nc_dataset, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"
    
    assert not aggregate_single_file(input_path, output_path, 'tas', 'monthly', 'mode', mock_stop_event)
    assert "Unsupported aggregation method" in caplog.text

def test_aggregate_single_file_interrupted(mock_nc_dataset, mock_stop_event, tmp_path):
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"
    mock_stop_event.is_set.return_value = True
    
    assert not aggregate_single_file(input_path, output_path, 'tas', 'monthly', 'mean', mock_stop_event)

def test_aggregate_single_file_exception(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    mocker.patch('gridflow.processing.temporal_aggregate.nc.Dataset', side_effect=Exception("Fail"))
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"
    
    assert not aggregate_single_file(input_path, output_path, 'tas', 'monthly', 'mean', mock_stop_event)
    assert "Failed to aggregate" in caplog.text

# ############################################################################
# Tests for TemporalAggregator
# ############################################################################

def test_aggregator_init(mock_stop_event):
    settings = {'workers': 2, 'variable': 'tas', 'output_frequency': 'monthly', 'method': 'mean'}
    agg = TemporalAggregator(settings, mock_stop_event)
    assert agg.settings == settings
    assert agg._stop_event == mock_stop_event
    assert agg.executor is None

def test_shutdown(aggregator):
    aggregator.executor = MagicMock()
    aggregator.shutdown()
    aggregator.executor.shutdown.assert_called_with(wait=True, cancel_futures=True)

def test_aggregate_all_no_files(aggregator):
    successful, total = aggregator.aggregate_all([])
    assert successful == 0
    assert total == 0

def test_aggregate_all_success(mocker, aggregator):
    mock_tpe = mocker.patch('gridflow.processing.temporal_aggregate.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    mock_futures = [MagicMock(spec=Future, result=lambda: True)]
    mock_executor.submit.return_value = mock_futures[0]
    mocker.patch('gridflow.processing.temporal_aggregate.as_completed', return_value=mock_futures)
    
    successful, total = aggregator.aggregate_all([(Path('in'), Path('out'))])
    assert successful == 1
    assert total == 1

def test_aggregate_all_failure(mocker, aggregator):
    mock_tpe = mocker.patch('gridflow.processing.temporal_aggregate.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    mock_futures = [MagicMock(spec=Future, result=lambda: False)]
    mock_executor.submit.return_value = mock_futures[0]
    mocker.patch('gridflow.processing.temporal_aggregate.as_completed', return_value=mock_futures)
    
    successful, total = aggregator.aggregate_all([(Path('in'), Path('out'))])
    assert successful == 0
    assert total == 1

def test_aggregate_all_interrupted(mocker, aggregator, mock_stop_event):
    mock_tpe = mocker.patch('gridflow.processing.temporal_aggregate.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    mock_futures = [MagicMock(spec=Future) for _ in range(2)]
    mock_futures[0].result.return_value = True
    mock_futures[1].result.return_value = True
    mock_executor.submit.side_effect = mock_futures

    def mock_as_completed(fs):
        yield mock_futures[0]
        mock_stop_event.is_set.return_value = True
        yield mock_futures[1]

    mocker.patch('gridflow.processing.temporal_aggregate.as_completed', side_effect=mock_as_completed)
    
    successful, total = aggregator.aggregate_all([(Path('in1'), Path('out1')), (Path('in2'), Path('out2'))])
    assert successful == 1
    assert total == 2

# ############################################################################
# Tests for run_aggregator_session
# ############################################################################

def test_run_aggregator_session_success(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    input_dir_path = tmp_path / 'input'
    input_dir_path.mkdir()
    (input_dir_path / 'file.nc').touch()
    
    settings = {
        'input_dir': str(input_dir_path),
        'output_dir': str(tmp_path / 'output'),
        'variable': 'tas',
        'output_frequency': 'monthly',
        'method': 'mean',
        'workers': 1
    }
    mocker.patch.object(TemporalAggregator, 'aggregate_all', return_value=(1, 1))
    
    run_aggregator_session(settings, mock_stop_event)
    assert "Found 1 NetCDF files" in caplog.text
    assert "Completed: 1/1" in caplog.text

def test_run_aggregator_session_no_files(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    input_dir_path = tmp_path / 'input'
    input_dir_path.mkdir()
    
    settings = {
        'input_dir': str(input_dir_path),
        'output_dir': str(tmp_path / 'output'),
        'variable': 'tas',
        'output_frequency': 'monthly',
        'method': 'mean',
        'workers': 1
    }
    with pytest.raises(SystemExit):
        run_aggregator_session(settings, mock_stop_event)
    assert "No NetCDF (.nc) files found" in caplog.text

def test_run_aggregator_session_invalid_input_dir(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    
    settings = {
        'input_dir': str(tmp_path / 'nonexist'),
        'output_dir': str(tmp_path / 'output'),
        'variable': 'tas',
        'output_frequency': 'monthly',
        'method': 'mean'
    }
    with pytest.raises(SystemExit):
        run_aggregator_session(settings, mock_stop_event)
    assert "Input directory not found" in caplog.text

def test_run_aggregator_session_exception(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.CRITICAL)
    
    settings = {
        'input_dir': str(tmp_path),
        'output_dir': str(tmp_path / 'output'),
        'variable': 'tas',
        'output_frequency': 'monthly',
        'method': 'mean'
    }
    mocker.patch('pathlib.Path.rglob', side_effect=Exception("Critical"))
    
    run_aggregator_session(settings, mock_stop_event)
    assert "critical error" in caplog.text.lower()
    assert mock_stop_event.set.called

# ############################################################################
# Tests for add_arguments
# ############################################################################

def test_add_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--input_dir', './in', '--output_dir', './out', '--variable', 'tas'])
    assert args.input_dir == './in'
    assert args.output_dir == './out'
    assert args.variable == 'tas'
    assert args.output_frequency == 'monthly'
    assert args.method == 'mean'

# ############################################################################
# Tests for main
# ############################################################################

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.temporal_aggregate.setup_logging')
@patch('gridflow.processing.temporal_aggregate.run_aggregator_session')
@patch('signal.signal')
def test_main_success(mock_signal, mock_session, mock_logging, mock_parser):
    mock_args = MagicMock(input_dir='./in', output_dir='./out', variable='tas', log_dir='./logs', log_level='info', demo=False)
    mock_parser.return_value.parse_args.return_value = mock_args
    main()
    mock_session.assert_called()
    mock_signal.assert_called_with(signal.SIGINT, signal_handler)

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.temporal_aggregate.setup_logging')
@patch('gridflow.processing.temporal_aggregate.run_aggregator_session')
def test_main_demo(mock_session, mock_logging, mock_parser, caplog):
    caplog.set_level(logging.INFO)
    mock_args = MagicMock(demo=True, log_dir='./logs', log_level='info')
    mock_parser.return_value.parse_args.return_value = mock_args
    main()
    assert "Running in demo mode" in caplog.text
    mock_session.assert_called()

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.temporal_aggregate.setup_logging')
def test_main_no_params(mock_logging, mock_parser, caplog):
    caplog.set_level(logging.ERROR)
    mock_args = MagicMock(demo=False, input_dir=None, output_dir=None, variable=None, log_dir='.', log_level='info')
    mock_parser.return_value.parse_args.return_value = mock_args
    with pytest.raises(SystemExit):
        main()
    assert "the following arguments are required" in caplog.text

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.temporal_aggregate.setup_logging')
@patch('gridflow.processing.temporal_aggregate.stop_event')
@patch('gridflow.processing.temporal_aggregate.run_aggregator_session')
def test_main_interrupted(mock_session, mock_stop, mock_logging, mock_parser, caplog):
    caplog.set_level(logging.WARNING)
    mock_args = MagicMock(demo=True, log_dir='./logs', log_level='info')
    mock_parser.return_value.parse_args.return_value = mock_args
    mock_stop.is_set.return_value = True
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 130
    assert "Execution was interrupted" in caplog.text
