# tests/test_temporal_aggregate.py

import argparse
import logging
import signal
import sys
import threading
import os
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
from datetime import datetime

# Updated imports to match the new gridflow.processing.temporal_aggregate structure
import gridflow.processing.temporal_aggregate as temporal_aggregate
from gridflow.processing.temporal_aggregate import (
    signal_handler,
    find_time_variable,
    aggregate_single_file,
    TemporalAggregator,
    run_aggregator_session,
    add_arguments,
    main,
    HAS_UI_LIBS
)

# ############################################################################
# Fixtures
# ############################################################################

class MockNetCDFVariable:
    """Helper to simulate a subscriptable NetCDF variable with metadata."""
    def __init__(self):
        self.dtype = np.float32
        self.dimensions = ('time', 'lat', 'lon')
        self._FillValue = -999.0
        self.units = 'K'
        self.__dict__.update({'long_name': 'temperature'})

    def __getitem__(self, key):
        # Returns a 3D masked array to match dimensions
        return np.ma.array([[[273.15]], [[274.15]], [[275.15]]])

@pytest.fixture
def mock_stop_event():
    """Fixture for a mock threading.Event."""
    event = MagicMock(spec=threading.Event)
    event.is_set.return_value = False
    return event

@pytest.fixture
def aggregator(mock_stop_event):
    """Fixture for a TemporalAggregator instance."""
    settings = {
        'workers': 1, 
        'variable': 'tas', 
        'output_frequency': 'monthly', 
        'method': 'mean',
        'is_gui_mode': False
    }
    return TemporalAggregator(settings, mock_stop_event)

@pytest.fixture
def mock_nc_dataset(mocker):
    """Orchestrates clean mocks for NetCDF Source and Destination datasets."""
    mock_src = MagicMock(name="SourceDataset")
    mock_dst = MagicMock(name="DestinationDataset")

    mocker.patch('gridflow.processing.temporal_aggregate.nc.Dataset', side_effect=[mock_src, mock_src, mock_dst])
    
    # Setup context manager behavior
    mock_src.__enter__.return_value = mock_src
    mock_dst.__enter__.return_value = mock_dst

    # Setup Source Time Variable
    mock_time = MagicMock(name="TimeVar")
    mock_time.units = 'days since 2000-01-01'
    mock_time.calendar = 'standard'
    mock_time.axis = 'T'
    # Return a numpy array when sliced/indexed
    mock_time.__getitem__.return_value = np.array([0, 31, 59])

    # Setup Source Data Variable
    mock_var = MockNetCDFVariable()

    # Setup Dimensions
    dim_time = MagicMock(name="TimeDim")
    dim_time.isunlimited.return_value = True
    dim_lat = MagicMock(name="LatDim")
    dim_lat.__len__.return_value = 1
    
    # Configure mock_src attributes without overwriting __dict__
    mock_src.file_format = 'NETCDF4'
    mock_src.dimensions = {'time': dim_time, 'lat': dim_lat}
    mock_src.variables = {'time': mock_time, 'tas': mock_var}
    mock_src.Conventions = 'CF-1.7'

    # Setup Destination variable creation
    mock_create_var = MagicMock(name="CreatedVar")
    mock_dst.createVariable.return_value = mock_create_var
    
    return mock_src, mock_dst, mock_create_var

# ############################################################################
# Tests for Core Logic & Helpers
# ############################################################################

def test_signal_handler(caplog, mocker):
    """Verify stop_event is set on signal."""
    caplog.set_level(logging.WARNING)
    mock_event = mocker.patch('gridflow.processing.temporal_aggregate.stop_event')
    signal_handler(None, None)
    assert mock_event.set.called
    assert "Stop signal received" in caplog.text

def test_find_time_variable():
    """Verify identification of time dimension via attributes and names."""
    mock_ds = MagicMock()
    # Test by axis
    v1 = MagicMock(axis='T')
    mock_ds.variables = {'t1': v1}
    assert find_time_variable(mock_ds) == 't1'
    
    # Test by standard_name
    v2 = MagicMock(standard_name='time', spec=['standard_name'])
    mock_ds.variables = {'t2': v2}
    assert find_time_variable(mock_ds) == 't2'

# ############################################################################
# Tests for aggregate_single_file
# ############################################################################

def test_aggregate_single_file_success(mocker, mock_nc_dataset, mock_stop_event, tmp_path):
    """Test successful monthly mean aggregation."""
    input_path = tmp_path / "input.nc"
    input_path.touch() 
    output_path = tmp_path / "output.nc"

    mock_src, mock_dst, _ = mock_nc_dataset
    
    # Ensure the axis is set correctly for discovery
    mock_src.variables['time'].axis = 'T'
    
    mocker.patch('numpy.meshgrid', return_value=(np.array([[0]]), np.array([[0]])))
    
    # Mock date conversion utilities
    mocker.patch('gridflow.processing.temporal_aggregate.nc.num2date', return_value=[
        datetime(2000, 1, 1), datetime(2000, 1, 15), datetime(2000, 2, 1)
    ])
    mocker.patch('gridflow.processing.temporal_aggregate.nc.date2num', side_effect=[15.0, 45.0])

    success, msg = aggregate_single_file(input_path, output_path, 'tas', 'monthly', 'mean', mock_stop_event)

    assert success is True
    assert msg == "output.nc"

def test_aggregate_single_file_interrupted(mock_stop_event):
    """Verify graceful exit when stop_event is set."""
    mock_stop_event.is_set.return_value = True
    success, msg = aggregate_single_file(Path("in.nc"), Path("out.nc"), 'tas', 'monthly', 'mean', mock_stop_event)
    assert not success
    assert msg == "Interrupted"

def test_aggregate_single_file_failure_missing_var(mocker, mock_nc_dataset, mock_stop_event, tmp_path):
    """Test failure when target variable is missing from NetCDF."""
    input_path = tmp_path / "input.nc"
    input_path.touch()

    mock_src, _, _ = mock_nc_dataset

    # Remove the target variable from the mock dictionary
    mock_src.variables = {'time': mock_src.variables['time']} 

    success, msg = aggregate_single_file(input_path, Path("out.nc"), 'tas', 'monthly', 'mean', mock_stop_event)
    
    assert success is False
    assert "not found" in msg

# ############################################################################
# Tests for TemporalAggregator Class
# ############################################################################

def test_aggregator_aggregate_all_success(mocker, aggregator):
    """Verify parallel execution orchestration and successful count."""
    mock_tpe = mocker.patch('gridflow.processing.temporal_aggregate.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    
    # Simulate a successful task result (Tuple[bool, str])
    mock_future = MagicMock(spec=Future)
    mock_future.result.return_value = (True, "file_monthly.nc")
    mock_executor.submit.return_value = mock_future
    mocker.patch('gridflow.processing.temporal_aggregate.as_completed', return_value=[mock_future])
    
    tasks = [(Path('in.nc'), Path('out.nc'))]
    success_count, total = aggregator.aggregate_all(tasks)
    
    assert success_count == 1
    assert total == 1

def test_aggregator_aggregate_all_interrupted(mocker, aggregator, mock_stop_event):
    """Verify that remaining tasks are cancelled if stop_event is set."""
    mock_tpe = mocker.patch('gridflow.processing.temporal_aggregate.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    
    f1 = MagicMock(spec=Future)
    f1.result.return_value = (True, "res1.nc")
    f2 = MagicMock(spec=Future)
    
    mock_executor.submit.side_effect = [f1, f2]

    def mock_as_completed(fs):
        yield f1
        mock_stop_event.is_set.return_value = True
        yield f2 # Will be cancelled/skipped in loop

    mocker.patch('gridflow.processing.temporal_aggregate.as_completed', side_effect=mock_as_completed)
    
    tasks = [(Path('in1.nc'), Path('out1.nc')), (Path('in2.nc'), Path('out2.nc'))]
    success_count, total = aggregator.aggregate_all(tasks)
    
    assert success_count == 1
    assert f2.cancel.called

# ############################################################################
# Tests for Session & CLI
# ############################################################################

@patch('gridflow.processing.temporal_aggregate.TemporalAggregator')
def test_run_aggregator_session_discovery(MockAgg, mock_stop_event, tmp_path, caplog, capsys):
    """Verify file discovery and task creation in session orchestrator."""
    caplog.set_level(logging.INFO)
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "test.nc").touch()

    settings = {
        'input_dir': str(in_dir),
        'output_dir': str(tmp_path / "out"),
        'variable': 'tas',
        'output_frequency': 'monthly',
        'method': 'mean',
        'is_gui_mode': False
    }

    mock_agg_instance = MockAgg.return_value
    mock_agg_instance.aggregate_all.return_value = (1, 1)

    run_aggregator_session(settings, mock_stop_event)

    captured = capsys.readouterr()
    assert "Found 1 NetCDF files" in captured.out
    assert "aggregated successfully" in caplog.text

@patch('argparse.ArgumentParser.parse_args')
@patch('gridflow.processing.temporal_aggregate.setup_logging')
@patch('gridflow.processing.temporal_aggregate.run_aggregator_session')
@patch('signal.signal')
def test_main_success(mock_signal, mock_session, mock_setup, mock_args_call):
    """Verify CLI main entry point logic."""
    mock_args = argparse.Namespace(
        input_dir='in', output_dir='out', variable='tas',
        output_frequency='monthly', method='mean',
        log_dir='./logs', log_level='verbose', demo=False,
        is_gui_mode=False, workers=4
    )
    mock_args_call.return_value = mock_args
    
    main()
    mock_setup.assert_called_once()
    mock_session.assert_called_once()
    mock_signal.assert_called_once()

@patch('argparse.ArgumentParser.parse_args')
@patch('gridflow.processing.temporal_aggregate.run_aggregator_session')
@patch('gridflow.processing.temporal_aggregate.setup_logging')
def test_main_demo_mode(mock_setup, mock_session, mock_args_call, capsys, caplog):
    """Verify demo mode defaults and command printing."""
    caplog.set_level(logging.INFO)
    mock_args = argparse.Namespace(
        demo=True, log_dir='./logs', log_level='verbose',
        is_gui_mode=False, input_dir=None, output_dir=None,
        variable=None, output_frequency='monthly', method='mean'
    )
    mock_args_call.return_value = mock_args
    
    main()
    
    # If HAS_UI_LIBS is true, it prints to stdout, otherwise logs it
    captured = capsys.readouterr()
    if HAS_UI_LIBS:
        assert "Running in demo mode" in captured.out
        assert "gridflow aggregate" in captured.out
    else:
        assert "Running in demo mode" in caplog.text
    
    mock_session.assert_called_once()
    called_settings = mock_session.call_args[0][0]
    assert called_settings['input_dir'] == './downloads_cmip6'

@patch('gridflow.processing.temporal_aggregate.stop_event')
def test_main_interrupted_exit_code(mock_stop, mocker):
    """Verify main exits with code 130 on user interrupt."""
    mocker.patch('sys.argv', ['script.py', '--demo'])
    mocker.patch('gridflow.processing.temporal_aggregate.run_aggregator_session')
    mocker.patch('gridflow.processing.temporal_aggregate.setup_logging')
    
    mock_stop.is_set.return_value = True
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 130

if __name__ == "__main__":
    pytest.main([__file__])