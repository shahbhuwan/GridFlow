# tests/test_crop_netcdf.py

import argparse
import logging
import signal
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import pytest
import numpy as np

import gridflow.processing.crop_netcdf as crop_netcdf
from gridflow.processing.crop_netcdf import (
    find_coordinate_vars,
    crop_single_file,
    Cropper,
    run_crop_session,
    signal_handler,
    add_arguments,
    main,
    HAS_UI_LIBS
)

# ############################################################################
# Fixtures
# ############################################################################

@pytest.fixture
def mock_stop_event():
    """Fixture for a mock threading.Event."""
    event = MagicMock(spec=threading.Event)
    event.is_set.return_value = False
    return event

@pytest.fixture
def mock_bounds():
    """Fixture for spatial cropping bounds."""
    return {'min_lat': 30.0, 'max_lat': 40.0, 'min_lon': -110.0, 'max_lon': -100.0}

@pytest.fixture
def cropper(mock_stop_event):
    """Fixture for a Cropper instance."""
    settings = {
        'workers': 1, 
        'bounds': {'min_lat': 0, 'max_lat': 90, 'min_lon': -180, 'max_lon': 180},
        'is_gui_mode': False
    }
    return Cropper(settings, mock_stop_event)

# ############################################################################
# Tests for Helpers & Signal Handling
# ############################################################################

def test_signal_handler(caplog, mocker):
    """Verify stop_event is set on SIGINT."""
    caplog.set_level(logging.WARNING)
    mock_event = mocker.patch('gridflow.processing.crop_netcdf.stop_event')
    signal_handler(None, None)
    assert mock_event.set.called
    assert "Stop signal received" in caplog.text

def test_find_coordinate_vars_success():
    """Verify identification of lat/lon via standard names."""
    mock_ds = MagicMock()
    mock_ds.variables = {
        'lat': MagicMock(standard_name='latitude'),
        'lon': MagicMock(standard_name='longitude'),
    }
    lat_var, lon_var = find_coordinate_vars(mock_ds)
    assert lat_var == 'lat'
    assert lon_var == 'lon'

# ############################################################################
# Tests for crop_single_file
# ############################################################################

def test_crop_single_file_success(mocker, mock_stop_event, mock_bounds, tmp_path):
    """Test successful cropping with coordinate wrapping and data verification."""
    input_path = tmp_path / "input.nc"
    input_path.touch()
    output_path = tmp_path / "output.nc"

    mock_src = MagicMock(name="SourceDS")
    mock_dst = MagicMock(name="DestDS")
    
    # Pass 1: Open src to read coords
    # Pass 2: Open src to read data, Open dst to write data
    mocker.patch('gridflow.processing.crop_netcdf.nc.Dataset', side_effect=[mock_src, mock_src, mock_dst])

    # Setup Source
    mock_src.__enter__.return_value = mock_src
    mock_src.file_format = 'NETCDF4'
    mock_src.Conventions = 'CF-1.7'
    mock_src.ncattrs.return_value = ['Conventions']
    mock_src.getncattr.side_effect = lambda k: 'CF-1.7' if k == 'Conventions' else None

    # Dimensions
    dim_lat = MagicMock(name="LatDim")
    dim_lat.__len__.return_value = 2
    dim_lat.isunlimited.return_value = False
    
    dim_lon = MagicMock(name="LonDim")
    dim_lon.__len__.return_value = 2
    dim_lon.isunlimited.return_value = False
    
    mock_src.dimensions = {'lat_dim': dim_lat, 'lon_dim': dim_lon}

    # Variables
    lat_var = MagicMock(name="LatVar")
    lat_var.dimensions = ('lat_dim',)
    lat_var.standard_name = 'latitude'
    lat_var.ndim = 1
    # Latitudes 35 and 36 fall within bounds [30, 40]
    lat_var.__getitem__.return_value = np.array([35.0, 36.0])
    lat_var.ncattrs.return_value = []
    
    lon_var = MagicMock(name="LonVar")
    lon_var.dimensions = ('lon_dim',)
    lon_var.standard_name = 'longitude'
    lon_var.ndim = 1
    # 251.0 (wrapped) is -109.0, which falls within bounds [-110, -100]
    lon_var.__getitem__.return_value = np.array([250.0, 251.0]) 
    lon_var.ncattrs.return_value = []
    
    data_var = MagicMock(name="DataVar")
    data_var.dimensions = ('lat_dim', 'lon_dim')
    data_var.dtype = np.float32
    data_var.ndim = 2
    data_var.ncattrs.return_value = ['_FillValue', 'units']
    data_var.getncattr.side_effect = lambda k: -999.0 if k == '_FillValue' else 'K'
    data_var.__getitem__.return_value = np.array([[1, 2], [3, 4]])

    mock_src.variables = {'lat': lat_var, 'lon': lon_var, 'tas': data_var}

    # Setup Destination
    mock_dst.__enter__.return_value = mock_dst
    mock_create_var = MagicMock(name="CreatedVar")
    mock_dst.createVariable.return_value = mock_create_var
    mock_dst.ncattrs.return_value = []

    # Execution
    success, msg = crop_single_file(input_path, output_path, mock_bounds, mock_stop_event)

    # Assertions
    assert success is True
    assert msg == "output.nc"
    
    # Verify fill value handled at creation for the data variable
    mock_dst.createVariable.assert_any_call('tas', np.float32, ('lat_dim', 'lon_dim'), fill_value=-999.0)
    
    # Verify coordinate variables were created
    mock_dst.createVariable.assert_any_call('lat', ANY, ('lat_dim',), fill_value=None)
    mock_dst.createVariable.assert_any_call('lon', ANY, ('lon_dim',), fill_value=None)

    # Verify that data was actually written to the destination variable
    # In our mock setup, since it's a full slice, it should be the whole array
    assert mock_create_var.__setitem__.called

def test_crop_single_file_no_coords(mocker, mock_stop_event, mock_bounds, tmp_path):
    """Verify failure when lat/lon are missing."""
    mocker.patch('gridflow.processing.crop_netcdf.nc.Dataset')
    success, msg = crop_single_file(tmp_path / "in.nc", tmp_path / "out.nc", mock_bounds, mock_stop_event)
    assert not success
    assert "Missing coordinate variables" in msg

def test_crop_single_file_interrupted(mocker, mock_stop_event, mock_bounds, tmp_path):
    """Verify graceful exit when stop signal is detected."""
    mock_stop_event.is_set.return_value = True
    success, msg = crop_single_file(Path("in.nc"), Path("out.nc"), mock_bounds, mock_stop_event)
    assert not success
    assert msg == "Interrupted"

# ############################################################################
# Tests for Cropper Class
# ############################################################################

def test_cropper_crop_all_success(mocker, cropper):
    """Verify parallel execution orchestration and success count."""
    mock_tpe = mocker.patch('gridflow.processing.crop_netcdf.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    
    mock_future = MagicMock(spec=Future)
    mock_future.result.return_value = (True, "file_cropped.nc")
    mock_executor.submit.return_value = mock_future
    mocker.patch('gridflow.processing.crop_netcdf.as_completed', return_value=[mock_future])
    
    tasks = [(Path('in.nc'), Path('out.nc'))]
    success_count, total = cropper.crop_all(tasks)
    
    assert success_count == 1
    assert total == 1

# ############################################################################
# Tests for Session & CLI
# ############################################################################

def test_run_crop_session_discovery(mocker, mock_stop_event, tmp_path, caplog, capsys):
    """Verify file discovery and task creation in session."""
    caplog.set_level(logging.INFO)
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "test.nc").touch()

    settings = {
        'input_dir': str(in_dir),
        'output_dir': str(tmp_path / "out"),
        'bounds': {'min_lat': 0, 'max_lat': 10, 'min_lon': 0, 'max_lon': 10},
        'is_gui_mode': False
    }

    mock_cropper_instance = mocker.patch('gridflow.processing.crop_netcdf.Cropper').return_value
    mock_cropper_instance.crop_all.return_value = (1, 1)

    run_crop_session(settings, mock_stop_event)

    captured = capsys.readouterr()
    # Check for Rich console output if UI libs are present, otherwise log
    if HAS_UI_LIBS:
        assert "Found 1 NetCDF files" in captured.out
    else:
        assert "Found 1 NetCDF files" in caplog.text

@patch('argparse.ArgumentParser.parse_args')
@patch('gridflow.processing.crop_netcdf.setup_logging')
@patch('gridflow.processing.crop_netcdf.run_crop_session')
@patch('signal.signal')
def test_main_success(mock_signal, mock_session, mock_setup, mock_args_call):
    """Verify CLI main entry point logic."""
    mock_args = argparse.Namespace(
        input_dir='in', output_dir='out', 
        min_lat=0.0, max_lat=10.0, min_lon=0.0, max_lon=10.0,
        log_dir='./logs', log_level='verbose', demo=False,
        is_gui_mode=False, workers=4
    )
    mock_args_call.return_value = mock_args
    
    main()
    mock_setup.assert_called_once()
    mock_session.assert_called_once()
    mock_signal.assert_called_once()

@patch('argparse.ArgumentParser.parse_args')
@patch('gridflow.processing.crop_netcdf.run_crop_session')
@patch('gridflow.processing.crop_netcdf.setup_logging')
def test_main_demo_mode(mock_setup, mock_session, mock_args_call, capsys, caplog):
    """Verify demo mode defaults and command reporting."""
    caplog.set_level(logging.INFO)
    mock_args = argparse.Namespace(
        demo=True, log_dir='./logs', log_level='verbose',
        is_gui_mode=False, input_dir=None, output_dir=None,
        min_lat=None, max_lat=None, min_lon=None, max_lon=None
    )
    mock_args_call.return_value = mock_args
    
    main()
    
    captured = capsys.readouterr()
    if HAS_UI_LIBS:
        assert "Running in demo mode" in captured.out
        assert "gridflow crop" in captured.out
    else:
        assert "Running in demo mode" in caplog.text
    
    mock_session.assert_called_once()
    called_settings = mock_session.call_args[0][0]
    assert called_settings['bounds']['min_lat'] == 25.0

@patch('gridflow.processing.crop_netcdf.stop_event')
def test_main_interrupted_exit_code(mock_stop, mocker):
    """Verify main exits with code 130 on interrupt."""
    mocker.patch('sys.argv', ['script.py', '--demo'])
    mocker.patch('gridflow.processing.crop_netcdf.run_crop_session')
    mocker.patch('gridflow.processing.crop_netcdf.setup_logging')
    
    mock_stop.is_set.return_value = True
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 130

if __name__ == "__main__":
    pytest.main([__file__])