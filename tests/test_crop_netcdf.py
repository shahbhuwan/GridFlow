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
def mock_bounds():
    return {'min_lat': 30.0, 'max_lat': 40.0, 'min_lon': -110.0, 'max_lon': -100.0}

@pytest.fixture
def cropper(mock_stop_event):
    settings = {'workers': 1, 'bounds': {'min_lat': 0, 'max_lat': 90, 'min_lon': -180, 'max_lon': 180}}
    return Cropper(settings, mock_stop_event)

# ############################################################################
# Tests for signal_handler
# ############################################################################

def test_signal_handler(caplog, mocker):
    caplog.set_level(logging.INFO)
    mock_event = mocker.patch('gridflow.processing.crop_netcdf.stop_event')
    signal_handler(None, None)
    assert mock_event.set.called
    assert "Stop signal received" in caplog.text
    assert "Please wait for ongoing tasks" in caplog.text

# ############################################################################
# Tests for find_coordinate_vars
# ############################################################################

def test_find_coordinate_vars_success():
    mock_dataset = MagicMock()
    mock_dataset.variables = {
        'lat': MagicMock(standard_name='latitude'),
        'lon': MagicMock(standard_name='longitude'),
    }
    lat_var, lon_var = find_coordinate_vars(mock_dataset)
    assert lat_var == 'lat'
    assert lon_var == 'lon'

def test_find_coordinate_vars_fallback():
    mock_dataset = MagicMock()
    mock_dataset.variables = {
        'latitude': MagicMock(spec=[]),
        'longitude': MagicMock(spec=[]),
    }
    lat_var, lon_var = find_coordinate_vars(mock_dataset)
    assert lat_var == 'latitude'
    assert lon_var == 'longitude'

def test_find_coordinate_vars_partial():
    mock_dataset = MagicMock()
    mock_dataset.variables = {
        'lat': MagicMock(standard_name='latitude'),
    }
    lat_var, lon_var = find_coordinate_vars(mock_dataset)
    assert lat_var == 'lat'
    assert lon_var is None

def test_find_coordinate_vars_none():
    mock_dataset = MagicMock()
    mock_dataset.variables = {}
    lat_var, lon_var = find_coordinate_vars(mock_dataset)
    assert lat_var is None
    assert lon_var is None

# ############################################################################
# Tests for crop_single_file
# ############################################################################

def test_crop_single_file_success(mocker, mock_stop_event, mock_bounds, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"

    # Patch netCDF4.Dataset to return different mocks for input and output
    mock_src = MagicMock()
    mock_dst = MagicMock()
    mock_dataset = mocker.patch('gridflow.processing.crop_netcdf.nc.Dataset', side_effect=[mock_src, mock_dst])

    # Input file attributes
    mock_src.__enter__.return_value = mock_src
    mock_src.file_format = 'NETCDF4'
    mock_src.ncattrs.return_value = ['global_attr']
    mock_src.getncattr.return_value = 'value'

    dim_lat = MagicMock()
    dim_lat.__len__.return_value = 2
    dim_lat.isunlimited.return_value = False

    dim_lon = MagicMock()
    dim_lon.__len__.return_value = 2
    dim_lon.isunlimited.return_value = False

    dim_time = MagicMock()
    dim_time.__len__.return_value = 1
    dim_time.isunlimited.return_value = True

    mock_src.dimensions = {
        'lat_dim': dim_lat,
        'lon_dim': dim_lon,
        'time': dim_time,
    }

    lat_mock = MagicMock()
    lat_mock.standard_name = 'latitude'
    lat_mock.dimensions = ('lat_dim',)
    lat_mock.__getitem__.return_value = np.array([35.0, 36.0])
    lat_mock.ncattrs.return_value = []

    lon_mock = MagicMock()
    lon_mock.standard_name = 'longitude'
    lon_mock.dimensions = ('lon_dim',)
    lon_mock.__getitem__.return_value = np.array([250.0, 251.0])  # To test wrapping
    lon_mock.ncattrs.return_value = []

    data_mock = MagicMock()
    data_mock.dimensions = ('time', 'lat_dim', 'lon_dim')
    data_mock.dtype = np.float32
    data_mock.__getitem__.return_value = np.ma.masked_array(data=np.array([[[1, 2], [3, 4]]]), mask=False)
    data_mock.ncattrs.return_value = ['_FillValue', 'other_attr']
    data_mock.getncattr.side_effect = lambda k: -999.0 if k == '_FillValue' else 'attr'

    lat_only_mock = MagicMock()
    lat_only_mock.dimensions = ('lat_dim',)
    lat_only_mock.dtype = np.float32
    lat_only_mock.__getitem__.return_value = np.array([35.0, 36.0])
    lat_only_mock.ncattrs.return_value = []

    lon_only_mock = MagicMock()
    lon_only_mock.dimensions = ('lon_dim',)
    lon_only_mock.dtype = np.float32
    lon_only_mock.__getitem__.return_value = np.array([250.0, 251.0])
    lon_only_mock.ncattrs.return_value = []

    no_dim_mock = MagicMock()
    no_dim_mock.dimensions = ('time',)
    no_dim_mock.dtype = np.float32
    no_dim_mock.__getitem__.return_value = np.array([1])
    no_dim_mock.ncattrs.return_value = []

    mock_src.variables = {
        'lat': lat_mock,
        'lon': lon_mock,
        'data': data_mock,
        'lat_only': lat_only_mock,
        'lon_only': lon_only_mock,
        'no_dim': no_dim_mock,
    }

    # Output file mock
    mock_dst.__enter__.return_value = mock_dst
    mock_dst.createDimension.return_value = None
    mock_create_var = MagicMock()
    mock_dst.createVariable.return_value = mock_create_var

    # Run the function
    result = crop_single_file(input_path, output_path, mock_bounds, mock_stop_event)

    # Assertions
    assert result is True
    assert "Successfully cropped" in caplog.text
    mock_dst.createVariable.assert_any_call('data', np.float32, ('time', 'lat_dim', 'lon_dim'), fill_value=-999.0)
    mock_create_var.setncatts.assert_any_call({'other_attr': 'attr'})

def test_crop_single_file_no_coords(mocker, mock_stop_event, mock_bounds, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"

    mock_dataset = mocker.patch('gridflow.processing.crop_netcdf.nc.Dataset')
    mock_src = mock_dataset.return_value.__enter__.return_value = mock_dataset
    mock_src.variables = {}
    assert not crop_single_file(input_path, output_path, mock_bounds, mock_stop_event)
    assert "Could not find coordinate variables" in caplog.text

def test_crop_single_file_no_data_in_bounds(mocker, mock_stop_event, mock_bounds, tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"

    mock_dataset = mocker.patch('gridflow.processing.crop_netcdf.nc.Dataset')
    mock_src = mock_dataset.return_value.__enter__.return_value
    lat_mock = MagicMock(spec=['__getitem__'])
    lat_mock.__getitem__.return_value = np.array([1.0,2.0])
    lon_mock = MagicMock(spec=['__getitem__'])
    lon_mock.__getitem__.return_value = np.array([3.0,4.0])
    mock_src.variables = {'lat': lat_mock, 'lon': lon_mock}
    assert not crop_single_file(input_path, output_path, mock_bounds, mock_stop_event)
    assert "No data within the specified bounds" in caplog.text

def test_crop_single_file_interrupted(mocker, mock_stop_event, mock_bounds, tmp_path):
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"

    mock_dataset = mocker.patch('gridflow.processing.crop_netcdf.nc.Dataset')
    mock_src = mock_dataset.return_value.__enter__.return_value
    lat_mock = MagicMock(spec=['__getitem__'])
    lat_mock.__getitem__.return_value = np.array([35.0])
    lon_mock = MagicMock(spec=['__getitem__'])
    lon_mock.__getitem__.return_value = np.array([-105.0])
    mock_src.variables = {
        'lat': lat_mock,
        'lon': lon_mock,
        'data1': MagicMock(spec=[]),
        'data2': MagicMock(spec=[]),
    }
    mock_src.dimensions = {}
    mock_stop_event.is_set.side_effect = [False, False, True]  # Interrupt during var loop
    assert not crop_single_file(input_path, output_path, mock_bounds, mock_stop_event)

def test_crop_single_file_exception(mocker, mock_stop_event, mock_bounds, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"

    mocker.patch('gridflow.processing.crop_netcdf.nc.Dataset', side_effect=Exception("Fail"))
    assert not crop_single_file(input_path, output_path, mock_bounds, mock_stop_event)
    assert "Failed to crop" in caplog.text

def test_crop_single_file_no_fill_value(mocker, mock_stop_event, mock_bounds, tmp_path):
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"

    mock_src = MagicMock()
    mock_dst = MagicMock()
    mocker.patch('gridflow.processing.crop_netcdf.nc.Dataset', side_effect=[mock_src, mock_dst])

    mock_src.__enter__.return_value = mock_src
    mock_src.file_format = 'NETCDF4'
    mock_src.dimensions = {'lat_dim': MagicMock(), 'lon_dim': MagicMock()}

    # FIX: Added 'dtype' to spec and configured its value
    lat_mock = MagicMock(spec=['__getitem__', 'dimensions', 'standard_name', 'ncattrs', 'dtype'])
    lat_mock.__getitem__.return_value = np.array([35.0])
    lat_mock.dimensions = ('lat_dim',)
    lat_mock.standard_name = 'latitude'
    lat_mock.ncattrs.return_value = []
    lat_mock.dtype = np.float32

    # FIX: Added 'dtype' to spec and configured its value
    lon_mock = MagicMock(spec=['__getitem__', 'dimensions', 'standard_name', 'ncattrs', 'dtype'])
    lon_mock.__getitem__.return_value = np.array([-105.0])
    lon_mock.dimensions = ('lon_dim',)
    lon_mock.standard_name = 'longitude'
    lon_mock.ncattrs.return_value = []
    lon_mock.dtype = np.float32

    data_mock = MagicMock(spec=['ncattrs', 'getncattr', 'dimensions', 'dtype', '__getitem__'])
    data_mock.ncattrs.return_value = ['other_attr']
    data_mock.getncattr.side_effect = lambda k: 'attr'
    data_mock.dimensions = ('lat_dim', 'lon_dim')
    data_mock.dtype = np.float32
    data_mock.__getitem__.return_value = np.array([[1]])

    mock_src.variables = {
        'lat': lat_mock,
        'lon': lon_mock,
        'data': data_mock,
    }
    mock_dst.__enter__.return_value = mock_dst
    mock_dst.createDimension.return_value = None
    mock_create_var = MagicMock()
    mock_dst.createVariable.return_value = mock_create_var

    crop_single_file(input_path, output_path, mock_bounds, mock_stop_event)
    mock_dst.createVariable.assert_any_call('data', ANY, ANY, fill_value=None)
    mock_create_var.setncatts.assert_any_call({'other_attr': 'attr'})

# ############################################################################
# Tests for Cropper
# ############################################################################

def test_cropper_init(mock_stop_event):
    settings = {'workers': 2, 'bounds': {}}
    crop = Cropper(settings, mock_stop_event)
    assert crop.settings == settings
    assert crop._stop_event == mock_stop_event
    assert crop.executor is None

def test_shutdown(cropper):
    cropper.executor = MagicMock()
    cropper.shutdown()
    cropper.executor.shutdown.assert_called_with(wait=True, cancel_futures=True)

def test_crop_all_no_files(cropper):
    successful, total = cropper.crop_all([])
    assert successful == 0
    assert total == 0

def test_crop_all_success(mocker, cropper):
    mock_tpe = mocker.patch('gridflow.processing.crop_netcdf.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    mock_futures = [MagicMock(spec=Future, result=lambda: True)]
    mock_executor.submit.return_value = mock_futures[0]
    mocker.patch('gridflow.processing.crop_netcdf.as_completed', return_value=mock_futures)
    successful, total = cropper.crop_all([(Path('in'), Path('out'))])
    assert successful == 1
    assert total == 1

def test_crop_all_failure(mocker, cropper):
    mock_tpe = mocker.patch('gridflow.processing.crop_netcdf.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    mock_futures = [MagicMock(spec=Future, result=lambda: False)]
    mock_executor.submit.return_value = mock_futures[0]
    mocker.patch('gridflow.processing.crop_netcdf.as_completed', return_value=mock_futures)
    successful, total = cropper.crop_all([(Path('in'), Path('out'))])
    assert successful == 0
    assert total == 1

def test_crop_all_interrupted(mocker, cropper, mock_stop_event):
    mock_tpe = mocker.patch('gridflow.processing.crop_netcdf.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    mock_futures = [MagicMock(spec=Future) for _ in range(2)]
    mock_futures[0].result.return_value = True
    mock_futures[1].result.return_value = True
    mock_executor.submit.side_effect = mock_futures

    def mock_as_completed(fs):
        yield mock_futures[0]
        mock_stop_event.is_set.return_value = True
        yield mock_futures[1]

    mocker.patch('gridflow.processing.crop_netcdf.as_completed', side_effect=mock_as_completed)
    successful, total = cropper.crop_all([(Path('in1'), Path('out1')), (Path('in2'), Path('out2'))])
    assert successful == 1
    assert total == 2

# ############################################################################
# Tests for run_crop_session
# ############################################################################

def test_run_crop_session_success(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    input_dir_path = tmp_path / 'input'
    input_dir_path.mkdir()
    sub_dir = input_dir_path / 'sub'
    sub_dir.mkdir()
    (sub_dir / 'file.nc').touch()
    output_dir_path = tmp_path / 'output'
    settings = {
        'input_dir': str(input_dir_path),
        'output_dir': str(output_dir_path),
        'bounds': {'min_lat': 0, 'max_lat': 90, 'min_lon': -180, 'max_lon': 180},
        'workers': 1
    }
    mocker.patch.object(Cropper, 'crop_all', return_value=(1, 1))
    run_crop_session(settings, mock_stop_event)
    assert "Found 1 NetCDF files" in caplog.text
    assert "Completed: 1/1" in caplog.text

def test_run_crop_session_no_files(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    input_dir_path = tmp_path / 'input'
    input_dir_path.mkdir()
    output_dir_path = tmp_path / 'output'
    settings = {
        'input_dir': str(input_dir_path),
        'output_dir': str(output_dir_path),
        'bounds': {},
        'workers': 1
    }
    with pytest.raises(SystemExit):
        run_crop_session(settings, mock_stop_event)
    assert "No NetCDF (.nc) files found" in caplog.text

def test_run_crop_session_invalid_input_dir(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    output_dir_path = tmp_path / 'output'
    settings = {
        'input_dir': str(tmp_path / 'nonexist'),
        'output_dir': str(output_dir_path),
        'bounds': {},
    }
    with pytest.raises(SystemExit):
        run_crop_session(settings, mock_stop_event)
    assert "Input directory not found" in caplog.text

def test_run_crop_session_exception(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.CRITICAL)
    input_dir_path = tmp_path / 'input'
    input_dir_path.mkdir()
    output_dir_path = tmp_path / 'output'
    settings = {
        'input_dir': str(input_dir_path),
        'output_dir': str(output_dir_path),
        'bounds': {},
    }
    mocker.patch('pathlib.Path.rglob', side_effect=Exception("Critical"))
    run_crop_session(settings, mock_stop_event)
    assert "critical error" in caplog.text.lower()
    assert mock_stop_event.set.called

# ############################################################################
# Tests for add_arguments
# ############################################################################

def test_add_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--input_dir', './in', '--output_dir', './out', '--min_lat', '30', '--max_lat', '40', '--min_lon', '-110', '--max_lon', '-100', '--workers', '2'])
    assert args.input_dir == './in'
    assert args.output_dir == './out'
    assert args.min_lat == 30
    assert args.max_lat == 40
    assert args.min_lon == -110
    assert args.max_lon == -100
    assert args.workers == 2

# ############################################################################
# Tests for main
# ############################################################################

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.crop_netcdf.setup_logging')
@patch('gridflow.processing.crop_netcdf.run_crop_session')
@patch('signal.signal')
def test_main_success(mock_signal, mock_session, mock_logging, mock_parser):
    from argparse import Namespace
    mock_args = Namespace(input_dir='./in', output_dir='./out', min_lat=30, max_lat=40, min_lon=-110, max_lon=-100, log_dir='./logs', log_level='info', demo=False)
    mock_parser.return_value.parse_args.return_value = mock_args
    main()
    mock_session.assert_called()
    mock_signal.assert_called_with(signal.SIGINT, signal_handler)

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.crop_netcdf.setup_logging')
@patch('gridflow.processing.crop_netcdf.run_crop_session')
def test_main_demo(mock_session, mock_logging, mock_parser, caplog):
    caplog.set_level(logging.INFO)
    from argparse import Namespace
    mock_args = Namespace(demo=True, log_dir='./logs', log_level='info')
    mock_parser.return_value.parse_args.return_value = mock_args
    main()
    assert "Running in demo mode" in caplog.text
    mock_session.assert_called()

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.crop_netcdf.setup_logging')
def test_main_no_params(mock_logging, mock_parser, caplog):
    caplog.set_level(logging.ERROR)
    from argparse import Namespace
    
    # FIX: Added log_dir and log_level to the mock Namespace object
    mock_args = Namespace(
        demo=False, input_dir=None, output_dir=None, 
        min_lat=None, max_lat=None, min_lon=None, max_lon=None,
        log_dir='./logs', log_level='info'
    )
    mock_parser.return_value.parse_args.return_value = mock_args
    with pytest.raises(SystemExit):
        main()
    assert "all input/output and lat/lon arguments are required" in caplog.text

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.crop_netcdf.setup_logging')
@patch('gridflow.processing.crop_netcdf.stop_event')
@patch('gridflow.processing.crop_netcdf.run_crop_session')
def test_main_interrupted(mock_run, mock_stop, mock_logging, mock_parser, caplog):
    caplog.set_level(logging.WARNING)
    mock_stop.is_set.return_value = True
    from argparse import Namespace
    mock_parser.return_value.parse_args.return_value = Namespace(demo=False, input_dir='./in', output_dir='./out', min_lat=30.0, max_lat=40.0, min_lon=-110.0, max_lon=-100.0, log_dir='./logs', log_level='info')
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 130
    assert "Execution was interrupted" in caplog.text