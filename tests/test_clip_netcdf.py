# tests/test_clip_netcdf.py

import argparse
import logging
import builtins
import signal
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

import gridflow.processing.clip_netcdf as clip_netcdf
from gridflow.processing.clip_netcdf import (
    clip_single_file,
    Clipper,
    run_clip_session,
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
def mock_clip_geom():
    geom = MagicMock()
    geom.bounds = (-110, 30, -100, 40)
    return geom

@pytest.fixture
def clipper(mock_stop_event):
    settings = {'workers': 1}
    return Clipper(settings, mock_stop_event)

# ############################################################################
# Tests for signal_handler
# ############################################################################

def test_signal_handler(caplog, mocker):
    caplog.set_level(logging.INFO)
    mock_event = mocker.patch('gridflow.processing.clip_netcdf.stop_event')
    signal_handler(None, None)
    assert mock_event.set.called
    assert "Stop signal received" in caplog.text
    assert "Please wait for ongoing tasks" in caplog.text

# ############################################################################
# Tests for clip_single_file
# ############################################################################

def test_clip_single_file_success(mocker, mock_stop_event, mock_clip_geom, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"

    # Patch netCDF4.Dataset to return different mocks for input and output
    mock_src = MagicMock()
    mock_dst = MagicMock()
    mock_dataset = mocker.patch('gridflow.processing.clip_netcdf.nc.Dataset', side_effect=[mock_src, mock_dst])

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
        'lat': dim_lat,
        'lon': dim_lon,
        'time': dim_time,
    }

    lat_mock = MagicMock()
    lat_mock.standard_name = 'latitude'
    lat_mock.__getitem__.return_value = np.array([35.0, 36.0])
    lat_mock.ncattrs.return_value = []

    lon_mock = MagicMock()
    lon_mock.standard_name = 'longitude'
    lon_mock.__getitem__.return_value = np.array([250.0, 251.0])
    lon_mock.ncattrs.return_value = []

    data_mock = MagicMock()
    data_mock.dimensions = ('time', 'lat', 'lon')
    data_mock.dtype = np.float32
    data_mock.__getitem__.return_value = np.ma.masked_array(data=np.array([[[1, 2], [3, 4]]]), mask=False)
    data_mock.ncattrs.return_value = ['_FillValue']
    data_mock.getncattr.side_effect = lambda k: -999.0 if k == '_FillValue' else 'attr'

    mock_src.variables = {
        'lat': lat_mock,
        'lon': lon_mock,
        'data': data_mock,
    }

    # Output file mock
    mock_dst.__enter__.return_value = mock_dst
    mock_dst.createDimension.return_value = None
    mock_dst.createVariable.return_value = MagicMock()

    # Patch contains_func to return spatial mask
    mocker.patch('gridflow.processing.clip_netcdf.contains_func', return_value=np.array([[True, False], [False, True]]))

    # Run the function
    result = clip_single_file(input_path, output_path, mock_clip_geom, mock_stop_event)

    # Assertions
    assert result is True
    assert "Successfully clipped" in caplog.text

def test_clip_single_file_no_intersect(mocker, mock_stop_event, mock_clip_geom, tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"

    mock_dataset = mocker.patch('gridflow.processing.clip_netcdf.nc.Dataset')
    mock_src = mock_dataset.return_value.__enter__.return_value
    mock_src.variables = {
        'lat': MagicMock(__getitem__=lambda self, key: np.array([1,2])),
        'lon': MagicMock(__getitem__=lambda self, key: np.array([3,4])),
    }
    mocker.patch('gridflow.processing.clip_netcdf.contains_func', return_value=np.array([[False, False], [False, False]]))
    assert not clip_single_file(input_path, output_path, mock_clip_geom, mock_stop_event)
    assert "No grid points" in caplog.text

def test_clip_single_file_no_coords(mocker, mock_stop_event, mock_clip_geom, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"

    mock_dataset = mocker.patch('gridflow.processing.clip_netcdf.nc.Dataset')
    mock_src = mock_dataset.return_value.__enter__.return_value
    mock_src.variables = {}
    assert not clip_single_file(input_path, output_path, mock_clip_geom, mock_stop_event)
    assert "Could not find coordinate variables" in caplog.text

def test_clip_single_file_interrupted(mocker, mock_stop_event, mock_clip_geom, tmp_path):
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"

    mock_dataset = mocker.patch('gridflow.processing.clip_netcdf.nc.Dataset')
    mock_src = mock_dataset.return_value.__enter__.return_value
    mock_src.variables = {
        'lat': MagicMock(__getitem__=lambda self, key: np.array([1])),
        'lon': MagicMock(__getitem__=lambda self, key: np.array([2])),
        'data1': MagicMock(),
        'data2': MagicMock(),
    }
    mock_src.dimensions = {}
    mocker.patch('gridflow.processing.clip_netcdf.contains_func', return_value=np.array([[True]]))
    mock_stop_event.is_set.side_effect = [False, False, True]  # Interrupt during var loop
    assert not clip_single_file(input_path, output_path, mock_clip_geom, mock_stop_event)

def test_clip_single_file_exception(mocker, mock_stop_event, mock_clip_geom, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"

    mocker.patch('gridflow.processing.clip_netcdf.nc.Dataset', side_effect=Exception("Fail"))
    assert not clip_single_file(input_path, output_path, mock_clip_geom, mock_stop_event)
    assert "Failed to clip" in caplog.text

# ############################################################################
# Tests for Clipper
# ############################################################################

def test_clipper_init(mock_stop_event):
    settings = {'workers': 2}
    clip = Clipper(settings, mock_stop_event)
    assert clip.settings == settings
    assert clip._stop_event == mock_stop_event
    assert clip.executor is None

def test_shutdown(clipper):
    clipper.executor = MagicMock()
    clipper.shutdown()
    clipper.executor.shutdown.assert_called_with(wait=True, cancel_futures=True)

def test_clip_all_no_files(clipper):
    successful, total = clipper.clip_all([], None)
    assert successful == 0
    assert total == 0

def test_clip_all_success(mocker, clipper):
    mock_tpe = mocker.patch('gridflow.processing.clip_netcdf.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    mock_futures = [MagicMock(spec=Future, result=lambda: True)]
    mock_executor.submit.return_value = mock_futures[0]
    mocker.patch('gridflow.processing.clip_netcdf.as_completed', return_value=mock_futures)
    successful, total = clipper.clip_all([(Path('in'), Path('out'))], MagicMock())
    assert successful == 1
    assert total == 1

def test_clip_all_failure(mocker, clipper):
    mock_tpe = mocker.patch('gridflow.processing.clip_netcdf.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    mock_futures = [MagicMock(spec=Future, result=lambda: False)]
    mock_executor.submit.return_value = mock_futures[0]
    mocker.patch('gridflow.processing.clip_netcdf.as_completed', return_value=mock_futures)
    successful, total = clipper.clip_all([(Path('in'), Path('out'))], MagicMock())
    assert successful == 0
    assert total == 1

def test_clip_all_interrupted(mocker, clipper, mock_stop_event):
    mock_tpe = mocker.patch('gridflow.processing.clip_netcdf.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    mock_futures = [MagicMock(spec=Future) for _ in range(2)]
    mock_futures[0].result.return_value = True
    mock_futures[1].result.return_value = True
    mock_executor.submit.side_effect = mock_futures

    def mock_as_completed(fs):
        yield mock_futures[0]
        mock_stop_event.is_set.return_value = True
        yield mock_futures[1]

    mocker.patch('gridflow.processing.clip_netcdf.as_completed', side_effect=mock_as_completed)
    successful, total = clipper.clip_all([(Path('in1'), Path('out1')), (Path('in2'), Path('out2'))], MagicMock())
    assert successful == 1
    assert total == 2

# ############################################################################
# Tests for run_clip_session
# ############################################################################

def test_run_clip_session_success(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    input_dir_path = tmp_path / 'input'
    input_dir_path.mkdir()
    (input_dir_path / 'file.nc').touch()
    output_dir_path = tmp_path / 'output'
    shapefile_path = tmp_path / 'shape.shp'
    shapefile_path.touch()
    settings = {
        'input_dir': str(input_dir_path),
        'output_dir': str(output_dir_path),
        'shapefile': str(shapefile_path),
        'buffer_km': 0,
        'workers': 1
    }
    mock_gpd = mocker.patch('gridflow.processing.clip_netcdf.gpd.read_file', return_value=MagicMock(crs='EPSG:4326', to_crs=lambda crs: MagicMock(unary_union=MagicMock())))
    mocker.patch.object(Clipper, 'clip_all', return_value=(1, 1))
    run_clip_session(settings, mock_stop_event)
    assert "Found 1 NetCDF files" in caplog.text
    assert "Completed: 1/1" in caplog.text

def test_run_clip_session_with_buffer(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    input_dir_path = tmp_path / 'input'
    input_dir_path.mkdir()
    (input_dir_path / 'file.nc').touch()
    output_dir_path = tmp_path / 'output'
    shapefile_path = tmp_path / 'shape.shp'
    shapefile_path.touch()
    settings = {
        'input_dir': str(input_dir_path),
        'output_dir': str(output_dir_path),
        'shapefile': str(shapefile_path),
        'buffer_km': 1,
        'workers': 1
    }
    mock_gdf = MagicMock(crs='EPSG:4326')
    mock_gdf_meters = MagicMock()
    mock_gdf_meters.__setitem__ = lambda self, key, value: None
    mock_gdf.to_crs.side_effect = [mock_gdf_meters, mock_gdf]
    mocker.patch('gridflow.processing.clip_netcdf.gpd.read_file', return_value=mock_gdf)
    run_clip_session(settings, mock_stop_event)
    assert "Applying 1km buffer" in caplog.text

def test_run_clip_session_no_files(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    input_dir_path = tmp_path / 'input'
    input_dir_path.mkdir()
    output_dir_path = tmp_path / 'output'
    shapefile_path = tmp_path / 'shape.shp'
    shapefile_path.touch()
    settings = {
        'input_dir': str(input_dir_path),
        'output_dir': str(output_dir_path),
        'shapefile': str(shapefile_path),
        'buffer_km': 0,
        'workers': 1
    }
    mocker.patch('gridflow.processing.clip_netcdf.gpd.read_file', return_value=MagicMock(crs='EPSG:4326', to_crs=lambda crs: MagicMock(unary_union=MagicMock())))
    with pytest.raises(SystemExit):
        run_clip_session(settings, mock_stop_event)
    assert "No NetCDF (.nc) files found" in caplog.text

def test_run_clip_session_invalid_input_dir(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    output_dir_path = tmp_path / 'output'
    shapefile_path = tmp_path / 'shape.shp'
    shapefile_path.touch()
    settings = {
        'input_dir': str(tmp_path / 'nonexist'),
        'output_dir': str(output_dir_path),
        'shapefile': str(shapefile_path),
        'buffer_km': 0,
    }
    with pytest.raises(SystemExit):
        run_clip_session(settings, mock_stop_event)
    assert "Input directory not found" in caplog.text

def test_run_clip_session_invalid_shapefile(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    input_dir_path = tmp_path / 'input'
    input_dir_path.mkdir()
    output_dir_path = tmp_path / 'output'
    settings = {
        'input_dir': str(input_dir_path),
        'output_dir': str(output_dir_path),
        'shapefile': str(tmp_path / 'nonexist.shp'),
        'buffer_km': 0,
    }
    with pytest.raises(SystemExit):
        run_clip_session(settings, mock_stop_event)
    assert "Shapefile not found" in caplog.text

def test_run_clip_session_exception(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.CRITICAL)
    input_dir_path = tmp_path / 'input'
    input_dir_path.mkdir()
    output_dir_path = tmp_path / 'output'
    shapefile_path = tmp_path / 'shape.shp'
    shapefile_path.touch()
    settings = {
        'input_dir': str(input_dir_path),
        'output_dir': str(output_dir_path),
        'shapefile': str(shapefile_path),
    }
    mocker.patch('gridflow.processing.clip_netcdf.gpd.read_file', side_effect=Exception("Critical"))
    run_clip_session(settings, mock_stop_event)
    assert "critical error" in caplog.text
    assert mock_stop_event.set.called

# ############################################################################
# Tests for add_arguments
# ############################################################################

def test_add_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--input_dir', './in', '--output_dir', './out', '--shapefile', 'shape.shp', '--workers', '2'])
    assert args.input_dir == './in'
    assert args.output_dir == './out'
    assert args.shapefile == 'shape.shp'
    assert args.workers == 2
    assert args.buffer_km == 0

# ############################################################################
# Tests for main
# ############################################################################

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.clip_netcdf.setup_logging')
@patch('gridflow.processing.clip_netcdf.run_clip_session')
@patch('signal.signal')
def test_main_success(mock_signal, mock_session, mock_logging, mock_parser):
    mock_args = MagicMock(input_dir='./in', output_dir='./out', shapefile='shape.shp', log_dir='./logs', log_level='info', demo=False)
    mock_parser.return_value.parse_args.return_value = mock_args
    main()
    mock_session.assert_called()
    mock_signal.assert_called_with(signal.SIGINT, signal_handler)

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.clip_netcdf.setup_logging')
@patch('gridflow.processing.clip_netcdf.run_clip_session')
def test_main_demo(mock_session, mock_logging, mock_parser, caplog):
    caplog.set_level(logging.INFO)
    mock_args = MagicMock(demo=True, log_dir='./logs', log_level='info')
    mock_parser.return_value.parse_args.return_value = mock_args
    main()
    assert "Running in demo mode" in caplog.text
    mock_session.assert_called()

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.clip_netcdf.setup_logging')
def test_main_no_params(mock_logging, mock_parser, caplog):
    caplog.set_level(logging.ERROR)
    mock_args = MagicMock(demo=False, input_dir=None, output_dir=None, shapefile=None)
    mock_parser.return_value.parse_args.return_value = mock_args
    with pytest.raises(SystemExit):
        main()
    assert "all --input_dir, --output_dir, and --shapefile arguments are required" in caplog.text

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.clip_netcdf.setup_logging')
@patch('gridflow.processing.clip_netcdf.stop_event')
def test_main_interrupted(mock_stop, mock_logging, mock_parser, caplog):
    caplog.set_level(logging.WARNING)
    mock_stop.is_set.return_value = True
    mock_parser.return_value.parse_args.return_value = MagicMock()
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 130
    assert "Execution was interrupted" in caplog.text