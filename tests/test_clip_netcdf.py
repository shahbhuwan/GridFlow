# tests/test_clip_netcdf.py

import argparse
import logging
import signal
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import numpy as np

# Import from the new modular structure
import gridflow.processing.clip_netcdf as clip_module
from gridflow.processing.clip_netcdf import (
    Clipper,
    FileManager,
    create_clipping_session,
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
def file_manager(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return FileManager(str(input_dir), str(output_dir))

@pytest.fixture
def clipper(file_manager, mock_stop_event):
    settings = {'workers': 1, 'shapefile': 'test.shp', 'is_gui_mode': True}
    return Clipper(file_manager, mock_stop_event, **settings)

# ############################################################################
# Tests for signal_handler
# ############################################################################

def test_signal_handler(caplog):
    caplog.set_level(logging.INFO)
    stop_event.clear()
    signal_handler(None, None)
    assert stop_event.is_set()
    assert "Stop signal received" in caplog.text

# ############################################################################
# Tests for FileManager
# ############################################################################

def test_file_manager_init_error(tmp_path, mocker):
    mock_exit = mocker.patch("sys.exit")
    
    FileManager(str(tmp_path / "nonexistent"), str(tmp_path / "out"))
    
    assert mock_exit.called
    assert mock_exit.call_args == call(1)

def test_file_manager_get_files(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "f1.nc").touch()
    (in_dir / "sub").mkdir()
    (in_dir / "sub" / "f2.nc").touch()
    (in_dir / "ignored.txt").touch()
    
    fm = FileManager(str(in_dir), str(tmp_path / "out"))
    files = fm.get_netcdf_files()
    assert len(files) == 2
    assert all(f.suffix == ".nc" for f in files)

def test_file_manager_output_path(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    fm = FileManager(str(in_dir), str(tmp_path / "out"))
    
    input_file = in_dir / "sub" / "data.nc"
    expected = tmp_path / "out" / "sub" / "data_clipped.nc"
    assert fm.get_output_path(input_file) == expected

# ############################################################################
# Tests for Clipper.clip_file (The Core Engine)
# ############################################################################

def test_clip_file_success(mocker, clipper, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    input_path = tmp_path / "input" / "test.nc"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.touch()
    
    mock_src, mock_dst = MagicMock(), MagicMock()
    
    mocker.patch('gridflow.processing.clip_netcdf.nc.Dataset', side_effect=[mock_src, mock_src, mock_dst])
    
    clipper.shapefile_geom = MagicMock()
    clipper.shapefile_geom.bounds = (-110, 30, -100, 40)
    mock_src.__enter__.return_value = mock_src
    mock_src.file_format = 'NETCDF4'
    mock_src.ncattrs.return_value = []
    
    dim_lat, dim_lon = MagicMock(), MagicMock()
    dim_lat.__len__.return_value, dim_lat.isunlimited.return_value = 2, False
    dim_lon.__len__.return_value, dim_lon.isunlimited.return_value = 2, False
    mock_src.dimensions = {'lat': dim_lat, 'lon': dim_lon}

    mock_src.variables = {
        'lat': MagicMock(standard_name='latitude', __getitem__=lambda s, k: np.array([35.0, 36.0])),
        'lon': MagicMock(standard_name='longitude', __getitem__=lambda s, k: np.array([-105.0, -104.0])),
        'tas': MagicMock(dimensions=('lat', 'lon'), dtype=np.float32, ncattrs=lambda: [])
    }
    mock_dst.__enter__.return_value = mock_dst
    mocker.patch('gridflow.processing.clip_netcdf.contains_func', return_value=np.array([[True, True], [True, True]]))

    success, msg = clipper.clip_file(input_path)
    assert success is True
    assert "Clipped" in msg

def test_clip_file_no_coords(mocker, clipper, tmp_path):
    input_path = tmp_path / "input" / "test.nc"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.touch()
    mock_src = MagicMock()
    mocker.patch('gridflow.processing.clip_netcdf.nc.Dataset', return_value=MagicMock(__enter__=lambda s: mock_src))
    mock_src.variables = {'time': MagicMock()}
    success, msg = clipper.clip_file(input_path)
    assert success is False
    assert "No coordinates found" in msg

def test_clip_file_outside_bounds(mocker, clipper, tmp_path):
    input_path = tmp_path / "input" / "test.nc"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.touch()
    mock_src = MagicMock()
    mocker.patch('gridflow.processing.clip_netcdf.nc.Dataset', return_value=MagicMock(__enter__=lambda s: mock_src))
    clipper.shapefile_geom = MagicMock()
    clipper.shapefile_geom.bounds = (0, 0, 1, 1)
    mock_src.variables = {
        'lat': MagicMock(standard_name='latitude', __getitem__=lambda s, k: np.array([42.0])),
        'lon': MagicMock(standard_name='longitude', __getitem__=lambda s, k: np.array([-95.0]))
    }
    success, msg = clipper.clip_file(input_path)
    assert success is False
    assert "Outside bounds" in msg

def test_clip_file_interrupted(mocker, clipper, tmp_path, mock_stop_event):
    """Tests that clipping stops gracefully when the stop_event is set mid-loop."""
    input_path = tmp_path / "input" / "test.nc"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.touch()
    
    mock_src = MagicMock()
    
    mocker.patch('gridflow.processing.clip_netcdf.nc.Dataset', return_value=MagicMock(__enter__=lambda s: mock_src))
    
    clipper.shapefile_geom = MagicMock()
    clipper.shapefile_geom.bounds = (-180, -90, 180, 90)
    
    mock_src.variables = {
        'lat': MagicMock(__getitem__=lambda s, k: np.array([0]), standard_name='latitude'),
        'lon': MagicMock(__getitem__=lambda s, k: np.array([0]), standard_name='longitude'),
        'var1': MagicMock(),
        'var2': MagicMock()
    }
    mocker.patch('gridflow.processing.clip_netcdf.contains_func', return_value=np.array([[True]]))
    
    mock_stop_event.is_set.side_effect = [False, True]

    success, msg = clipper.clip_file(input_path)
    assert success is False
    assert "Interrupted" in msg

# ############################################################################
# Tests for Clipper.process_all (Parallelism)
# ############################################################################

def test_process_all_success(mocker, clipper):
    # Mock clip_file to return success
    mocker.patch.object(clipper, 'clip_file', return_value=(True, "Success"))
    
    files = [Path("f1.nc"), Path("f2.nc")]
    success, total = clipper.process_all(files)
    
    assert success == 2
    assert total == 2

def test_process_all_mixed_results(mocker, clipper, caplog):
    caplog.set_level(logging.WARNING)
    mocker.patch.object(clipper, 'clip_file', side_effect=[(True, "OK"), (False, "Bad")])
    files = [Path("f1.nc"), Path("f2.nc")]
    success, total = clipper.process_all(files)
    assert success == 1
    assert any("Failed f2.nc: Bad" in record.message for record in caplog.records)

def test_process_all_interrupted(mocker, clipper, mock_stop_event):
    mocker.patch.object(clipper, 'clip_file', return_value=(True, "OK"))
    
    # Interrupt after first file
    def side_effect(*args, **kwargs):
        mock_stop_event.is_set.return_value = True
        return (True, "OK")
    
    clipper.clip_file.side_effect = side_effect
    
    files = [Path("f1.nc"), Path("f2.nc"), Path("f3.nc")]
    success, total = clipper.process_all(files)
    
    # Should stop early
    assert success < 3

# ############################################################################
# Tests for Shapefile Loading
# ############################################################################

def test_load_shapefile_missing(clipper):
    clipper.settings['shapefile'] = "nonexistent.shp"
    assert clipper.load_shapefile() is False

def test_load_shapefile_success(mocker, clipper, tmp_path):
    shp = tmp_path / "test.shp"
    shp.touch()
    clipper.settings['shapefile'] = str(shp)
    mock_gdf = MagicMock()
    mock_crs = MagicMock()
    mock_crs.to_epsg.return_value = 4326
    mock_gdf.crs = mock_crs
    mock_gdf.union_all.return_value = "geometry"
    mocker.patch('gridflow.processing.clip_netcdf.gpd.read_file', return_value=mock_gdf)
    assert clipper.load_shapefile() is True
    assert clipper.shapefile_geom == "geometry"

def test_load_shapefile_with_buffer(mocker, clipper, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    shp = tmp_path / "test.shp"
    shp.touch()
    clipper.settings.update({'shapefile': str(shp), 'buffer_km': 10})
    mock_gdf = MagicMock()
    mock_crs = MagicMock()
    mock_crs.to_epsg.return_value = 4326
    mock_gdf.crs = mock_crs
    mock_gdf_m = MagicMock()
    mock_gdf.to_crs.side_effect = [mock_gdf_m, mock_gdf] 
    mocker.patch('gridflow.processing.clip_netcdf.gpd.read_file', return_value=mock_gdf)
    clipper.load_shapefile()
    assert "Applying 10km buffer" in caplog.text

# ############################################################################
# Tests for Session Orchestration
# ############################################################################

def test_create_clipping_session_no_files(mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    mock_exit = mocker.patch("sys.exit")
    
    settings = {'input_dir': str(in_dir), 'output_dir': str(tmp_path / "out"), 'shapefile': 's.shp'}
    create_clipping_session(settings, mock_stop_event)
    
    assert "No NetCDF files found" in caplog.text
    assert mock_exit.called
    assert mock_exit.call_args == call(0)

@patch('gridflow.processing.clip_netcdf.Clipper')
def test_create_clipping_session_load_fail(mock_clipper_cls, mock_stop_event, tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "f.nc").touch()
    
    mock_clip = mock_clipper_cls.return_value
    mock_clip.load_shapefile.return_value = False # Fail to load shp
    
    settings = {'input_dir': str(in_dir), 'output_dir': 'out', 'shapefile': 's.shp'}
    with pytest.raises(SystemExit):
        create_clipping_session(settings, mock_stop_event)

# ############################################################################
# CLI / Main Tests
# ############################################################################

def test_add_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['-i', 'in', '-o', 'out', '-s', 'shp', '--buffer_km', '5'])
    assert args.input_dir == 'in'
    assert args.buffer_km == 5.0

@patch('gridflow.processing.clip_netcdf.create_clipping_session')
@patch('gridflow.processing.clip_netcdf.setup_logging')
@patch('argparse.ArgumentParser.parse_args')
def test_main_demo(mock_args, mock_logging, mock_session, caplog, mocker):
    caplog.set_level(logging.INFO)
    mock_ns = MagicMock(demo=True, config=None, log_dir='l', log_level='v', is_gui_mode=True)
    mock_args.return_value = mock_ns
    
    mocker.patch('gridflow.processing.clip_netcdf.stop_event.is_set', return_value=False)
    
    main()
    assert "Running in demo mode" in caplog.text
    assert mock_session.called

@patch('gridflow.processing.clip_netcdf.setup_logging')
@patch('argparse.ArgumentParser.parse_args')
def test_main_missing_args(mock_args, mock_logging, caplog):
    caplog.set_level(logging.ERROR)

    mock_ns = MagicMock(
        demo=False,
        config=None,
        input_dir=None,
        output_dir=None,
        shapefile=None,
        is_gui_mode=False
    )
    mock_args.return_value = mock_ns

    with pytest.raises(SystemExit):
        main()

    assert "No shapefile selected" in caplog.text
