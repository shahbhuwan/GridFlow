# tests/test_unit_convert.py

import argparse
import logging
import signal
import sys
import threading
import os
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import pytest
import numpy as np

# Updated imports to match the new gridflow.processing.unit_convert structure
import gridflow.processing.unit_convert as unit_convert
from gridflow.processing.unit_convert import (
    k_to_c,
    flux_to_mm_day,
    m_s_to_km_h,
    Converter,
    FileManager,
    create_conversion_session,
    signal_handler,
    add_arguments,
    main,
    CONVERSIONS,
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
def file_manager(tmp_path):
    """Fixture for a FileManager instance."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    return FileManager(str(input_dir), str(output_dir))

@pytest.fixture
def converter(file_manager, mock_stop_event):
    """Fixture for a Converter instance."""
    settings = {
        'workers': 1,
        'variable': 'tas',
        'target_unit': 'C',
        'is_gui_mode': False
    }
    return Converter(file_manager, mock_stop_event, **settings)

@pytest.fixture
def mock_nc_dataset(mocker):
    """Fixture to mock netCDF4.Dataset for a successful conversion."""
    mock_src = MagicMock()
    mock_dst = MagicMock()
    
    # Mock the context manager `with nc.Dataset(...) as src:`
    mocker.patch('gridflow.processing.unit_convert.nc.Dataset', side_effect=[mock_src, mock_dst])
    mock_src.__enter__.return_value = mock_src
    mock_dst.__enter__.return_value = mock_dst
    
    # --- Configure Source Mock (input file) ---
    dim_time = MagicMock()
    dim_time.isunlimited.return_value = True
    dim_time.__len__.return_value = 1

    # Target variable to be converted
    var_to_convert = MagicMock()
    var_to_convert.units = 'K'
    var_to_convert.ncattrs.return_value = ['units', '_FillValue']
    var_to_convert.getncattr.side_effect = lambda k: 'K' if k == 'units' else -999.0
    var_to_convert.__getitem__.return_value = np.array([273.15])
    var_to_convert.dtype = np.float32
    var_to_convert.dimensions = ('time',)

    # Another variable that should just be copied
    other_var = MagicMock()
    other_var.ncattrs.return_value = []
    other_var.__getitem__.return_value = np.array([10])
    other_var.dtype = np.float32
    other_var.dimensions = ('time',)
    
    mock_src.configure_mock(
        file_format='NETCDF4',
        dimensions={'time': dim_time},
        variables={'tas': var_to_convert, 'other': other_var}
    )
    mock_src.__dict__.update({'global_attr': 'value'})

    # --- Configure Destination Mock (output file) ---
    mock_create_var = MagicMock()
    mock_dst.createVariable.return_value = mock_create_var
    
    return mock_nc_dataset, mock_src, mock_dst, mock_create_var

# ############################################################################
# Tests for Conversion Logic
# ############################################################################

def test_k_to_c():
    assert k_to_c(273.15) == 0
    assert k_to_c(373.15) == 100

def test_flux_to_mm_day():
    assert flux_to_mm_day(1) == 86400
    assert flux_to_mm_day(0.5) == 43200

def test_m_s_to_km_h():
    assert m_s_to_km_h(1) == 3.6
    assert m_s_to_km_h(10) == 36

# ############################################################################
# Tests for signal_handler
# ############################################################################

def test_signal_handler(caplog, mocker):
    """Test that the signal handler sets the stop event and logs messages."""
    caplog.set_level(logging.WARNING)
    mock_event = mocker.patch('gridflow.processing.unit_convert.stop_event')
    signal_handler(None, None)
    assert mock_event.set.called
    assert "Stop signal received" in caplog.text

# ############################################################################
# Tests for FileManager
# ############################################################################

def test_file_manager_discovery(file_manager, tmp_path):
    nc_file = Path(file_manager.input_dir) / "test.nc"
    nc_file.touch()
    txt_file = Path(file_manager.input_dir) / "ignore.txt"
    txt_file.touch()
    
    files = file_manager.get_netcdf_files()
    assert len(files) == 1
    assert files[0].name == "test.nc"

def test_file_manager_output_path(file_manager):
    input_path = Path(file_manager.input_dir) / "subdir" / "data.nc"
    output_path = file_manager.get_output_path(input_path)
    assert output_path.name == "data_converted.nc"
    assert "subdir" in str(output_path.parent)

# ############################################################################
# Tests for Converter.convert_file
# ############################################################################

def test_convert_file_success(converter, mock_nc_dataset, caplog):
    """Test a successful unit conversion of a single file."""
    caplog.set_level(logging.INFO)
    input_path = Path(converter.file_manager.input_dir) / "input.nc"
    input_path.touch()
    
    _, _, mock_dst, mock_create_var = mock_nc_dataset

    success, msg = converter.convert_file(input_path)

    assert success is True
    assert "Converted" in msg
    
    # Verify metadata and dimensions were copied
    assert mock_dst.setncatts.call_args[0][0]['global_attr'] == 'value'
    mock_dst.createDimension.assert_called_with('time', None)

    # Verify converted variable was written correctly
    mock_dst.createVariable.assert_any_call('tas', np.float32, ('time',), fill_value=-999.0)
    # Check that the converted data (0.0) was written
    np.testing.assert_array_almost_equal(mock_create_var.__setitem__.call_args_list[0].args[1], np.array([0.0]))
    assert mock_create_var.units == 'C'

def test_convert_file_no_conversion_defined(converter, caplog):
    """Test skipping a file if the conversion is not defined."""
    input_path = Path(converter.file_manager.input_dir) / "in.nc"
    
    converter.settings['variable'] = 'nonexistent_var'
    success, msg = converter.convert_file(input_path)
    
    assert not success
    assert "No conversion defined" in msg

def test_convert_file_variable_not_found(converter, mock_nc_dataset, caplog):
    """Test skipping a file if the target variable is not inside it."""
    input_path = Path(converter.file_manager.input_dir) / "in.nc"
    
    _, mock_src, _, _ = mock_nc_dataset
    mock_src.variables = {'other': MagicMock()} 
    
    success, msg = converter.convert_file(input_path)
    
    assert not success
    assert "Variable 'tas' not found" in msg

def test_convert_file_unit_mismatch(converter, mock_nc_dataset, caplog):
    input_path = Path(converter.file_manager.input_dir) / "in.nc"
    
    _, mock_src, _, _ = mock_nc_dataset
    mock_src.variables['tas'].units = 'm/s' 
    
    success, msg = converter.convert_file(input_path)
    
    assert not success
    assert "Unit mismatch" in msg

def test_convert_file_interrupted(converter, mock_stop_event):
    """Test that the function exits early if the stop event is set."""
    mock_stop_event.is_set.return_value = True
    success, msg = converter.convert_file(Path("in.nc"))
    assert not success
    assert msg == "Interrupted"

def test_convert_file_exception_handling(mocker, converter, caplog):
    """Test that exceptions are caught and logged."""
    mocker.patch('gridflow.processing.unit_convert.nc.Dataset', side_effect=Exception("Disk error"))
    input_path = Path(converter.file_manager.input_dir) / "in.nc"
    input_path.touch()
    
    success, msg = converter.convert_file(input_path)
    assert not success
    assert "Disk error" in msg

# ############################################################################
# Tests for Converter.process_all
# ############################################################################

def test_process_all_no_files(converter):
    """Test process_all with an empty file list."""
    successful, total = converter.process_all([])
    assert successful == 0
    assert total == 0

def test_process_all_success(mocker, converter):
    """Test a successful run of process_all."""
    mock_tpe = mocker.patch('gridflow.processing.unit_convert.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    # Simulate a future that returns True (success)
    mock_future = MagicMock(spec=Future)
    mock_future.result.return_value = (True, "Success")
    mock_executor.submit.return_value = mock_future
    mocker.patch('gridflow.processing.unit_convert.as_completed', return_value=[mock_future])
    
    files = [Path('in.nc')]
    successful, total = converter.process_all(files)
    
    assert successful == 1
    assert total == 1

def test_process_all_interrupted(mocker, converter, mock_stop_event):
    """Test interruption during process_all."""
    mock_tpe = mocker.patch('gridflow.processing.unit_convert.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    
    f1 = MagicMock(spec=Future)
    f1.result.return_value = (True, "S1")
    f2 = MagicMock(spec=Future)
    f2.result.return_value = (True, "S2")
    mock_executor.submit.side_effect = [f1, f2]

    def mock_as_completed(fs):
        yield f1
        mock_stop_event.is_set.return_value = True
        yield f2

    mocker.patch('gridflow.processing.unit_convert.as_completed', side_effect=mock_as_completed)
    
    files = [Path('in1.nc'), Path('in2.nc')]
    successful, total = converter.process_all(files)
    
    assert successful == 1 # Second task skipped due to stop_event
    assert total == 2

# ############################################################################
# Tests for create_conversion_session
# ############################################################################

@patch('gridflow.processing.unit_convert.Converter')
@patch('gridflow.processing.unit_convert.FileManager')
def test_create_conversion_session_success(MockFM, MockConv, mock_stop_event, caplog):
    """Test a successful end-to-end conversion session."""
    caplog.set_level(logging.INFO)
    
    mock_fm_instance = MockFM.return_value
    mock_fm_instance.get_netcdf_files.return_value = [Path('file.nc')]
    
    mock_conv_instance = MockConv.return_value
    mock_conv_instance.process_all.return_value = (1, 1)
    
    settings = {
        'input_dir': 'in', 'output_dir': 'out', 
        'variable': 'tas', 'target_unit': 'C', 'is_gui_mode': False
    }
    
    create_conversion_session(settings, mock_stop_event)
    
    assert "Completed: 1/1 files converted successfully" in caplog.text

# ############################################################################
# Tests for Argument Parsing and Main
# ############################################################################

def test_add_arguments():
    """Test that arguments are added to the parser correctly."""
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([
        '--input_dir', 'in',
        '--output_dir', 'out',
        '--variable', 'tas',
        '--target_unit', 'C',
        '--workers', '4'
    ])
    assert args.input_dir == 'in'
    assert args.output_dir == 'out'
    assert args.variable == 'tas'
    assert args.target_unit == 'C'
    assert args.workers == 4

@patch('argparse.ArgumentParser.parse_args')
@patch('gridflow.processing.unit_convert.setup_logging')
@patch('gridflow.processing.unit_convert.create_conversion_session')
@patch('signal.signal')
def test_main_success(mock_signal, mock_session, mock_logging, mock_args_call):
    """Test the main function with valid arguments."""
    mock_args = argparse.Namespace(
        input_dir='in', output_dir='out', variable='tas', target_unit='C',
        log_dir='./logs', log_level='verbose', demo=False, config=None,
        is_gui_mode=False, workers=4
    )
    mock_args_call.return_value = mock_args
    
    main()
    
    mock_logging.assert_called_with('./logs', 'verbose', prefix="unit_converter")
    mock_session.assert_called()

@patch('argparse.ArgumentParser.parse_args')
@patch('gridflow.processing.unit_convert.create_conversion_session')
@patch('gridflow.processing.unit_convert.setup_logging')
def test_main_demo_mode(mock_logging, mock_session, mock_args_call, caplog, capsys):
    """Test the main function in demo mode."""
    caplog.set_level(logging.INFO)
    mock_args = argparse.Namespace(
        log_dir='./logs', log_level='verbose', demo=True, config=None, 
        is_gui_mode=False, input_dir=None, output_dir=None, variable=None, 
        target_unit=None, workers=4
    )
    mock_args_call.return_value = mock_args
    
    main()
    
    # Check output for demo mode indication
    captured = capsys.readouterr()
    found_demo_msg = "demo mode" in caplog.text.lower() or "demo mode" in captured.out.lower()
    assert found_demo_msg
    
    mock_session.assert_called()
    called_settings = mock_session.call_args[0][0]
    assert called_settings['variable'] == 'tas'
    assert called_settings['target_unit'] == 'C'

@patch('gridflow.processing.unit_convert.stop_event')
def test_main_interrupted(mock_stop_event, mocker):
    """Test that main exits with code 130 if interrupted."""
    mocker.patch('sys.argv', ['script.py', '--demo'])
    mocker.patch('gridflow.processing.unit_convert.create_conversion_session')
    mocker.patch('gridflow.processing.unit_convert.setup_logging')
    mock_stop_event.is_set.return_value = True # Simulate interruption
    
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 130

if __name__ == "__main__":
    pytest.main([__file__])