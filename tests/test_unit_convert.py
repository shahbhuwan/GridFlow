# tests/test_unit_convert.py

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

# Assuming the script to be tested is in a package 'gridflow.processing'
# If it's a standalone script, you might need to adjust the import path.
import gridflow.processing.unit_convert as unit_convert
from gridflow.processing.unit_convert import (
    k_to_c,
    flux_to_mm_day,
    m_s_to_km_h,
    convert_single_file,
    UnitConverter,
    run_conversion_session,
    signal_handler,
    add_arguments,
    main,
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
def converter(mock_stop_event):
    """Fixture for a UnitConverter instance."""
    settings = {
        'workers': 1,
        'variable': 'tas',
        'target_unit': 'C'
    }
    return UnitConverter(settings, mock_stop_event)

# ############################################################################
# Tests for Conversion Logic
# ############################################################################

def test_k_to_c():
    assert k_to_c(273.15) == 0
    assert k_to_c(373.15) == 100

def test_flux_to_mm_day():
    # 1 kg m-2 s-1 is 86400 mm/day
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
    caplog.set_level(logging.INFO)
    mock_event = mocker.patch('gridflow.processing.unit_convert.stop_event')
    signal_handler(None, None)
    assert mock_event.set.called
    assert "Stop signal received" in caplog.text
    assert "Please wait for ongoing tasks" in caplog.text

# ############################################################################
# Tests for convert_single_file
# ############################################################################

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

def test_convert_single_file_success(mock_nc_dataset, mock_stop_event, tmp_path, caplog):
    """Test a successful unit conversion of a single file."""
    caplog.set_level(logging.INFO)
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "output.nc"
    
    _, _, mock_dst, mock_create_var = mock_nc_dataset

    result = convert_single_file(input_path, output_path, 'tas', 'C', mock_stop_event)

    assert result is True
    assert "Successfully converted 'tas'" in caplog.text
    
    # Verify metadata and dimensions were copied
    assert mock_dst.setncatts.call_args[0][0]['global_attr'] == 'value'
    mock_dst.createDimension.assert_called_with('time', None)

    # Verify converted variable was written correctly
    mock_dst.createVariable.assert_any_call('tas', np.float32, ('time',), fill_value=-999.0)
    # Check that the converted data (0.0) was written
    np.testing.assert_array_almost_equal(mock_create_var.__setitem__.call_args_list[0].args[1], np.array([0.0]))
    assert mock_create_var.units == 'C'

    # Verify the other variable was copied correctly
    mock_dst.createVariable.assert_any_call('other', np.float32, ('time',), fill_value=None)
    np.testing.assert_array_equal(mock_create_var.__setitem__.call_args_list[1].args[1], np.array([10]))

def test_convert_single_file_no_conversion_defined(mock_stop_event, tmp_path, caplog):
    """Test skipping a file if the conversion is not defined."""
    caplog.set_level(logging.ERROR)
    assert not convert_single_file(tmp_path / "in.nc", tmp_path / "out.nc", 'nonexistent_var', 'C', mock_stop_event)
    assert "No conversion defined for variable 'nonexistent_var'" in caplog.text

def test_convert_single_file_variable_not_found(mock_nc_dataset, mock_stop_event, tmp_path, caplog):
    """Test skipping a file if the target variable is not inside it."""
    caplog.set_level(logging.WARNING)
    _, mock_src, _, _ = mock_nc_dataset
    mock_src.variables = {'other': MagicMock()} # Remove 'tas'
    assert not convert_single_file(tmp_path / "in.nc", tmp_path / "out.nc", 'tas', 'C', mock_stop_event)
    assert "Variable 'tas' not found" in caplog.text

def test_convert_single_file_unit_mismatch(mock_nc_dataset, mock_stop_event, tmp_path, caplog):
    """Test skipping a file if the source unit does not match."""
    caplog.set_level(logging.ERROR)
    _, mock_src, _, _ = mock_nc_dataset
    mock_src.variables['tas'].units = 'm/s' # Mismatched unit
    assert not convert_single_file(tmp_path / "in.nc", tmp_path / "out.nc", 'tas', 'C', mock_stop_event)
    assert "Source unit mismatch" in caplog.text

def test_convert_single_file_interrupted(mock_stop_event, tmp_path):
    """Test that the function exits early if the stop event is set."""
    mock_stop_event.is_set.return_value = True
    assert not convert_single_file(tmp_path / "in.nc", tmp_path / "out.nc", 'tas', 'C', mock_stop_event)

def test_convert_single_file_exception_handling(mocker, mock_stop_event, tmp_path, caplog):
    """Test that exceptions are caught and logged."""
    caplog.set_level(logging.ERROR)
    mocker.patch('gridflow.processing.unit_convert.nc.Dataset', side_effect=Exception("Disk error"))
    output_path = tmp_path / "output.nc"
    output_path.touch() # Create a file to test deletion
    
    assert not convert_single_file(tmp_path / "in.nc", output_path, 'tas', 'C', mock_stop_event)
    assert "Failed to convert" in caplog.text
    assert "Disk error" in caplog.text
    assert not output_path.exists() # Ensure cleanup happens

# ############################################################################
# Tests for UnitConverter Class
# ############################################################################

def test_converter_init(mock_stop_event):
    """Test the constructor of the UnitConverter."""
    settings = {'workers': 2, 'variable': 'tas', 'target_unit': 'C'}
    converter = UnitConverter(settings, mock_stop_event)
    assert converter.settings == settings
    assert converter._stop_event == mock_stop_event
    assert converter.executor is None

def test_converter_shutdown(converter):
    """Test the shutdown method."""
    converter.executor = MagicMock()
    converter.shutdown()
    converter.executor.shutdown.assert_called_with(wait=True, cancel_futures=True)

def test_convert_all_no_files(converter):
    """Test convert_all with an empty file list."""
    successful, total = converter.convert_all([])
    assert successful == 0
    assert total == 0

def test_convert_all_success(mocker, converter):
    """Test a successful run of convert_all."""
    mock_tpe = mocker.patch('gridflow.processing.unit_convert.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    # Simulate a future that returns True (success)
    mock_future = MagicMock(spec=Future)
    mock_future.result.return_value = True
    mock_executor.submit.return_value = mock_future
    mocker.patch('gridflow.processing.unit_convert.as_completed', return_value=[mock_future])
    
    files = [(Path('in.nc'), Path('out.nc'))]
    successful, total = converter.convert_all(files)
    
    assert successful == 1
    assert total == 1

def test_convert_all_interrupted(mocker, converter, mock_stop_event):
    """Test interruption during convert_all."""
    mock_tpe = mocker.patch('gridflow.processing.unit_convert.ThreadPoolExecutor')
    mock_executor = mock_tpe.return_value
    
    # Simulate two tasks
    f1 = MagicMock(spec=Future, result=lambda: True)
    f2 = MagicMock(spec=Future, result=lambda: True)
    mock_executor.submit.side_effect = [f1, f2]

    # Mock as_completed to set the stop event after the first future completes
    def mock_as_completed(fs):
        yield f1
        mock_stop_event.is_set.return_value = True
        yield f2 # This one will be skipped by the loop break

    mocker.patch('gridflow.processing.unit_convert.as_completed', side_effect=mock_as_completed)
    
    files = [(Path('in1.nc'), Path('out1.nc')), (Path('in2.nc'), Path('out2.nc'))]
    successful, total = converter.convert_all(files)
    
    assert successful == 1 # Only the first one should complete
    assert total == 2

# ############################################################################
# Tests for run_conversion_session
# ############################################################################

@patch('gridflow.processing.unit_convert.UnitConverter')
def test_run_conversion_session_success(MockConverter, mock_stop_event, tmp_path, caplog):
    """Test a successful end-to-end conversion session."""
    caplog.set_level(logging.INFO)
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    (input_dir / 'file.nc').touch()
    
    settings = {
        'input_dir': str(input_dir),
        'output_dir': str(tmp_path / 'output'),
        'variable': 'tas',
        'target_unit': 'C',
        'workers': 1
    }
    
    mock_converter_instance = MockConverter.return_value
    mock_converter_instance.convert_all.return_value = (1, 1)
    
    run_conversion_session(settings, mock_stop_event)
    
    assert "Found 1 NetCDF files" in caplog.text
    assert "Completed: 1/1 files converted successfully" in caplog.text
    mock_converter_instance.convert_all.assert_called_once()

def test_run_conversion_session_invalid_input_dir(mock_stop_event, tmp_path, caplog):
    """Test session exit if input directory does not exist."""
    caplog.set_level(logging.ERROR)
    settings = {
        'input_dir': str(tmp_path / 'nonexistent'),
        'output_dir': str(tmp_path / 'output')
    }
    with pytest.raises(SystemExit):
        run_conversion_session(settings, mock_stop_event)
    assert "Input directory not found" in caplog.text

def test_run_conversion_session_no_files_found(mock_stop_event, tmp_path, caplog):
    """Test session exit if no NetCDF files are found."""
    caplog.set_level(logging.WARNING)
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    settings = {
        'input_dir': str(input_dir),
        'output_dir': str(tmp_path / 'output')
    }
    with pytest.raises(SystemExit):
        run_conversion_session(settings, mock_stop_event)
    assert "No NetCDF (.nc) files found" in caplog.text

def test_run_conversion_session_critical_error(mocker, mock_stop_event, tmp_path, caplog):
    """Test that the stop event is set on critical failure."""
    caplog.set_level(logging.CRITICAL)
    mocker.patch('pathlib.Path.is_dir', side_effect=Exception("Critical error"))
    settings = {'input_dir': 'any', 'output_dir': 'any'}
    run_conversion_session(settings, mock_stop_event)
    assert "A critical error occurred" in caplog.text
    assert mock_stop_event.set.called

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
    assert args.demo is False

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.unit_convert.setup_logging')
@patch('gridflow.processing.unit_convert.run_conversion_session')
@patch('signal.signal')
def test_main_success(mock_signal, mock_session, mock_logging, mock_parser):
    """Test the main function with valid arguments."""
    mock_args = argparse.Namespace(
        input_dir='in', output_dir='out', variable='tas', target_unit='C',
        log_dir='./logs', log_level='verbose', demo=False
    )
    mock_parser.return_value.parse_args.return_value = mock_args
    
    main()
    
    mock_logging.assert_called_with('./logs', 'verbose', prefix="unit_converter")
    mock_signal.assert_called_with(signal.SIGINT, signal_handler)
    mock_session.assert_called()

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.unit_convert.setup_logging')
@patch('gridflow.processing.unit_convert.run_conversion_session')
def test_main_demo_mode(mock_session, mock_logging, mock_parser, caplog):
    """Test the main function in demo mode."""
    caplog.set_level(logging.INFO)
    mock_args = argparse.Namespace(log_dir='./logs', log_level='verbose', demo=True)
    mock_parser.return_value.parse_args.return_value = mock_args
    
    main()
    
    assert "Running in demo mode" in caplog.text
    mock_session.assert_called()
    called_settings = mock_session.call_args[0][0]
    assert called_settings['variable'] == 'tas'
    assert called_settings['target_unit'] == 'C'

def test_main_missing_args(mocker, caplog):
    """Test that main exits if required arguments are missing."""
    caplog.set_level(logging.ERROR)
    # Patch sys.argv to simulate calling the script from the command line
    mocker.patch('sys.argv', ['script.py', '--input_dir', 'in'])
    
    # Argparse prints to stderr and exits, so we can't check caplog.
    # We just need to confirm that it exits.
    with pytest.raises(SystemExit):
        main()

@patch('gridflow.processing.unit_convert.stop_event')
def test_main_interrupted(mock_stop_event, mocker):
    """Test that main exits with code 130 if interrupted."""
    mocker.patch('sys.argv', ['script.py', '--demo'])
    mocker.patch('gridflow.processing.unit_convert.run_conversion_session')
    mocker.patch('gridflow.processing.unit_convert.setup_logging')
    mock_stop_event.is_set.return_value = True # Simulate interruption
    
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 130

# This allows running tests directly with `python test_unit_convert.py`
if __name__ == "__main__":
    pytest.main([__file__])