# tests/test_catalog_generator.py

import argparse
import logging
import signal
import sys
import threading
import json
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
import netCDF4 as nc

import gridflow.processing.catalog_generator as catalog_generator
from gridflow.processing.catalog_generator import (
    signal_handler,
    extract_metadata_from_file,
    Cataloger,
    run_catalog_session,
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
def cataloger(mock_stop_event):
    settings = {'input_dir': './input', 'output_dir': './output'}
    return Cataloger(settings, mock_stop_event)

@pytest.fixture
def mock_nc_dataset(mocker):
    mock_ds = MagicMock(spec=nc.Dataset)
    mock_ds.__enter__.return_value = mock_ds
    mock_ds.__exit__.return_value = None
    mocker.patch('gridflow.processing.catalog_generator.nc.Dataset', return_value=mock_ds)
    return mock_ds

# ############################################################################
# Tests for signal_handler
# ############################################################################

def test_signal_handler(caplog, mocker):
    caplog.set_level(logging.INFO)
    mock_event = mocker.patch('gridflow.processing.catalog_generator.stop_event')
    signal_handler(None, None)
    assert mock_event.set.called
    assert "Stop signal received" in caplog.text
    assert "Please wait for ongoing tasks" in caplog.text

# ############################################################################
# Tests for extract_metadata_from_file
# ############################################################################

def test_extract_metadata_from_file_success(mock_nc_dataset, tmp_path):
    file_path = tmp_path / "test.nc"
    mock_nc_dataset.activity_id = "CMIP"
    mock_nc_dataset.source_id = "Model"
    mock_nc_dataset.variant_label = "r1i1p1f1"
    mock_nc_dataset.variable_id = "tas"
    mock_nc_dataset.institution_id = "Inst"

    result = extract_metadata_from_file(file_path)
    assert result["file_path"] == str(file_path)
    assert result["metadata"] == {
        "activity_id": "CMIP",
        "source_id": "Model",
        "variant_label": "r1i1p1f1",
        "variable_id": "tas",
        "institution_id": "Inst"
    }
    assert result["error"] is None

def test_extract_metadata_from_file_partial(mock_nc_dataset, tmp_path):
    file_path = tmp_path / "test.nc"
    mock_nc_dataset.activity_id = "CMIP"
    mock_nc_dataset.source_id = "Model"

    result = extract_metadata_from_file(file_path)
    assert result["metadata"] == {
        "activity_id": "CMIP",
        "source_id": "Model",
        "variant_label": None,
        "variable_id": None,
        "institution_id": None
    }
    assert result["error"] is None

def test_extract_metadata_from_file_exception(mocker, tmp_path):
    mocker.patch('gridflow.processing.catalog_generator.nc.Dataset', side_effect=Exception("Read error"))
    file_path = tmp_path / "test.nc"

    result = extract_metadata_from_file(file_path)
    assert result["metadata"] == {}
    assert "Failed to read or extract metadata: Read error" in result["error"]

# ############################################################################
# Tests for Cataloger
# ############################################################################

def test_cataloger_init(mock_stop_event):
    settings = {'input_dir': './input', 'output_dir': './output'}
    cat = Cataloger(settings, mock_stop_event)
    assert cat.settings == settings
    assert cat._stop_event == mock_stop_event
    assert cat.catalog == {}
    assert cat.duplicates == []
    assert cat.skipped_count == 0
    assert cat.included_count == 0

def test_find_and_deduplicate_files_no_files(cataloger, tmp_path):
    cataloger.settings['input_dir'] = str(tmp_path)
    files = cataloger.find_and_deduplicate_files()
    assert files == []

def test_find_and_deduplicate_files_unique(cataloger, tmp_path):
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    (input_dir / 'file1.nc').touch()
    (input_dir / 'file2.nc').touch()
    cataloger.settings['input_dir'] = str(input_dir)
    files = cataloger.find_and_deduplicate_files()
    assert len(files) == 2
    assert len(cataloger.duplicates) == 0

def test_find_and_deduplicate_files_duplicates(cataloger, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    (input_dir / 'tas_file.nc').touch()
    (input_dir / 'ScenarioMIP_250km_tas_file.nc').touch()
    cataloger.settings['input_dir'] = str(input_dir)
    files = cataloger.find_and_deduplicate_files()
    assert len(files) == 1
    assert files[0].name == 'tas_file.nc'  # Prefer non-prefixed
    assert len(cataloger.duplicates) == 1
    assert "Duplicate found" in caplog.text

def test_find_and_deduplicate_files_stopped(cataloger, mock_stop_event, tmp_path):
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    (input_dir / 'file1.nc').touch()
    mock_stop_event.is_set.return_value = True
    cataloger.settings['input_dir'] = str(input_dir)
    files = cataloger.find_and_deduplicate_files()
    assert files == []  # Stopped early

def test_generate_no_files(cataloger, mocker, caplog):
    caplog.set_level(logging.WARNING)
    mocker.patch.object(cataloger, 'find_and_deduplicate_files', return_value=[])
    cataloger.generate()
    assert "No NetCDF files found" in caplog.text

def test_generate_success(cataloger, mocker, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    file_path = tmp_path / 'test.nc'
    mocker.patch.object(cataloger, 'find_and_deduplicate_files', return_value=[file_path])
    mocker.patch('gridflow.processing.catalog_generator.extract_metadata_from_file', return_value={
        "file_path": str(file_path),
        "metadata": {
            "activity_id": "CMIP",
            "source_id": "Model",
            "variant_label": "r1i1p1f1",
            "variable_id": "tas",
            "institution_id": "Inst"
        },
        "error": None
    })
    mocker.patch.object(cataloger, '_save_results')
    cataloger.generate()
    assert cataloger.included_count == 1
    assert cataloger.skipped_count == 0
    assert len(cataloger.catalog) == 1

def test_generate_incomplete_metadata(cataloger, mocker, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    file_path = tmp_path / 'test.nc'
    mocker.patch.object(cataloger, 'find_and_deduplicate_files', return_value=[file_path])
    mocker.patch('gridflow.processing.catalog_generator.extract_metadata_from_file', return_value={
        "file_path": str(file_path),
        "metadata": {"activity_id": "CMIP"},
        "error": None
    })
    mocker.patch.object(cataloger, '_save_results')
    cataloger.generate()
    assert cataloger.included_count == 0
    assert cataloger.skipped_count == 1
    assert "Incomplete metadata" in caplog.text

def test_generate_error(cataloger, mocker, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    file_path = tmp_path / 'test.nc'
    mocker.patch.object(cataloger, 'find_and_deduplicate_files', return_value=[file_path])
    mocker.patch('gridflow.processing.catalog_generator.extract_metadata_from_file', return_value={
        "file_path": str(file_path),
        "metadata": {},
        "error": "Error"
    })
    mocker.patch.object(cataloger, '_save_results')
    cataloger.generate()
    assert cataloger.included_count == 0
    assert cataloger.skipped_count == 1
    assert "Skipping" in caplog.text

def test_generate_stopped(cataloger, mocker, mock_stop_event, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    file1 = tmp_path / 'file1.nc'
    file2 = tmp_path / 'file2.nc'
    mocker.patch.object(cataloger, 'find_and_deduplicate_files', return_value=[file1, file2])
    mocker.patch('gridflow.processing.catalog_generator.extract_metadata_from_file', return_value={
        "file_path": str(file1),
        "metadata": {
            "activity_id": "CMIP",
            "source_id": "Model",
            "variant_label": "r1i1p1f1",
            "variable_id": "tas",
            "institution_id": "Inst"
        },
        "error": None
    })
    mock_stop_event.is_set.side_effect = [False, True]
    mocker.patch.object(cataloger, '_save_results')
    cataloger.generate()
    assert "Catalog generation stopped by user." in caplog.text
    assert cataloger.included_count == 1  # Processed one before stop

def test_process_metadata_result_success(cataloger):
    result = {
        "file_path": "/path/test.nc",
        "metadata": {
            "activity_id": "CMIP",
            "source_id": "Model",
            "variant_label": "r1i1p1f1",
            "variable_id": "tas",
            "institution_id": "Inst"
        },
        "error": None
    }
    cataloger._process_metadata_result(result)
    assert cataloger.included_count == 1
    assert cataloger.skipped_count == 0
    key = "CMIP:Model:r1i1p1f1"
    assert key in cataloger.catalog
    assert cataloger.catalog[key]["variables"]["tas"]["file_count"] == 1
    assert cataloger.catalog[key]["variables"]["tas"]["files"] == ["/path/test.nc"]

def test_process_metadata_result_incomplete(cataloger, caplog):
    caplog.set_level(logging.DEBUG)
    result = {
        "file_path": "/path/test.nc",
        "metadata": {"activity_id": "CMIP"},
        "error": None
    }
    cataloger._process_metadata_result(result)
    assert cataloger.included_count == 0
    assert cataloger.skipped_count == 1
    assert "Incomplete metadata" in caplog.text

def test_process_metadata_result_error(cataloger, caplog):
    caplog.set_level(logging.DEBUG)
    result = {
        "file_path": "/path/test.nc",
        "metadata": {},
        "error": "Error"
    }
    cataloger._process_metadata_result(result)
    assert cataloger.included_count == 0
    assert cataloger.skipped_count == 1
    assert "Skipping" in caplog.text

def test_save_results_success(cataloger, mocker, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    cataloger.settings['output_dir'] = str(tmp_path)
    cataloger.catalog = {"key": {}}
    cataloger.duplicates = ["/dup.nc"]
    cataloger.included_count = 1
    cataloger.skipped_count = 0
    mocker.patch('builtins.open', mock_open())
    
    with patch('gridflow.processing.catalog_generator.HAS_UI_LIBS', False):
        cataloger._save_results()
        
    assert "Catalog saved" in caplog.text
    assert "List of duplicate files saved" in caplog.text
    assert "Summary: Included 1 files" in caplog.text

def test_save_results_demo(cataloger, mocker, tmp_path):
    cataloger.settings['output_dir'] = str(tmp_path)
    cataloger.settings['demo'] = True
    cataloger.catalog = {"key": {}}
    mocker.patch('builtins.open', mock_open())
    cataloger._save_results()
    # Checks filename for demo
    open.assert_any_call(tmp_path / 'cmip6_catalog.json', 'w', encoding='utf-8')

def test_save_results_exception(cataloger, mocker, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    cataloger.settings['output_dir'] = str(tmp_path)
    cataloger.catalog = {"key": {}}
    mocker.patch('builtins.open', side_effect=IOError("Save error"))
    cataloger._save_results()
    assert "Failed to save catalog: Save error" in caplog.text

def test_is_non_prefixed():
    assert Cataloger._is_non_prefixed("tas_file.nc") is True
    assert Cataloger._is_non_prefixed("pr_file.nc") is True
    assert Cataloger._is_non_prefixed("ScenarioMIP_250km_file.nc") is False

def test_get_base_filename():
    assert Cataloger._get_base_filename("ScenarioMIP_250km_file.nc") == "file.nc"
    assert Cataloger._get_base_filename("CMIP6_file.nc") == "file.nc"
    assert Cataloger._get_base_filename("other_file.nc") == "other_file.nc"

# ############################################################################
# Tests for run_catalog_session
# ############################################################################

def test_run_catalog_session_success(mocker, mock_stop_event, tmp_path):
    mocker.patch.object(Cataloger, 'generate')
    settings = {'input_dir': str(tmp_path), 'output_dir': str(tmp_path)}
    run_catalog_session(settings, mock_stop_event)
    Cataloger.generate.assert_called_once()

def test_run_catalog_session_exception(mocker, mock_stop_event, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch.object(Cataloger, 'generate', side_effect=Exception("Critical"))
    run_catalog_session({}, mock_stop_event)
    assert "Failed: A critical error occurred" in caplog.text
    assert mock_stop_event.set.called

# ############################################################################
# Tests for add_arguments
# ############################################################################

def test_add_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--input_dir', './in', '--output_dir', './out'])
    assert args.input_dir == './in'
    assert args.output_dir == './out'

# ############################################################################
# Tests for main
# ############################################################################

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.catalog_generator.setup_logging')
@patch('gridflow.processing.catalog_generator.run_catalog_session')
@patch('signal.signal')
def test_main_success(mock_signal, mock_run_session, mock_setup_logging, mock_parser, caplog):
    """
    CLI mode (is_gui_mode=False): installs signal handler, calls session,
    and does NOT exit as long as a fresh stop_event remains clear.
    """
    caplog.set_level(logging.INFO)

    fresh_evt = threading.Event() 

    mock_args = MagicMock(
        input_dir='./in',
        output_dir='./out',
        log_dir='./logs',
        log_level='info',
        demo=False,
        is_gui_mode=False,   
        stop_event=fresh_evt, 
    )
    mock_parser.return_value.parse_args.return_value = mock_args

    from gridflow.processing.catalog_generator import main, signal_handler
    main()

    mock_setup_logging.assert_called_once_with('./logs', 'info', prefix='catalog_generator')
    mock_run_session.assert_called_once()
    mock_signal.assert_called_with(signal.SIGINT, signal_handler)
    assert "Execution was interrupted." not in caplog.text

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.catalog_generator.setup_logging')
@patch('gridflow.processing.catalog_generator.run_catalog_session')
def test_main_demo(mock_run_session, mock_setup_logging, mock_parser, caplog):
    """
    Demo + GUI mode (is_gui_mode=True): uses demo defaults, runs session,
    and never sys.exit(...) even if some other test toggled the module global.
    """
    caplog.set_level(logging.INFO)

    fresh_evt = threading.Event()

    mock_args = MagicMock(
        demo=True,
        log_dir='./logs',
        log_level='info',
        is_gui_mode=True,      
        stop_event=fresh_evt,   
        input_dir=None,         
        output_dir=None,
    )
    mock_parser.return_value.parse_args.return_value = mock_args

    from gridflow.processing.catalog_generator import main
    main()

    # FIX: In GUI mode, main() explicitly skips setup_logging
    mock_setup_logging.assert_not_called()
    mock_run_session.assert_called_once()
    assert "Running in demo mode." in caplog.text
    assert "Execution was interrupted." not in caplog.text

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.catalog_generator.setup_logging')
def test_main_no_params(mock_logging, mock_parser, caplog):
    caplog.set_level(logging.ERROR)
    mock_args = MagicMock(demo=False, input_dir=None, output_dir=None)
    mock_parser.return_value.parse_args.return_value = mock_args
    with pytest.raises(SystemExit):
        main()
    assert "Required arguments missing" in caplog.text

@patch('argparse.ArgumentParser')
@patch('gridflow.processing.catalog_generator.setup_logging')
@patch('gridflow.processing.catalog_generator.stop_event')
def test_main_interrupted(mock_stop, mock_logging, mock_parser, caplog, tmp_path):
    caplog.set_level(logging.INFO)
    mock_stop.is_set.return_value = True
    
    input_dir = tmp_path / 'in'
    input_dir.mkdir()

    mock_args = MagicMock(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / 'out'),
        demo=False,
        is_gui_mode=False
    )
    mock_parser.return_value.parse_args.return_value = mock_args

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 130
    assert "Execution was interrupted" in caplog.text