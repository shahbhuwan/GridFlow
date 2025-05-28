import pytest
import json
import netCDF4
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging
from gridflow.catalog_generator import extract_metadata, is_non_prefixed_filename, get_base_filename, generate_catalog

@pytest.fixture
def mock_netcdf_file(tmp_path):
    """Create a single mock NetCDF file with valid metadata."""
    file_path = tmp_path / "test.nc"
    with netCDF4.Dataset(file_path, "w") as ds:
        ds.activity_id = "CMIP"
        ds.source_id = "CESM2"
        ds.variant_label = "r1i1p1f1"
        ds.variable_id = "tas"
        ds.institution_id = "NCAR"
    return file_path

@pytest.fixture
def mock_netcdf_files(tmp_path):
    """Create multiple mock NetCDF files with varying metadata, including duplicates."""
    input_dir = tmp_path / "input"
    sub_dir = input_dir / "subdir"
    input_dir.mkdir()
    sub_dir.mkdir()
    files = []
    # Non-prefixed file
    file_path = input_dir / "tas_model.nc"
    with netCDF4.Dataset(file_path, "w") as ds:
        ds.activity_id = "CMIP"
        ds.source_id = "Model0"
        ds.variant_label = "r1i1p1f1"
        ds.variable_id = "tas"
        ds.institution_id = "NCAR"
    files.append(file_path)
    # Prefixed duplicate in subdirectory
    file_path = sub_dir / "CMIP6_tas_model.nc"
    with netCDF4.Dataset(file_path, "w") as ds:
        ds.activity_id = "CMIP"
        ds.source_id = "Model0"
        ds.variant_label = "r1i1p1f1"
        ds.variable_id = "tas"
        ds.institution_id = "NCAR"
    files.append(file_path)
    # Another unique file
    file_path = input_dir / "pr_model.nc"
    with netCDF4.Dataset(file_path, "w") as ds:
        ds.activity_id = "CMIP"
        ds.source_id = "Model1"
        ds.variant_label = "r1i1p1f1"
        ds.variable_id = "pr"
        ds.institution_id = "NCAR"
    files.append(file_path)
    return input_dir, files

@pytest.fixture
def output_dir(tmp_path):
    """Create an output directory for test results."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

# Test extract_metadata
def test_extract_metadata_success(mock_netcdf_file):
    """Test extract_metadata with a valid NetCDF file."""
    result = extract_metadata(str(mock_netcdf_file))
    assert result["file_path"] == str(mock_netcdf_file)
    assert result["metadata"] == {
        "activity_id": "CMIP",
        "source_id": "CESM2",
        "variant_label": "r1i1p1f1",
        "variable_id": "tas",
        "institution_id": "NCAR"
    }
    assert result["error"] is None

def test_extract_metadata_invalid(tmp_path):
    """Test extract_metadata with a non-existent file."""
    invalid_file = tmp_path / "invalid.nc"
    result = extract_metadata(str(invalid_file))
    assert result["file_path"] == str(invalid_file)
    assert result["metadata"] == {}
    assert "File" in result["error"]
    assert "does not exist" in result["error"]

def test_extract_metadata_corrupt_file(tmp_path):
    """Test extract_metadata with a corrupt file."""
    corrupt_file = tmp_path / "corrupt.nc"
    corrupt_file.write_bytes(b"not a netcdf file")
    result = extract_metadata(str(corrupt_file))
    assert result["file_path"] == str(corrupt_file)
    assert result["metadata"] == {}
    assert "Failed to extract metadata" in result["error"]

def test_extract_metadata_partial_metadata(mock_netcdf_file):
    """Test extract_metadata with partial metadata."""
    with netCDF4.Dataset(mock_netcdf_file, "w") as ds:
        ds.activity_id = "CMIP"
        ds.variable_id = "tas"
    result = extract_metadata(str(mock_netcdf_file))
    assert result["metadata"] == {
        "activity_id": "CMIP",
        "source_id": "",
        "variant_label": "",
        "variable_id": "tas",
        "institution_id": ""
    }
    assert result["error"] is None

# Test is_non_prefixed_filename
def test_is_non_prefixed_filename():
    """Test is_non_prefixed_filename function."""
    assert is_non_prefixed_filename("tas_model.nc")
    assert is_non_prefixed_filename("pr_model.nc")
    assert not is_non_prefixed_filename("CMIP6_tas_model.nc")
    assert not is_non_prefixed_filename("random.nc")

# Test get_base_filename
def test_get_base_filename():
    """Test get_base_filename function."""
    assert get_base_filename("CMIP6_tas_model.nc") == "tas_model.nc"
    assert get_base_filename("ScenarioMIP_250km_model.nc") == "model.nc"
    assert get_base_filename("tas_model.nc") == "tas_model.nc"

# Test generate_catalog
def test_generate_catalog_single(mock_netcdf_file, output_dir, caplog):
    """Test generate_catalog with a single NetCDF file."""
    caplog.set_level(logging.INFO)
    input_dir = mock_netcdf_file.parent
    with patch("os.cpu_count", return_value=4):
        result = generate_catalog(str(input_dir), str(output_dir))
        assert len(result) == 1
        key = "CMIP:CESM2:r1i1p1f1"
        assert key in result
        assert result[key]["activity_id"] == "CMIP"
        assert result[key]["source_id"] == "CESM2"
        assert result[key]["variant_label"] == "r1i1p1f1"
        assert result[key]["institution_id"] == "NCAR"
        assert "variables" in result[key]
        assert result[key]["variables"]["tas"]["file_count"] == 1
        assert result[key]["variables"]["tas"]["files"][0]["path"] == str(mock_netcdf_file)
        assert (output_dir / "catalog.json").exists()
        assert "Generated catalog with 1 groups" in caplog.text
        assert "Extracting metadata for 1 unique NetCDF files" in caplog.text
        assert "Processing 1 metadata entries with 4 workers" in caplog.text

def test_generate_catalog_duplicates(mock_netcdf_files, output_dir, caplog):
    """Test generate_catalog with duplicate files (prefers non-prefixed)."""
    caplog.set_level(logging.INFO)  # Changed to INFO to capture all logs
    input_dir, _ = mock_netcdf_files
    with patch("os.cpu_count", return_value=4):
        result = generate_catalog(str(input_dir), str(output_dir))
        assert len(result) == 2
        assert "CMIP:Model0:r1i1p1f1" in result
        assert "CMIP:Model1:r1i1p1f1" in result
        assert result["CMIP:Model0:r1i1p1f1"]["variables"]["tas"]["files"][0]["path"].endswith("tas_model.nc")
        assert (output_dir / "duplicates.json").exists()
        with open(output_dir / "duplicates.json") as f:
            duplicates = json.load(f)
        assert len(duplicates) == 1
        assert duplicates[0]["file_path"].endswith("CMIP6_tas_model.nc")
        assert "Duplicate filename detected: CMIP6_tas_model.nc matches tas_model.nc" in caplog.text
        assert "Extracting metadata for 2 unique NetCDF files" in caplog.text
        assert "Processing 2 metadata entries with 4 workers" in caplog.text

def test_generate_catalog_no_files(tmp_path, output_dir, caplog):
    """Test generate_catalog with an empty input directory."""
    caplog.set_level(logging.WARNING)
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    result = generate_catalog(str(input_dir), str(output_dir))
    assert result == {}
    assert not (output_dir / "catalog.json").exists()
    assert "No NetCDF files found in" in caplog.text

def test_generate_catalog_invalid_input_dir(output_dir, caplog):
    """Test generate_catalog with a non-existent input directory."""
    caplog.set_level(logging.CRITICAL)
    input_dir = "non_existent_dir"
    result = generate_catalog(input_dir, str(output_dir))
    assert result == {}
    assert not (output_dir / "catalog.json").exists()
    assert "Input directory non_existent_dir does not exist" in caplog.text

def test_generate_catalog_demo_mode_no_files(tmp_path, output_dir, caplog):
    """Test generate_catalog in demo mode with no files."""
    caplog.set_level(logging.CRITICAL)
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    result = generate_catalog(str(input_dir), str(output_dir), demo_mode=True)
    assert result == {}
    assert not (output_dir / "cmip6_catalog.json").exists()
    assert f"No NetCDF files found in {input_dir}. Run 'gridflow download --demo' to generate sample files." in caplog.text

def test_generate_catalog_incomplete_metadata(mock_netcdf_file, output_dir, caplog):
    """Test generate_catalog with a file missing required metadata."""
    caplog.set_level(logging.INFO)  # Changed to INFO to capture all logs
    file_path = mock_netcdf_file
    with netCDF4.Dataset(file_path, "w") as ds:
        ds.activity_id = "CMIP"
        ds.source_id = ""
        ds.variant_label = "r1i1p1f1"
        ds.variable_id = "tas"
    input_dir = file_path.parent
    with patch("os.cpu_count", return_value=4):
        result = generate_catalog(str(input_dir), str(output_dir))
        assert result == {}
        assert (output_dir / "catalog.json").exists()
        assert "Skipping" in caplog.text
        assert "Incomplete metadata (missing: source_id)" in caplog.text
        assert "Extracting metadata for 1 unique NetCDF files" in caplog.text
        assert "Processing 1 metadata entries with 4 workers" in caplog.text

def test_generate_catalog_save_error(mock_netcdf_file, output_dir, caplog):
    """Test generate_catalog with failure to save catalog."""
    caplog.set_level(logging.INFO)  # Changed to INFO to capture all logs
    input_dir = mock_netcdf_file.parent
    with patch("builtins.open", side_effect=Exception("Write error")):
        with patch("os.cpu_count", return_value=4):
            result = generate_catalog(str(input_dir), str(output_dir))
            assert result == {}
            assert not (output_dir / "catalog.json").exists()
            assert "Failed to save catalog" in caplog.text
            assert "Extracting metadata for 1 unique NetCDF files" in caplog.text
            assert "Processing 1 metadata entries with 4 workers" in caplog.text

def test_generate_catalog_default_workers(mock_netcdf_file, output_dir, caplog):
    """Test generate_catalog with default workers (None)."""
    caplog.set_level(logging.INFO)
    input_dir = mock_netcdf_file.parent
    with patch("os.cpu_count", return_value=4):
        result = generate_catalog(str(input_dir), str(output_dir), workers=None)
        assert len(result) == 1
        key = "CMIP:CESM2:r1i1p1f1"
        assert key in result
        assert result[key]["variables"]["tas"]["file_count"] == 1
        assert (output_dir / "catalog.json").exists()
        assert "Extracting metadata for 1 unique NetCDF files" in caplog.text
        assert "Processing 1 metadata entries with 4 workers" in caplog.text
