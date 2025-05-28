import shutil
import logging
import numpy as np
import netCDF4 as nc
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from gridflow.crop_netcdf import find_coordinate_vars, get_crop_indices, normalize_lon, crop_netcdf_file, crop_netcdf

@pytest.fixture
def sample_netcdf(tmp_path):
    """Create a sample NetCDF file with -180 to 180 longitude range."""
    nc_path = tmp_path / "test.nc"
    with nc.Dataset(nc_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)  # Unlimited dimension
        ds.createDimension("lat", 10)
        ds.createDimension("lon", 20)
        time = ds.createVariable("time", "f4", ("time",))
        lat = ds.createVariable("lat", "f4", ("lat",))
        lon = ds.createVariable("lon", "f4", ("lon",))
        tas = ds.createVariable("tas", "f4", ("time", "lat", "lon"), fill_value=-9999)
        lat.standard_name = "latitude"
        lon.standard_name = "longitude"
        time.units = "days since 2000-01-01"
        lat[:] = np.linspace(-45, 45, 10)
        lon[:] = np.linspace(-180, 180, 20)
        time[:] = [0]
        tas[0, :, :] = np.random.rand(10, 20)
    return nc_path

@pytest.fixture
def sample_netcdf_0_360(tmp_path):
    """Create a sample NetCDF file with 0 to 360 longitude range."""
    nc_path = tmp_path / "test_0_360.nc"
    with nc.Dataset(nc_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", 10)
        ds.createDimension("lon", 20)
        time = ds.createVariable("time", "f4", ("time",))
        lat = ds.createVariable("lat", "f4", ("lat",))
        lon = ds.createVariable("lon", "f4", ("lon",))
        tas = ds.createVariable("tas", "f4", ("time", "lat", "lon"), fill_value=-9999)
        lat.standard_name = "latitude"
        lon.standard_name = "longitude"
        time.units = "days since 2000-01-01"
        lat[:] = np.linspace(-45, 45, 10)
        lon[:] = np.linspace(0, 360, 20)
        time[:] = [0]
        tas[0, :, :] = np.random.rand(10, 20)
    return nc_path

@pytest.fixture
def sample_netcdf_no_fill(tmp_path):
    """Create a sample NetCDF file without _FillValue."""
    nc_path = tmp_path / "test_no_fill.nc"
    with nc.Dataset(nc_path, "w", format="NETCDF4") as ds:
        ds.createDimension("lat", 10)
        ds.createDimension("lon", 20)
        lat = ds.createVariable("lat", "f4", ("lat",))
        lon = ds.createVariable("lon", "f4", ("lon",))
        tas = ds.createVariable("tas", "f4", ("lat", "lon"))
        lat.standard_name = "latitude"
        lon.standard_name = "longitude"
        lat[:] = np.linspace(-45, 45, 10)
        lon[:] = np.linspace(-180, 180, 20)
        tas[:, :] = np.random.rand(10, 20)
    return nc_path

@pytest.fixture
def sample_netcdf_invalid(tmp_path):
    """Create a sample NetCDF file with invalid format (missing variables)."""
    nc_path = tmp_path / "test_invalid.nc"
    with nc.Dataset(nc_path, "w", format="NETCDF4") as ds:
        ds.createDimension("dim", 10)
        var = ds.createVariable("data", "f4", ("dim",))
        var[:] = np.random.rand(10)
    return nc_path

# Test find_coordinate_vars
def test_find_coordinate_vars_standard_names(sample_netcdf, caplog):
    """Test finding lat/lon vars using standard_name."""
    caplog.set_level(logging.DEBUG)
    with nc.Dataset(sample_netcdf, "r") as ds:
        lat_var, lon_var = find_coordinate_vars(ds)
        assert lat_var == "lat"
        assert lon_var == "lon"
        assert "Found lat_var=lat, lon_var=lon" in caplog.text

def test_find_coordinate_vars_alt_names(tmp_path, caplog):
    """Test finding lat/lon vars using alternative names."""
    caplog.set_level(logging.DEBUG)
    nc_path = tmp_path / "test_alt.nc"
    with nc.Dataset(nc_path, "w") as ds:
        ds.createDimension("y", 10)
        ds.createDimension("x", 20)
        lat = ds.createVariable("y", "f4", ("y",))
        lon = ds.createVariable("x", "f4", ("x",))
        lat[:] = np.linspace(-45, 45, 10)
        lon[:] = np.linspace(-180, 180, 20)
    with nc.Dataset(nc_path, "r") as ds:
        lat_var, lon_var = find_coordinate_vars(ds)
        assert lat_var == "y"
        assert lon_var == "x"
        assert "Found lat_var=y, lon_var=x" in caplog.text

def test_find_coordinate_vars_2d_coords(tmp_path, caplog):
    """Test failure when lat/lon vars are 2D."""
    caplog.set_level(logging.ERROR)
    nc_path = tmp_path / "test_2d.nc"
    with nc.Dataset(nc_path, "w") as ds:
        ds.createDimension("dim1", 10)
        ds.createDimension("dim2", 20)
        lat = ds.createVariable("lat", "f4", ("dim1", "dim2"))
        lon = ds.createVariable("lon", "f4", ("dim1", "dim2"))
        lat.standard_name = "latitude"
        lon.standard_name = "longitude"
        lat[:, :] = np.random.rand(10, 20)
        lon[:, :] = np.random.rand(10, 20)
    with nc.Dataset(nc_path, "r") as ds:
        lat_var, lon_var = find_coordinate_vars(ds)
        assert lat_var is None
        assert lon_var is None
        assert "Latitude or longitude is not 1D in dataset" in caplog.text

def test_find_coordinate_vars_no_vars(tmp_path, caplog):
    """Test failure when no lat/lon vars are found."""
    caplog.set_level(logging.ERROR)
    nc_path = tmp_path / "test_no_vars.nc"
    with nc.Dataset(nc_path, "w") as ds:
        ds.createDimension("dim", 10)
        var = ds.createVariable("data", "f4", ("dim",))
        var[:] = np.random.rand(10)
    with nc.Dataset(nc_path, "r") as ds:
        lat_var, lon_var = find_coordinate_vars(ds)
        assert lat_var is None
        assert lon_var is None
        assert "No latitude or longitude variables found in dataset" in caplog.text

def test_find_coordinate_vars_empty_dataset(tmp_path, caplog):
    """Test failure when dataset has no variables."""
    caplog.set_level(logging.ERROR)
    nc_path = tmp_path / "test_empty.nc"
    with nc.Dataset(nc_path, "w") as ds:
        ds.createDimension("dim", 10)
    with nc.Dataset(nc_path, "r") as ds:
        lat_var, lon_var = find_coordinate_vars(ds)
        assert lat_var is None
        assert lon_var is None
        assert "No latitude or longitude variables found in dataset" in caplog.text

# Test get_crop_indices
def test_get_crop_indices_basic():
    """Test basic cropping indices."""
    data = np.array([0, 10, 20, 30, 40])
    start, end = get_crop_indices(data, 10, 30)
    assert start == 1
    assert end == 3

def test_get_crop_indices_antimeridian():
    """Test antimeridian handling for longitude."""
    data = np.array([350, 0, 10, 20, 30])
    start, end = get_crop_indices(data, 350, 10, is_longitude=True)
    assert start == 0
    assert end == 2

def test_get_crop_indices_no_data():
    """Test when no data is within bounds."""
    data = np.array([0, 10, 20])
    start, end = get_crop_indices(data, 50, 60)
    assert start is None
    assert end is None

def test_get_crop_indices_single_point():
    """Test when bounds cover a single grid point."""
    data = np.array([0, 10, 20])
    start, end = get_crop_indices(data, 10, 10)
    assert start == 1
    assert end == 1

def test_get_crop_indices_empty_array():
    """Test with empty coordinate array."""
    data = np.array([])
    start, end = get_crop_indices(data, 0, 10)
    assert start is None
    assert end is None

# Test normalize_lon
def test_normalize_lon_0_360():
    """Test longitude normalization for 0-360 range."""
    assert normalize_lon(-10, 0, 360) == 350
    assert normalize_lon(370, 0, 360) == 10
    assert normalize_lon(0, 0, 360) == 0
    assert normalize_lon(360, 0, 360) == 360

def test_normalize_lon_180():
    """Test longitude normalization for -180-180 range."""
    assert normalize_lon(190, -180, 180) == -170
    assert normalize_lon(-190, -180, 180) == 170
    assert normalize_lon(180, -180, 180) == 180
    assert normalize_lon(-180, -180, 180) == -180

def test_normalize_lon_no_change():
    """Test longitude normalization when no change is needed."""
    assert normalize_lon(100, 0, 360) == 100
    assert normalize_lon(-100, -180, 180) == -100
    assert normalize_lon(0, -180, 180) == 0

# Test crop_netcdf_file
def test_crop_netcdf_file_success(sample_netcdf, tmp_path, caplog):
    """Test successful cropping of a NetCDF file."""
    caplog.set_level(logging.INFO)
    output_path = tmp_path / "output.nc"
    result = crop_netcdf_file(sample_netcdf, output_path, min_lat=-10, max_lat=10, min_lon=-20, max_lon=20, buffer_km=0)
    assert result is True
    assert output_path.exists()
    with nc.Dataset(output_path, "r") as ds:
        assert "lat" in ds.variables
        assert "lon" in ds.variables
        assert ds.variables["lat"].shape[0] == 2  # -9 to 9
        assert ds.variables["lon"].shape[0] == 2  # -9 to 9
        assert ds.variables["tas"].getncattr('_FillValue') == -9999
    assert f"Cropped file created: {output_path}" in caplog.text

def test_crop_netcdf_file_0_360(sample_netcdf_0_360, tmp_path, caplog):
    """Test cropping with 0-360 longitude range and antimeridian."""
    caplog.set_level(logging.INFO)
    output_path = tmp_path / "output.nc"
    result = crop_netcdf_file(
        sample_netcdf_0_360, output_path,
        min_lat=-10, max_lat=10,
        min_lon=340, max_lon=20,
        buffer_km=0
    )
    assert result is True
    assert output_path.exists()
    with nc.Dataset(output_path, "r") as ds:
        lon_vals = ds.variables["lon"][:]
        # Allow wraparound: either >=340 or <=20, allowing numerical fuzz
        assert all((lon >= 340 or lon <= 20 or np.isclose(lon % 360, [340, 0, 10, 20], atol=1).any())
                   for lon in lon_vals)


def test_crop_netcdf_file_buffer(sample_netcdf, tmp_path, caplog):
    """Test cropping with a buffer that results in no data."""
    caplog.set_level(logging.DEBUG)
    output_path = tmp_path / "output.nc"
    result = crop_netcdf_file(sample_netcdf, output_path, min_lat=0, max_lat=0, min_lon=0, max_lon=0, buffer_km=111)
    assert result is False
    assert not output_path.exists()
    assert "No data within lat/lon bounds for test.nc" in caplog.text

def test_crop_netcdf_file_no_fill(sample_netcdf_no_fill, tmp_path):
    """Test cropping a file without _FillValue."""
    output_path = tmp_path / "output.nc"
    result = crop_netcdf_file(sample_netcdf_no_fill, output_path, min_lat=-10, max_lat=10, min_lon=-20, max_lon=20, buffer_km=0)
    assert result is True
    assert output_path.exists()
    with nc.Dataset(output_path, "r") as ds:
        assert "tas" in ds.variables
        assert "_FillValue" not in ds.variables["tas"].ncattrs()

def test_crop_netcdf_file_invalid_bounds(sample_netcdf, tmp_path, caplog):
    """Test cropping with invalid latitude/longitude bounds."""
    caplog.set_level(logging.ERROR)
    output_path = tmp_path / "output.nc"

    # Invalid latitude
    result = crop_netcdf_file(
        sample_netcdf, output_path,
        min_lat=-100, max_lat=100,
        min_lon=-20, max_lon=20, buffer_km=0
    )
    assert result is False
    assert "Latitude bounds out of range" in caplog.text

    # Invalid longitude
    result = crop_netcdf_file(
        sample_netcdf, output_path,
        min_lat=-10, max_lat=10,
        min_lon=-200, max_lon=200, buffer_km=0
    )
    assert result is False
    assert "Invalid longitude bounds for -180-180" in caplog.text


def test_crop_netcdf_file_no_data(sample_netcdf, tmp_path, caplog):
    """Test cropping when no data is within bounds."""
    caplog.set_level(logging.ERROR)
    output_path = tmp_path / "output.nc"
    result = crop_netcdf_file(sample_netcdf, output_path, min_lat=50, max_lat=60, min_lon=0, max_lon=10, buffer_km=0)
    assert result is False
    assert not output_path.exists()
    assert "No data within lat/lon bounds for test.nc" in caplog.text

def test_crop_netcdf_file_2d_coords(tmp_path, caplog):
    """Test cropping failure with 2D coordinates."""
    caplog.set_level(logging.ERROR)
    nc_path = tmp_path / "test_2d.nc"
    with nc.Dataset(nc_path, "w") as ds:
        ds.createDimension("dim1", 10)
        ds.createDimension("dim2", 20)
        lat = ds.createVariable("lat", "f4", ("dim1", "dim2"))
        lon = ds.createVariable("lon", "f4", ("dim1", "dim2"))
        lat.standard_name = "latitude"
        lon.standard_name = "longitude"
        lat[:, :] = np.random.rand(10, 20)
        lon[:, :] = np.random.rand(10, 20)
    output_path = tmp_path / "output.nc"
    result = crop_netcdf_file(nc_path, output_path, min_lat=-10, max_lat=10, min_lon=-20, max_lon=20, buffer_km=0)
    assert result is False
    assert "Latitude or longitude is not 1D in dataset" in caplog.text

def test_crop_netcdf_file_invalid_file(tmp_path, caplog):
    """Test cropping with an invalid NetCDF file."""
    caplog.set_level(logging.ERROR)
    nc_path = tmp_path / "test_invalid.nc"
    with open(nc_path, "w") as f:
        f.write("not a netcdf file")
    output_path = tmp_path / "output.nc"
    result = crop_netcdf_file(nc_path, output_path, min_lat=-10, max_lat=10, min_lon=-20, max_lon=20, buffer_km=0)
    assert result is False
    assert "Failed to crop test_invalid.nc" in caplog.text

def test_crop_netcdf_file_missing_dimensions(tmp_path, caplog):
    """Test cropping with missing lat/lon data."""
    caplog.set_level(logging.ERROR)
    nc_path = tmp_path / "test_missing_dims.nc"
    with nc.Dataset(nc_path, "w") as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", 10)
        ds.createDimension("lon", 20)
        time = ds.createVariable("time", "f4", ("time",))
        lat = ds.createVariable("lat", "f4", ("lat",))
        lon = ds.createVariable("lon", "f4", ("lon",))
        lat.standard_name = "latitude"
        lon.standard_name = "longitude"
        time[:] = [0]
        # Intentionally leave lat/lon empty to trigger error
    output_path = tmp_path / "output.nc"
    result = crop_netcdf_file(nc_path, output_path, min_lat=-10, max_lat=10, min_lon=-20, max_lon=20, buffer_km=0)
    assert result is False
    assert "No data within lat/lon bounds for test_missing_dims.nc" in caplog.text

# Test crop_netcdf (main function)
def test_crop_netcdf_success(sample_netcdf, tmp_path, caplog):
    """Test successful cropping of a directory of NetCDF files."""
    caplog.set_level(logging.INFO)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    shutil.copy(sample_netcdf, input_dir / "test1.nc")
    shutil.copy(sample_netcdf, input_dir / "test2.nc")
    result = crop_netcdf(str(input_dir), str(output_dir), min_lat=-10, max_lat=10, min_lon=-20, max_lon=20, buffer_km=0, workers=1)
    assert result is True
    assert (output_dir / "test1_cropped.nc").exists()
    assert (output_dir / "test2_cropped.nc").exists()
    assert "Final Progress: 2/2 files (Successful: 2)" in caplog.text

def test_crop_netcdf_demo_mode(sample_netcdf, tmp_path, caplog):
    """Test crop_netcdf in demo mode."""
    caplog.set_level(logging.INFO)

    # Setup input directory as expected by demo mode
    input_dir = Path("cmip6_data")
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir()

    # Place sample file
    shutil.copy(sample_netcdf, input_dir / "test.nc")

    # Ensure cmip6_cropped_data output directory is clean
    output_dir = Path("cmip6_cropped_data")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    result = crop_netcdf(
        str(input_dir), str(output_dir),
        min_lat=0, max_lat=0, min_lon=0, max_lon=0,
        demo=True, workers=1
    )

    assert result is True
    assert (output_dir / "test_cropped.nc").exists()
    assert "Demo mode: Using bounds min_lat=35.0, max_lat=45.0, min_lon=-105.0, max_lon=-95.0, buffer_km=50.0" in caplog.text

    # Cleanup created folders
    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)


def test_crop_netcdf_no_files(tmp_path, caplog):
    """Test crop_netcdf when no NetCDF files are found."""
    caplog.set_level(logging.CRITICAL)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    result = crop_netcdf(str(input_dir), str(output_dir), min_lat=-10, max_lat=10, min_lon=-20, max_lon=20)
    assert result is False
    assert "No NetCDF files found in" in caplog.text

def test_crop_netcdf_invalid_bounds(tmp_path, caplog):
    """Test crop_netcdf with invalid bounds."""
    caplog.set_level(logging.ERROR)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    result = crop_netcdf(str(input_dir), str(output_dir), min_lat=10, max_lat=-10, min_lon=-20, max_lon=20)
    assert result is False
    assert "Invalid bounds: min_lat=10, max_lat=-10" in caplog.text

def test_crop_netcdf_negative_buffer(tmp_path, caplog):
    """Test crop_netcdf with negative buffer."""
    caplog.set_level(logging.ERROR)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    result = crop_netcdf(str(input_dir), str(output_dir), min_lat=-10, max_lat=10, min_lon=-20, max_lon=20, buffer_km=-10)
    assert result is False
    assert "Buffer cannot be negative" in caplog.text

def test_crop_netcdf_mixed_success(sample_netcdf, tmp_path, caplog):
    """Test crop_netcdf with some files succeeding and others failing."""
    caplog.set_level(logging.INFO)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "test1.nc").symlink_to(sample_netcdf)
    invalid_file = input_dir / "test2.nc"
    with nc.Dataset(invalid_file, "w") as ds:
        ds.createDimension("dim", 10)
        var = ds.createVariable("data", "f4", ("dim",))
        var[:] = np.random.rand(10)
    with patch("gridflow.crop_netcdf.crop_netcdf_file") as mock_crop:
        mock_crop.side_effect = [True, False]  # First file succeeds, second fails
        result = crop_netcdf(str(input_dir), str(output_dir), min_lat=-10, max_lat=10, min_lon=-20, max_lon=20, workers=2)
        assert result is True
        assert "Final Progress: 2/2 files (Successful: 1)" in caplog.text

def test_crop_netcdf_dir_creation_failure(tmp_path, caplog):
    """Test crop_netcdf when output directory creation fails."""
    caplog.set_level(logging.ERROR)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    with patch("pathlib.Path.mkdir", side_effect=Exception("Permission denied")):
        result = crop_netcdf(str(input_dir), str(output_dir), min_lat=-10, max_lat=10, min_lon=-20, max_lon=20)
        assert result is False
        assert "Failed to crop directory" in caplog.text

def test_crop_netcdf_workers_default(sample_netcdf, tmp_path):
    """Test crop_netcdf with default workers (None)."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "test1.nc").symlink_to(sample_netcdf)
    with patch("os.cpu_count", return_value=4):
        result = crop_netcdf(str(input_dir), str(output_dir), min_lat=-10, max_lat=10, min_lon=-20, max_lon=20, workers=None)
        assert result is True

def test_crop_netcdf_single_worker(sample_netcdf, tmp_path, caplog):
    """Test crop_netcdf with a single worker."""
    caplog.set_level(logging.INFO)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "test1.nc").symlink_to(sample_netcdf)
    result = crop_netcdf(str(input_dir), str(output_dir), min_lat=-10, max_lat=10, min_lon=-20, max_lon=20, workers=1)
    assert result is True
    assert (output_dir / "test1_cropped.nc").exists()
    assert "Completed: 1/1 files" in caplog.text

def test_crop_netcdf_no_cpu_count(sample_netcdf, tmp_path):
    """Test crop_netcdf when os.cpu_count returns None."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "test1.nc").symlink_to(sample_netcdf)
    with patch("os.cpu_count", return_value=None):
        result = crop_netcdf(str(input_dir), str(output_dir), min_lat=-10, max_lat=10, min_lon=-20, max_lon=20, workers=None)
        assert result is True
