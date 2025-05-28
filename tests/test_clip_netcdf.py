import logging
import shutil
import sys
import numpy as np
import netCDF4 as nc
import geopandas as gpd
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from gridflow.clip_netcdf import reproject_bounds, add_buffer, clip_single_file, clip_netcdf

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@pytest.fixture
def sample_netcdf(tmp_path):
    """Create a sample NetCDF file with 0 to 360 longitude range."""
    nc_path = tmp_path / "test.nc"
    with nc.Dataset(nc_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", 10)
        ds.createDimension("lon", 20)
        time = ds.createVariable("time", "f4", ("time",))
        lat = ds.createVariable("lat", "f4", ("lat",))
        lon = ds.createVariable("lon", "f4", ("lon",))
        tas = ds.createVariable("tas", "f4", ("time", "lat", "lon"), fill_value=-9999)
        scalar = ds.createVariable("scalar", "f4", ("time",))
        lat.standard_name = "latitude"
        lon.standard_name = "longitude"
        time.units = "days since 2000-01-01"
        lat[:] = np.linspace(-45, 45, 10)
        lon[:] = np.linspace(0, 360, 20)
        time[:] = [0]
        tas[0, :, :] = np.random.rand(10, 20)
        scalar[0] = 42.0
    return nc_path

@pytest.fixture
def sample_shapefile(tmp_path):
    """Create a sample shapefile with a point at (10, 10)."""
    shp_path = tmp_path / "shape.shp"
    gdf = gpd.GeoDataFrame({"geometry": [Point(10, 10)]}, crs="EPSG:4326")
    gdf.to_file(shp_path)
    return shp_path

@pytest.fixture
def sample_shapefile_multi(tmp_path):
    shp_path = tmp_path / "multi_shape.shp"
    poly = Polygon([(-20, -20), (20, -20), (20, 20), (-20, 20)])
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")
    gdf.to_file(shp_path)
    return shp_path

@pytest.fixture
def sample_shapefile_empty(tmp_path):
    """Create an empty shapefile."""
    shp_path = tmp_path / "empty_shape.shp"
    gdf = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    gdf.to_file(shp_path)
    return shp_path

# Test reproject_bounds
def test_reproject_bounds_basic(sample_shapefile, caplog):
    """Test reprojection of shapefile bounds."""
    caplog.set_level(logging.DEBUG)
    gdf = gpd.read_file(sample_shapefile)
    min_lon, min_lat, max_lon, max_lat, gdf_reproj = reproject_bounds(gdf, target_crs="EPSG:4326")
    assert min_lon == pytest.approx(10)
    assert min_lat == pytest.approx(10)
    assert max_lon == pytest.approx(10)
    assert max_lat == pytest.approx(10)
    assert gdf_reproj.crs == "EPSG:4326"
    assert "Reprojected bounds: min_lon=10.0000, min_lat=10.0000" in caplog.text

def test_reproject_bounds_different_crs(sample_shapefile):
    """Test reprojection from a different CRS."""
    gdf = gpd.read_file(sample_shapefile).to_crs("EPSG:3857")
    min_lon, min_lat, max_lon, max_lat, gdf_reproj = reproject_bounds(gdf, target_crs="EPSG:4326")
    assert min_lon == pytest.approx(10, abs=0.1)
    assert min_lat == pytest.approx(10, abs=0.1)
    assert max_lon == pytest.approx(10, abs=0.1)
    assert max_lat == pytest.approx(10, abs=0.1)

def test_reproject_bounds_large_bounds(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    # Create a large polygon spanning nearly the globe
    poly = Polygon([(-170, -80), (170, -80), (170, 80), (-170, 80)])
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")
    min_lon, min_lat, max_lon, max_lat, gdf_reproj = reproject_bounds(gdf, target_crs="EPSG:4326")
    assert min_lon == -170
    assert min_lat == -80
    assert max_lon == 170
    assert max_lat == 80
    assert "Shapefile bounds are large" in caplog.text

def test_reproject_bounds_empty_gdf(sample_shapefile_empty, caplog):
    """Test reprojection with empty GeoDataFrame."""
    caplog.set_level(logging.DEBUG)
    gdf = gpd.read_file(sample_shapefile_empty)
    with pytest.raises(ValueError, match="GeoDataFrame is empty"):
        reproject_bounds(gdf, target_crs="EPSG:4326")

# Test add_buffer
def test_add_buffer_zero(sample_shapefile):
    """Test add_buffer with zero buffer."""
    gdf = gpd.read_file(sample_shapefile)
    gdf_buffered = add_buffer(gdf, buffer_km=0)
    assert gdf_buffered.equals(gdf)

def test_add_buffer_positive(sample_shapefile_multi):
    """Test add_buffer with positive buffer."""
    gdf = gpd.read_file(sample_shapefile_multi)
    gdf_buffered = add_buffer(gdf, buffer_km=10)
    assert gdf_buffered.crs == "EPSG:4326"
    assert gdf_buffered.geometry.iloc[0].area > gdf.geometry.iloc[0].area

def test_add_buffer_negative(sample_shapefile_multi):
    """Test add_buffer with negative buffer."""
    gdf = gpd.read_file(sample_shapefile_multi)
    gdf_buffered = add_buffer(gdf, buffer_km=-5)
    assert gdf_buffered.equals(gdf)

# Test clip_single_file
def test_clip_single_file_success(sample_netcdf, sample_shapefile_multi, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    output_path = tmp_path / "output.nc"
    gdf = gpd.read_file(sample_shapefile_multi)
    gdf = gdf.to_crs("EPSG:4326")
    prep_geom = prep(gdf.unary_union)
    result = clip_single_file(sample_netcdf, prep_geom, output_path)
    assert result is True
    assert output_path.exists()
    with nc.Dataset(output_path, "r") as ds:
        assert "lat" in ds.variables
        assert "lon" in ds.variables
        assert ds.variables["tas"].shape == (1, 10, 20)
        assert ds.variables["tas"].getncattr('_FillValue') == -9999
        assert ds.variables["scalar"][0] == 42.0
        tas = ds.variables["tas"][0, :, :]
        assert np.any(~tas.mask)  # Some data should be unmasked

def test_clip_single_file_longitude_normalization(sample_netcdf, sample_shapefile_multi, tmp_path):
    output_path = tmp_path / "output.nc"
    gdf = gpd.read_file(sample_shapefile_multi)
    gdf = gdf.to_crs("EPSG:4326")
    prep_geom = prep(gdf.unary_union)
    result = clip_single_file(sample_netcdf, prep_geom, output_path)
    assert result is True
    assert output_path.exists()

def test_clip_single_file_no_fill(sample_netcdf, sample_shapefile_multi, tmp_path):
    nc_path = tmp_path / "test_no_fill.nc"
    with nc.Dataset(nc_path, "w") as ds:
        ds.createDimension("lat", 10)
        ds.createDimension("lon", 20)
        lat = ds.createVariable("lat", "f4", ("lat",))
        lon = ds.createVariable("lon", "f4", ("lon",))
        tas = ds.createVariable("tas", "f4", ("lat", "lon"))
        lat.standard_name = "latitude"
        lon.standard_name = "longitude"
        lat[:] = np.linspace(-45, 45, 10)
        lon[:] = np.linspace(0, 360, 20)
        tas[:, :] = np.random.rand(10, 20)
    output_path = tmp_path / "output.nc"
    gdf = gpd.read_file(sample_shapefile_multi)
    gdf = gdf.to_crs("EPSG:4326")
    prep_geom = prep(gdf.unary_union)
    result = clip_single_file(nc_path, prep_geom, output_path)
    assert result is True
    assert output_path.exists()

def test_clip_single_file_stop_flag(sample_netcdf, sample_shapefile_multi, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    output_path = tmp_path / "output.nc"
    gdf = gpd.read_file(sample_shapefile_multi)
    prep_geom = prep(gdf.unary_union)
    shutdown_event = MagicMock()
    shutdown_event.is_set.return_value = True
    result = clip_single_file(sample_netcdf, prep_geom, output_path, shutdown_event=shutdown_event)
    assert result is False
    assert not output_path.exists()
    assert "Clipping stopped before test.nc" in caplog.text

def test_clip_single_file_no_intersection(sample_netcdf, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    shp_path = tmp_path / "outside.shp"
    poly = Polygon([(180, 85), (180, 86), (181, 86), (181, 85)])
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")
    gdf.to_file(shp_path)
    prep_geom = prep(gdf.unary_union)
    output_path = tmp_path / "output.nc"
    result = clip_single_file(sample_netcdf, prep_geom, output_path)
    assert result is True
    assert output_path.exists()

def test_clip_single_file_missing_lat_lon(tmp_path, sample_shapefile_multi, caplog):
    caplog.set_level(logging.ERROR)
    nc_path = tmp_path / "test_missing.nc"
    with nc.Dataset(nc_path, "w") as ds:
        ds.createDimension("time", 1)
        ds.createVariable("data", "f4", ("time",))
    output_path = tmp_path / "output.nc"
    gdf = gpd.read_file(sample_shapefile_multi)
    prep_geom = prep(gdf.unary_union)
    result = clip_single_file(nc_path, prep_geom, output_path)
    assert result is False
    assert "Missing 'lat' or 'lon' variable in test_missing.nc" in caplog.text

def test_clip_single_file_invalid_file(tmp_path, sample_shapefile_multi, caplog):
    """Test clipping with invalid NetCDF file."""
    caplog.set_level(logging.ERROR)
    nc_path = tmp_path / "invalid.nc"
    with open(nc_path, "w") as f:
        f.write("not a netcdf file")
    output_path = tmp_path / "output.nc"
    gdf = gpd.read_file(sample_shapefile_multi)
    prep_geom = prep(gdf.unary_union)
    result = clip_single_file(nc_path, prep_geom, output_path)
    assert result is False
    assert "Failed to clip invalid.nc" in caplog.text

# Test clip_netcdf
def test_clip_netcdf_success(sample_netcdf, sample_shapefile_multi, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    # Use shutil.copy2 to preserve metadata and ensure file is closed
    shutil.copy2(sample_netcdf, input_dir / "test1.nc")
    shutil.copy2(sample_netcdf, input_dir / "test2.nc")
    # Verify files are accessible before processing
    for nc_file in [input_dir / "test1.nc", input_dir / "test2.nc"]:
        with open(nc_file, "rb") as f:
            f.read(1)  # Ensure file is readable
    result = clip_netcdf(str(input_dir), str(sample_shapefile_multi), str(output_dir), workers=2, buffer_km=10)
    assert result is True
    assert (output_dir / "test1_clipped.nc").exists()
    assert (output_dir / "test2_clipped.nc").exists()

def test_clip_netcdf_demo_mode(sample_netcdf, tmp_path, caplog):
    """Test clip_netcdf in demo mode with valid shapefile."""
    caplog.set_level(logging.INFO)
    input_dir = tmp_path / "cmip6_data"
    output_dir = tmp_path / "cmip6_clipped_data"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "test.nc").symlink_to(sample_netcdf)
    # Mock a GeoDataFrame with a valid polygon
    mock_gdf = gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, crs="EPSG:4326")
    with patch("geopandas.read_file", return_value=mock_gdf):
        result = clip_netcdf(str(input_dir), "dummy.shp", str(output_dir), demo=True, workers=1)
    assert result is True
    assert (output_dir / "test_clipped.nc").exists()

def test_clip_netcdf_no_files(tmp_path, sample_shapefile_multi, caplog):
    """Test clip_netcdf when no NetCDF files are found."""
    caplog.set_level(logging.CRITICAL)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    result = clip_netcdf(str(input_dir), str(sample_shapefile_multi), str(output_dir))
    assert result is False
    assert "No NetCDF files found in" in caplog.text

def test_clip_netcdf_invalid_shapefile(tmp_path, caplog):
    """Test clip_netcdf with invalid shapefile."""
    caplog.set_level(logging.ERROR)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    shapefile = tmp_path / "nonexistent.shp"
    with pytest.raises(FileNotFoundError, match="Shapefile does not exist"):
        clip_netcdf(str(input_dir), str(shapefile), str(output_dir))
    assert "Shapefile does not exist" in caplog.text

def test_clip_netcdf_invalid_shapefile_demo(tmp_path, caplog):
    """Test clip_netcdf with invalid shapefile in demo mode."""
    caplog.set_level(logging.CRITICAL)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    with patch("pathlib.Path.exists", return_value=False):
        result = clip_netcdf(str(input_dir), "dummy.shp", str(output_dir), demo=True)
        assert result is False
        assert "No shapefile found at" in caplog.text

def test_clip_netcdf_stop_flag_start(sample_netcdf, sample_shapefile_multi, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "test.nc").symlink_to(sample_netcdf)
    shutdown_event = MagicMock()
    shutdown_event.is_set.return_value = True
    result = clip_netcdf(str(input_dir), str(sample_shapefile_multi), str(output_dir), shutdown_event=shutdown_event, workers=1)
    assert result is False
    assert "Clipping operation stopped before starting" in caplog.text

def test_clip_netcdf_stop_flag_mid(sample_netcdf, sample_shapefile_multi, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    shutil.copy(sample_netcdf, input_dir / "test1.nc")
    shutil.copy(sample_netcdf, input_dir / "test2.nc")

    def mock_clip_single_file(input_path, *args, **kwargs):
        if Path(input_path).name == "test1.nc":
            output_path = Path(str(output_dir)) / "test1_clipped.nc"
            with nc.Dataset(output_path, "w") as ds:
                ds.createDimension("lat", 1)
                ds.createDimension("lon", 1)
                ds.createVariable("lat", "f4", ("lat",))[:] = [0]
                ds.createVariable("lon", "f4", ("lon",))[:] = [0]
            return True
        return False

    with patch("gridflow.clip_netcdf.clip_single_file", side_effect=mock_clip_single_file):
        shutdown_event = MagicMock()
        shutdown_event.is_set.return_value = False
        result = clip_netcdf(str(input_dir), str(sample_shapefile_multi), str(output_dir), shutdown_event=shutdown_event, workers=1)

    assert result is True
    assert (output_dir / "test1_clipped.nc").exists()
    assert not (output_dir / "test2_clipped.nc").exists()


def test_clip_netcdf_mixed_success(sample_netcdf, sample_shapefile_multi, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    # Use shutil.copy2 to preserve metadata and ensure file is closed
    shutil.copy2(sample_netcdf, input_dir / "test1.nc")
    shutil.copy2(sample_netcdf, input_dir / "test2.nc")
    # Verify files are accessible before processing
    for nc_file in [input_dir / "test1.nc", input_dir / "test2.nc"]:
        with open(nc_file, "rb") as f:
            f.read(1)  # Ensure file is readable
    result = clip_netcdf(str(input_dir), str(sample_shapefile_multi), str(output_dir))
    assert result is True
    assert (output_dir / "test1_clipped.nc").exists()

def test_clip_netcdf_no_success(sample_netcdf, sample_shapefile_multi, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    shutil.copy(sample_netcdf, input_dir / "test.nc")

    with patch("gridflow.clip_netcdf.clip_single_file", return_value=False):
        result = clip_netcdf(str(input_dir), str(sample_shapefile_multi), str(output_dir))
    
    assert result is False
    assert not any(output_dir.glob("*.nc"))


def test_clip_netcdf_dir_creation_failure(tmp_path, sample_shapefile_multi, caplog):
    caplog.set_level(logging.ERROR)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    with patch("pathlib.Path.mkdir", side_effect=Exception("Permission denied")):
        with pytest.raises(Exception, match="Permission denied"):
            clip_netcdf(str(input_dir), str(sample_shapefile_multi), str(output_dir))

def test_clip_netcdf_default_workers(sample_netcdf, sample_shapefile_multi, tmp_path):
    """Test clip_netcdf with default workers (None)."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "test.nc").symlink_to(sample_netcdf)
    with patch("os.cpu_count", return_value=4):
        result = clip_netcdf(str(input_dir), str(sample_shapefile_multi), str(output_dir), workers=None)
    assert result is True
    assert (output_dir / "test_clipped.nc").exists()

def test_clip_netcdf_frozen_env_invalid(sample_netcdf, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    shutil.copy(sample_netcdf, input_dir / "test.nc")
    with patch.dict(sys.__dict__, {'frozen': True, '_MEIPASS': 'frozen_base'}), \
         patch("pathlib.Path.exists", side_effect=[False, False]):
        try:
            result = clip_netcdf(str(input_dir), "nonexistent.shp", str(output_dir))
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass
    assert "Shapefile does not exist: nonexistent.shp" in caplog.text

def test_clip_netcdf_frozen_env_valid(sample_netcdf, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    shutil.copy(sample_netcdf, input_dir / "test.nc")
    # Mock a GeoDataFrame with a valid polygon
    mock_gdf = gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, crs="EPSG:4326")
    with patch.dict(sys.__dict__, {'frozen': True, '_MEIPASS': 'frozen_base'}), \
         patch("pathlib.Path.exists", side_effect=[False, True]), \
         patch("geopandas.read_file", return_value=mock_gdf):
        result = clip_netcdf(str(input_dir), "iowa_border.shp", str(output_dir))
    assert result is True
    assert (output_dir / "test_clipped.nc").exists()