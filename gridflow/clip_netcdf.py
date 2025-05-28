import time
import logging
import sys
import threading
import numpy as np
import netCDF4 as nc
import geopandas as gpd
from pathlib import Path
from threading import Lock
from .thread_stopper import ThreadManager
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from shapely.prepared import prep
from typing import Union, Optional
import shapely  # Added missing import

try:
    from shapely import contains_xy
    contains_func = contains_xy
except ImportError:
    from shapely.vectorized import contains
    contains_func = contains

logging_lock = Lock()

def reproject_bounds(gdf: gpd.GeoDataFrame, target_crs: str = 'EPSG:4326') -> tuple[float, float, float, float, gpd.GeoDataFrame]:
    """Reproject shapefile to target CRS and return bounds and reprojected GeoDataFrame."""
    if gdf.empty:
        with logging_lock:
            logging.error("GeoDataFrame is empty")
        raise ValueError("GeoDataFrame is empty")
    
    original_crs = gdf.crs
    gdf_reproj = gdf.to_crs(target_crs)
    bounds = gdf_reproj.total_bounds
    min_lon, min_lat, max_lon, max_lat = bounds
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    
    with logging_lock:
        logging.debug(f"[REPROJECT] Original CRS: {original_crs}, Target CRS: {target_crs}")
        logging.debug(f"[REPROJECT] Original bounds: min_lon={gdf.total_bounds[0]:.4f}, min_lat={gdf.total_bounds[1]:.4f}, "
                      f"max_lon={gdf.total_bounds[2]:.4f}, max_lat={gdf.total_bounds[3]:.4f}")
        logging.debug(f"[REPROJECT] Reprojected bounds: min_lon={min_lon:.4f}, min_lat={min_lat:.4f}, "
                      f"max_lon={max_lon:.4f}, max_lat={max_lat:.4f}")
        if lon_span > 180 or lat_span > 90:
            logging.warning(f"[REPROJECT] Shapefile bounds are large: lon_span={lon_span:.2f}, "
                            f"lat_span={lat_span:.2f}. Verify shapefile region.")
    
    return min_lon, min_lat, max_lon, max_lat, gdf_reproj

def add_buffer(gdf: gpd.GeoDataFrame, buffer_km: float = 0) -> gpd.GeoDataFrame:
    """
    Return *gdf* with all geometries buffered outward by *buffer_km* (kilometres).
    Uses an equal-area CRS for uniform buffering.
    """
    if buffer_km <= 0:
        return gdf
    gdf_m = gdf.to_crs("EPSG:6933")  # Metres everywhere
    gdf_m["geometry"] = gdf_m.buffer(buffer_km * 1_000)
    return gdf_m.to_crs("EPSG:4326")

def clip_single_file(
    input_file: Path,
    prep_geom,
    output_file: Path,
    shutdown_event: Optional[threading.Event] = None
) -> bool:
    """
    Mask every 2-D (lat, lon) field in *input_file* with *prep_geom* and write
    to *output_file*. Returns True on success, False on failure.
    """
    import time

    try:
        if not input_file.is_file():
            with logging_lock:
                logging.error(f"Input file does not exist: {input_file}")
            return False

        if shutdown_event and shutdown_event.is_set():
            with logging_lock:
                logging.info(f"Clipping stopped before {input_file.name}")
            return False

        # Retry opening the NetCDF file up to 3 times
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                with nc.Dataset(str(input_file), "r") as src:
                    if "lat" not in src.variables or "lon" not in src.variables:
                        with logging_lock:
                            logging.error(f"Missing 'lat' or 'lon' variable in {input_file.name}")
                        return False

                    # Extract the underlying geometry from PreparedGeometry
                    geom = prep_geom.context if hasattr(prep_geom, 'context') else prep_geom
                    if not isinstance(geom, (shapely.geometry.base.BaseGeometry, shapely.geometry.base.BaseMultipartGeometry)):
                        with logging_lock:
                            logging.error(f"Invalid geometry type for {input_file.name}: {type(geom)}")
                        return False

                    lat = src.variables["lat"][:]  # 1-D
                    lon = src.variables["lon"][:]
                    # Normalize longitude to [-180, 180]
                    lon_normalized = np.where(lon > 180, lon - 360, lon)
                    lon2d, lat2d = np.meshgrid(lon_normalized, lat)
                    # Create mask, default to False if geometry check fails
                    try:
                        mask2d = contains_func(geom, lon2d.flatten(), lat2d.flatten()).reshape(lon2d.shape)
                        if not np.any(mask2d):
                            with logging_lock:
                                logging.warning(f"No grid points intersect with geometry in {input_file.name}. Output will be fully masked.")
                    except Exception as e:
                        with logging_lock:
                            logging.warning(f"Geometry contains check failed for {input_file.name}: {e}. Creating fully masked output.")
                        mask2d = np.zeros(lon2d.shape, dtype=bool)

                    with nc.Dataset(str(output_file), "w", format=src.file_format) as dst:
                        # Copy dimensions
                        for dname, dim in src.dimensions.items():
                            dst.createDimension(dname, len(dim) if not dim.isunlimited() else None)

                        # Copy variables
                        for vname, varin in src.variables.items():
                            fill_kw = {}
                            if "_FillValue" in varin.ncattrs():
                                fill_kw["fill_value"] = varin.getncattr("_FillValue")

                            out = dst.createVariable(
                                vname, varin.datatype, varin.dimensions,
                                zlib=True, complevel=5, **fill_kw
                            )
                            out.setncatts({k: varin.getncattr(k)
                                          for k in varin.ncattrs() if k != "_FillValue"})

                            # Check shutdown during variable processing
                            if shutdown_event and shutdown_event.is_set():
                                with logging_lock:
                                    logging.info(f"Clipping interrupted for {input_file.name}")
                                return False

                            data = varin[:]
                            if ("lat" in varin.dimensions) and ("lon" in varin.dimensions):
                                fill_val = fill_kw.get("fill_value", np.nan)
                                if len(data.shape) > 2:
                                    mask_expanded = np.repeat(mask2d[np.newaxis, :, :], data.shape[0], axis=0)
                                else:
                                    mask_expanded = mask2d
                                data = np.where(mask_expanded, data, fill_val)
                            out[:] = data

                        # Copy global attributes
                        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

                # Ensure file handles are closed
                try:
                    src.close()
                    dst.close()
                except:
                    pass

                with logging_lock:
                    logging.info(f"Clipped file created: {output_file}")
                return True

            except Exception as e:
                if ("NetCDF: Not a valid ID" in str(e) or "NetCDF: HDF error" in str(e)) and attempt < max_attempts - 1:
                    with logging_lock:
                        logging.warning(f"Failed to open {input_file.name} on attempt {attempt + 1}: {e}. Retrying...")
                    time.sleep(0.2)
                    continue
                with logging_lock:
                    logging.error(f"Failed to clip {input_file.name}: {e}")
                return False

    except Exception as e:
        with logging_lock:
            logging.error(f"Failed to clip {input_file.name}: {e}")
        return False

def clip_netcdf(
    input_dir: str,
    shapefile_path: str,
    output_dir: str,
    shutdown_event: Optional[threading.Event] = None,
    workers: Optional[int] = None,
    buffer_km: float = 0,
    demo: bool = False
) -> bool:
    """
    Clip NetCDF files in *input_dir* using *shapefile_path* and save to *output_dir*.
    Returns True on success, False if stopped or failed.
    """
    thread_manager = ThreadManager(verbose=False)
    try:
        if shutdown_event and shutdown_event.is_set():
            with logging_lock:
                logging.info("Clipping operation stopped before starting")
            return False

        # Resolve default shapefile path
        default_shapefile = Path("./gridflow/iowa_border/iowa_border.shp")
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            base_path = Path(sys._MEIPASS)
            default_shapefile = base_path / "iowa_border" / "iowa_border.shp"

        # Use provided shapefile_path, fall back to default in demo mode
        shapefile_path = Path(shapefile_path or default_shapefile)
        if demo:
            shapefile_path = default_shapefile
            with logging_lock:
                logging.info(f"Demo mode: Using shapefile {shapefile_path}")

        # Verify shapefile exists
        if not shapefile_path.exists():
            if demo:
                with logging_lock:
                    logging.critical(f"No shapefile found at {shapefile_path}. Ensure the shapefile exists.")
                return False
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                alt_path = base_path / shapefile_path.name
                if alt_path.exists():
                    shapefile_path = alt_path
                else:
                    with logging_lock:
                        logging.error(f"Shapefile does not exist: {shapefile_path}")
                    raise FileNotFoundError(f"Shapefile does not exist: {shapefile_path}")
            else:
                with logging_lock:
                    logging.error(f"Shapefile does not exist: {shapefile_path}")
                raise FileNotFoundError(f"Shapefile does not exist: {shapefile_path}")

        # Resolve input and output directories
        input_dir = Path(input_dir or "./cmip6_data")
        output_dir = Path(output_dir or "./cmip6_clipped_data")
        output_dir.mkdir(parents=True, exist_ok=True)

        if demo:
            with logging_lock:
                logging.info(f"Demo mode: Using input_dir={input_dir}, output_dir={output_dir}, "
                             f"shapefile_path={shapefile_path}")

        # Read, buffer, and prepare geometry
        gdf_raw = gpd.read_file(shapefile_path)
        if gdf_raw.empty:
            with logging_lock:
                logging.error(f"Shapefile {shapefile_path} contains no geometries")
            raise ValueError(f"Shapefile {shapefile_path} contains no geometries")
        gdf_buf = add_buffer(gdf_raw, buffer_km=buffer_km)
        gdf_buf = gdf_buf.to_crs("EPSG:4326")
        union_geom = gdf_buf.unary_union
        if union_geom.is_empty:
            with logging_lock:
                logging.error(f"Union of geometries in {shapefile_path} is empty")
            raise ValueError(f"Union of geometries in {shapefile_path} is empty")
        prep_geom = prep(union_geom)

        # Gather NetCDF files
        nc_files = sorted(input_dir.glob("*.nc"))
        if not nc_files:
            with logging_lock:
                logging.critical(f"No NetCDF files found in {input_dir}. Run 'gridflow download --demo' to generate sample files.")
            return False

        # NEW: Log the number of files found
        total_files = len(nc_files)
        with logging_lock:
            logging.info(f"Found {total_files} NetCDF files to clip")  # NEW

        out_paths = [output_dir / f"{p.stem}_clipped{p.suffix}" for p in nc_files]

        # Threaded clipping
        workers = workers or (os.cpu_count() or 4)
        progress_int = max(1, len(nc_files) // 10)
        next_mark = progress_int
        completed = success = 0

        def worker_function(shutdown_event: threading.Event):
            nonlocal completed, success, next_mark  # MODIFIED: Added next_mark to nonlocals
            with ThreadPoolExecutor(max_workers=workers) as ex:
                future_to_nc = {
                    ex.submit(
                        clip_single_file,
                        nc_path,
                        prep_geom,
                        out_path,
                        shutdown_event
                    ): (nc_path, out_path)
                    for nc_path, out_path in zip(nc_files, out_paths)
                }

                for fut in as_completed(future_to_nc):
                    if shutdown_event.is_set():
                        with logging_lock:
                            logging.info("Clipping operation stopped")
                        return

                    nc_path, _ = future_to_nc[fut]
                    try:
                        if fut.result():
                            success += 1
                        else:
                            with logging_lock:
                                logging.error(f"Clipping failed for {nc_path.name}")
                            raise RuntimeError(f"Clipping failed for {nc_path.name}")
                    except Exception as e:
                        with logging_lock:
                            logging.error(f"Clipping failed for {nc_path.name}: {e}")
                        raise

                    completed += 1
                    # NEW: Log progress at intervals
                    with logging_lock:
                        if completed >= next_mark:
                            logging.info(f"Progress: {completed}/{total_files} files "
                                         f"(Successful: {success})")
                            next_mark += progress_int  # NEW

        # Add worker to ThreadManager
        thread_manager.add_worker(worker_function, "Clip_NetCDF_Worker")

        # Wait for threads to complete or stop
        while thread_manager.is_running() and not thread_manager.is_shutdown():
            time.sleep(0.1)

        # NEW: Log final progress
        with logging_lock:
            logging.info(f"Final Progress: {completed}/{total_files} files (Successful: {success})")  # NEW
            logging.info(f"Completed: {success}/{total_files} files")  # MODIFIED: Aligned with crop_netcdf

        if success == 0 and completed > 0:
            with logging_lock:
                logging.error("No files were clipped successfully")
            raise RuntimeError("No files were clipped successfully")
        return success > 0

    except Exception as e:
        with logging_lock:
            logging.error(f"Failed to clip directory {input_dir}: {e}")
        raise
    finally:
        thread_manager.stop()  # Ensure all threads are stopped


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#     try:
#         clip_netcdf(
#             input_dir="D:/GUI-Test/cmip6",
#             shapefile_path="D:/GUI-Test/conus.shp",
#             output_dir="D:/GUI-Test/cmip6_clipped",
#             buffer_km=100
#         )
#     except Exception as e:
#         logging.error(f"Script failed: {e}")