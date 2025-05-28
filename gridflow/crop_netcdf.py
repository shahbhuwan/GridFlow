# Copyright (c) 2025 Bhuwan Shah
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import time
import logging
import math
import threading
import netCDF4 as nc
import numpy as np
from pathlib import Path
from .thread_stopper import ThreadManager
from typing import Optional, Tuple
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import errno
import tempfile

logging_lock = Lock()

def find_coordinate_vars(dataset: nc.Dataset) -> Tuple[Optional[str], Optional[str]]:
    """Find latitude and longitude variables in the NetCDF dataset."""
    lat_var = None
    lon_var = None
    debug_info = ["Available variables and attributes:"]
    for var_name in dataset.variables:
        var = dataset.variables[var_name]
        attrs = {k: str(var.getncattr(k)) for k in var.ncattrs()} if var.ncattrs() else {}
        debug_info.append(f"  {var_name}: shape={var.shape}, attrs={attrs}")
        if hasattr(var, 'standard_name'):
            if var.standard_name == 'latitude':
                lat_var = var_name
            elif var.standard_name == 'longitude':
                lon_var = var_name
        elif var_name.lower() in ['lat', 'latitude', 'y', 'nav_lat']:
            lat_var = var_name
        elif var_name.lower() in ['lon', 'longitude', 'x', 'nav_lon']:
            lon_var = var_name
    if not lat_var or not lon_var:
        with logging_lock:
            logging.error(f"No latitude or longitude variables found in dataset\n" + "\n".join(debug_info))
        return None, None
    if len(dataset.variables[lat_var].shape) != 1 or len(dataset.variables[lon_var].shape) != 1:
        with logging_lock:
            logging.error(f"Latitude or longitude is not 1D in dataset")
        return None, None
    with logging_lock:
        logging.debug(f"Found lat_var={lat_var}, lon_var={lon_var}")
    return lat_var, lon_var

def get_crop_indices(coord_data: np.ndarray, min_val: float, max_val: float, is_longitude: bool = False) -> Tuple[Optional[int], Optional[int]]:
    """Find indices for cropping coordinate data within given bounds."""
    if len(coord_data) == 0:
        return None, None
    if is_longitude and min_val > max_val:
        indices1 = np.where(coord_data >= min_val)[0]
        indices2 = np.where(coord_data <= max_val)[0]
        indices = np.sort(np.union1d(indices1, indices2))
        if len(indices) == 0:
            return None, None
        if len(indices1) > 0 and len(indices2) > 0:
            return indices[0], indices[-1]
        return indices[0], indices[-1]
    else:
        indices = np.where((coord_data >= min_val) & (coord_data <= max_val))[0]
    if len(indices) == 0:
        return None, None
    return indices[0], indices[-1]

def normalize_lon(lon: float, lon_min: float, lon_max: float) -> float:
    """Wrap a longitude into [lon_min, lon_max], inclusive."""
    span = lon_max - lon_min
    if span == 0:
        return lon
    shifted = (lon - lon_min) % span
    normalized = lon_min + shifted
    if np.isclose(normalized, lon_min) and not np.isclose(lon, lon_min):
        normalized = lon_max
    return normalized

def crop_netcdf_file(
    input_path: Path,
    output_path: Path,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    buffer_km: float = 0.0,
    shutdown_event: Optional[threading.Event] = None,
    retries: int = 3,
    retry_delay: float = 1.0
) -> bool:
    """
    Crop a single NetCDF file by spatial bounds with retry logic for file writes.

    Args:
        input_path: Path to the input NetCDF file.
        output_path: Path to save the cropped NetCDF file.
        min_lat: Minimum latitude bound.
        max_lat: Maximum latitude bound.
        min_lon: Minimum longitude bound.
        max_lon: Maximum longitude bound.
        buffer_km: Buffer distance in kilometers to expand bounds.
        shutdown_event: Event to signal task interruption.
        retries: Number of retry attempts for file writes.
        retry_delay: Delay between retries in seconds.

    Returns:
        bool: True if cropping succeeds, False otherwise.
    """
    try:
        if shutdown_event and shutdown_event.is_set():
            with logging_lock:
                logging.info(f"Cropping stopped before {input_path.name}")
            return False

        # Ensure output directory exists
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with nc.Dataset(input_path, 'r') as src:
            lat_var, lon_var = find_coordinate_vars(src)
            if not lat_var or not lon_var:
                return False

            lat_data = src.variables[lat_var][:].filled(np.nan)
            lon_data = src.variables[lon_var][:].filled(np.nan)

            if min_lat < -90 or max_lat > 90:
                with logging_lock:
                    logging.error(f"Latitude bounds out of range: min_lat={min_lat}, max_lat={max_lat}")
                return False
            if min_lon < -360 or max_lon > 360:
                with logging_lock:
                    logging.error(f"Longitude bounds out of range: min_lon={min_lon}, max_lon={max_lon}")
                return False

            lon_min, lon_max = np.nanmin(lon_data), np.nanmax(lon_data)
            target_range = '0-360' if lon_min >= 0 else '-180-180'
            with logging_lock:
                logging.debug(f"NetCDF longitude range: {lon_min} to {lon_max}, using {target_range}")

            min_lon_n = normalize_lon(min_lon, lon_min, lon_max)
            max_lon_n = normalize_lon(max_lon, lon_min, lon_max)
            with logging_lock:
                logging.debug(f"Normalized input lon: min_lon={min_lon_n}, max_lon={max_lon_n}")

            if target_range == '-180-180' and min_lon_n > max_lon_n:
                with logging_lock:
                    logging.error(f"Invalid longitude bounds for -180-180: min_lon={min_lon_n}, max_lon={max_lon_n}")
                return False
            wrap_lon = (target_range == '0-360' and min_lon_n > max_lon_n)

            if buffer_km > 0:
                lat_buff = buffer_km / 111.0
                avg_lat = 0.5 * (min_lat + max_lat)
                lon_buff = buffer_km / (111.0 * math.cos(math.radians(avg_lat)))
                min_lat = max(-90, min_lat - lat_buff)
                max_lat = min(90, max_lat + lat_buff)
                min_lon_n = normalize_lon(min_lon_n - lon_buff, lon_min, lon_max)
                max_lon_n = normalize_lon(max_lon_n + lon_buff, lon_min, lon_max)
                with logging_lock:
                    logging.debug(f"Bounds with buffer: min_lat={min_lat}, max_lat={max_lat}, min_lon={min_lon_n}, max_lon={max_lon_n}")

            lat_idx = np.where((lat_data >= min_lat) & (lat_data <= max_lat))[0]
            if lat_idx.size == 0:
                with logging_lock:
                    logging.error(f"No data within lat/lon bounds for {input_path.name}")
                return False
            lat_start, lat_end = lat_idx[0], lat_idx[-1]
            lat_size = lat_end - lat_start + 1

            if wrap_lon:
                idx1 = np.where(lon_data >= min_lon_n)[0]
                idx2 = np.where(lon_data <= max_lon_n)[0]
                if idx1.size == 0 or idx2.size == 0:
                    with logging_lock:
                        logging.error(f"No data within lat/lon bounds for {input_path.name}")
                    return False
                lon_size = idx1.size + idx2.size
            else:
                lon_idx = np.where((lon_data >= min_lon_n) & (lon_data <= max_lon_n))[0]
                if lon_idx.size == 0:
                    with logging_lock:
                        logging.error(f"No data within lat/lon bounds for {input_path.name}")
                    return False
                lon_start, lon_end = lon_idx[0], lon_idx[-1]
                lon_size = lon_end - lon_start + 1

            lat_dim = src.variables[lat_var].dimensions[0]
            lon_dim = src.variables[lon_var].dimensions[0]

            for attempt in range(retries + 1):
                try:
                    with nc.Dataset(output_path, 'w', format=src.file_format) as dst:
                        dst.setncatts(src.__dict__)
                        for dim, d in src.dimensions.items():
                            size = None if d.isunlimited() else (
                                lat_size if dim == lat_dim else (lon_size if dim == lon_dim else d.size)
                            )
                            dst.createDimension(dim, size)

                        for name, var in src.variables.items():
                            if shutdown_event and shutdown_event.is_set():
                                with logging_lock:
                                    logging.info(f"Cropping interrupted for {input_path.name}")
                                return False

                            dims = var.dimensions
                            out_var = dst.createVariable(
                                name, var.dtype, dims,
                                zlib=True,
                                fill_value=(var.getncattr('_FillValue') if '_FillValue' in var.ncattrs() else None)
                            )
                            out_var.setncatts({k: v for k, v in var.__dict__.items() if k != '_FillValue'})

                            if name == lon_var:
                                if wrap_lon:
                                    new_lon = np.concatenate((lon_data[idx1], lon_data[idx2]))
                                else:
                                    new_lon = lon_data[lon_start:lon_end + 1]
                                out_var[:] = new_lon
                            elif name == lat_var:
                                out_var[:] = lat_data[lat_start:lat_end + 1]
                            else:
                                data = var[:]
                                if lat_dim in dims:
                                    lat_axis = dims.index(lat_dim)
                                    data = np.take(data, np.arange(lat_start, lat_end + 1), axis=lat_axis)
                                if lon_dim in dims:
                                    lon_axis = dims.index(lon_dim)
                                    if wrap_lon:
                                        data1 = np.take(data, idx1, axis=lon_axis)
                                        data2 = np.take(data, idx2, axis=lon_axis)
                                        data = np.concatenate((data1, data2), axis=lon_axis)
                                    else:
                                        data = np.take(data, np.arange(lon_start, lon_end + 1), axis=lon_axis)
                                out_var[:] = data

                    with logging_lock:
                        logging.info(f"Cropped file created: {output_path}")
                    return True

                except (OSError, PermissionError) as e:
                    if attempt < retries:
                        with logging_lock:
                            logging.warning(f"Attempt {attempt + 1} failed to write {output_path}: {e}. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        with logging_lock:
                            logging.error(f"Failed to crop {input_path.name} after {retries} attempts: {e}")
                        return False
                except Exception as e:
                    with logging_lock:
                        logging.error(f"Failed to crop {input_path.name}: {e}")
                    return False

    except Exception as e:
        with logging_lock:
            logging.error(f"Failed to crop {input_path.name}: {e}")
        return False

def crop_netcdf(
    input_dir: str,
    output_dir: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    buffer_km: float = 0.0,
    workers: int = None,
    demo: bool = False,
    shutdown_event: Optional[threading.Event] = None
) -> bool:
    """
    Crop all NetCDF files in a directory by spatial bounds in parallel.

    Args:
        input_dir: Path to directory containing input NetCDF files.
        output_dir: Path to directory to save cropped NetCDF files.
        min_lat: Minimum latitude bound.
        max_lat: Maximum latitude bound.
        min_lon: Minimum longitude bound.
        max_lon: Maximum longitude bound.
        buffer_km: Buffer distance in kilometers to expand bounds.
        workers: Number of parallel workers (defaults to number of CPU cores).
        demo: If True, use demo bounds (35N-45N, 95W-105W).
        shutdown_event: Event to signal task interruption.

    Returns:
        bool: True if any files were successfully processed, False otherwise.
    """
    thread_manager = ThreadManager(verbose=False)
    try:
        if shutdown_event and shutdown_event.is_set():
            with logging_lock:
                logging.info("Cropping operation stopped before starting")
            return False

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Ensure output directory exists and is writable
        for attempt in range(3):
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=output_dir, delete=True) as tmp:
                    tmp.write(b"test")
                    tmp.flush()
                break
            except (PermissionError, OSError) as e:
                if attempt < 2:
                    with logging_lock:
                        logging.warning(f"Attempt {attempt + 1} failed to create or write to {output_dir}: {e}. Retrying...")
                    time.sleep(1.0)
                else:
                    with logging_lock:
                        logging.error(f"Permission denied or failed to create output directory {output_dir}: {e}")
                    return False

        # Use demo bounds if specified
        if demo:
            input_dir = Path("./cmip6_data")
            output_dir = Path("./cmip6_cropped_data")
            min_lat, max_lat = 35.0, 45.0
            min_lon, max_lon = -105.0, -95.0
            buffer_km = 50.0
            with logging_lock:
                logging.info(f"Demo mode: Using bounds min_lat={min_lat}, max_lat={max_lat}, min_lon={min_lon}, max_lon={max_lon}, buffer_km={buffer_km}")

        # Validate bounds
        if min_lat >= max_lat or min_lon >= max_lon:
            with logging_lock:
                logging.error(f"Invalid bounds: min_lat={min_lat}, max_lat={max_lat}, min_lon={min_lon}, max_lon={max_lon}")
            return False
        if buffer_km < 0:
            with logging_lock:
                logging.error(f"Buffer cannot be negative: buffer_km={buffer_km}")
            return False

        # Find all NetCDF files
        nc_files = list(input_dir.glob("*.nc"))
        if not nc_files:
            with logging_lock:
                logging.critical(f"No NetCDF files found in {input_dir}. Run 'gridflow download --demo' to generate sample files.")
            return False

        total_files = len(nc_files)
        with logging_lock:
            logging.info(f"Found {total_files} NetCDF files to crop")

        # Prepare tasks
        tasks = [(nc_file, output_dir / f"{nc_file.stem}_cropped{nc_file.suffix}") for nc_file in nc_files]

        # Process files in parallel
        workers = workers or min(os.cpu_count() or 4, len(tasks))  # Limit workers to number of tasks
        completed = 0
        success_count = 0
        progress_interval = max(1, total_files // 10)
        next_threshold = progress_interval

        def worker_function(shutdown_event: threading.Event):
            nonlocal completed, success_count, next_threshold
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_task = {
                    executor.submit(crop_netcdf_file, in_file, out_file, min_lat, max_lat, min_lon, max_lon, buffer_km, shutdown_event, 3, 1.0): (in_file, out_file)
                    for in_file, out_file in tasks
                }
                for future in as_completed(future_to_task):
                    if shutdown_event.is_set():
                        with logging_lock:
                            logging.info("Cropping operation stopped")
                        return

                    in_file, out_file = future_to_task[future]
                    try:
                        result = future.result()
                        with logging_lock:
                            completed += 1
                            if result:
                                success_count += 1
                            if completed >= next_threshold:
                                logging.info(f"Progress: {completed}/{total_files} files (Successful: {success_count})")
                                next_threshold += progress_interval
                    except Exception as e:
                        with logging_lock:
                            logging.error(f"Error processing {in_file.name}: {e}")

        thread_manager.add_worker(worker_function, "Crop_NetCDF_Worker")

        while thread_manager.is_running() and not thread_manager.is_shutdown():
            time.sleep(0.1)

        with logging_lock:
            logging.info(f"Final Progress: {completed}/{total_files} files (Successful: {success_count})")
            logging.info(f"Completed: {success_count}/{total_files} files")
        return success_count > 0

    except Exception as e:
        with logging_lock:
            logging.error(f"Failed to crop directory {input_dir}: {e}")
        return False
    finally:
        thread_manager.stop()