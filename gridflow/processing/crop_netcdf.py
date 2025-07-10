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

import os
import sys
import signal
import logging
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Any

# --- Dependency Check ---
try:
    import numpy as np
    import netCDF4 as nc
except ImportError as e:
    print(f"FATAL: Missing required library: {e.name}. Please install it using 'pip install {e.name}'.")
    sys.exit(1)

# --- Local Imports ---
try:
    from gridflow.utils.logging_utils import setup_logging
except ImportError:
    print("FATAL: logging_utils.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

# --- Global Stop Event for Graceful Shutdown ---
stop_event = threading.Event()

def signal_handler(sig, frame):
    """Handles SIGINT (Ctrl+C) to gracefully shut down the application."""
    logging.warning("Stop signal received! Gracefully shutting down...")
    stop_event.set()
    logging.info("Please wait for ongoing tasks to complete.")

def find_coordinate_vars(dataset: nc.Dataset) -> Tuple[Optional[str], Optional[str]]:
    """Finds the names of latitude and longitude variables in a NetCDF dataset."""
    lat_var, lon_var = None, None
    for var_name, var in dataset.variables.items():
        if hasattr(var, 'standard_name'):
            if var.standard_name == 'latitude': lat_var = var_name
            elif var.standard_name == 'longitude': lon_var = var_name
        elif var_name.lower() in ['lat', 'latitude']: lat_var = var_name
        elif var_name.lower() in ['lon', 'longitude']: lon_var = var_name
    return lat_var, lon_var

def crop_single_file(
    input_path: Path,
    output_path: Path,
    bounds: Dict[str, float],
    stop_event: threading.Event
) -> bool:
    """Crops a single NetCDF file based on spatial bounds."""
    if stop_event.is_set(): return False
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with nc.Dataset(input_path, 'r') as src:
            lat_var_name, lon_var_name = find_coordinate_vars(src)
            if not lat_var_name or not lon_var_name:
                logging.error(f"Could not find coordinate variables in {input_path.name}.")
                return False

            lat_data = src.variables[lat_var_name][:]
            lon_data = src.variables[lon_var_name][:]

            # Handle longitude wrapping (e.g., 0-360 vs -180-180)
            lon_data_norm = np.where(lon_data > 180, lon_data - 360, lon_data)

            # Find indices within bounds
            lat_indices = np.where((lat_data >= bounds['min_lat']) & (lat_data <= bounds['max_lat']))[0]
            lon_indices = np.where((lon_data_norm >= bounds['min_lon']) & (lon_data_norm <= bounds['max_lon']))[0]

            if lat_indices.size == 0 or lon_indices.size == 0:
                logging.warning(f"No data within the specified bounds for {input_path.name}. Skipping.")
                return False
            
            lat_start, lat_end = lat_indices.min(), lat_indices.max()
            lon_start, lon_end = lon_indices.min(), lon_indices.max()

            with nc.Dataset(output_path, 'w', format=src.file_format) as dst:
                dst.setncatts(src.__dict__)
                
                lat_dim_name = src.variables[lat_var_name].dimensions[0]
                lon_dim_name = src.variables[lon_var_name].dimensions[0]
                dst.createDimension(lat_dim_name, lat_indices.size)
                dst.createDimension(lon_dim_name, lon_indices.size)
                for name, dim in src.dimensions.items():
                    if name not in [lat_dim_name, lon_dim_name]:
                        dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

                for name, var in src.variables.items():
                    if stop_event.is_set(): return False

                    # --- THIS IS THE FIX ---
                    # 1. Get the fill value, if it exists.
                    fill_value = var.getncattr('_FillValue') if '_FillValue' in var.ncattrs() else None
                    
                    # 2. Create the destination variable, passing the fill value at creation time.
                    dst_var = dst.createVariable(name, var.dtype, var.dimensions, fill_value=fill_value)
                    
                    # 3. Copy all other attributes, making sure to exclude the _FillValue.
                    attrs_to_copy = {k: var.getncattr(k) for k in var.ncattrs() if k != '_FillValue'}
                    dst_var.setncatts(attrs_to_copy)
                    # --- END OF FIX ---
                    
                    if lat_dim_name in var.dimensions and lon_dim_name in var.dimensions:
                        slices = [slice(None)] * var.ndim
                        lat_axis = var.dimensions.index(lat_dim_name)
                        lon_axis = var.dimensions.index(lon_dim_name)
                        slices[lat_axis] = slice(lat_start, lat_end + 1)
                        slices[lon_axis] = slice(lon_start, lon_end + 1)
                        dst_var[:] = var[tuple(slices)]
                    elif lat_dim_name in var.dimensions:
                        dst_var[:] = var[lat_start:lat_end + 1]
                    elif lon_dim_name in var.dimensions:
                        dst_var[:] = var[lon_start:lon_end + 1]
                    else:
                        dst_var[:] = var[:]
                        
        logging.info(f"Successfully cropped {input_path.name} to {output_path.name}")
        return True
    except Exception as e:
        logging.error(f"Failed to crop {input_path.name}: {e}", exc_info=False) # Set exc_info to False for cleaner logs
        if output_path.exists(): output_path.unlink(missing_ok=True)
        return False

class Cropper:
    """Manages the parallel cropping of NetCDF files."""
    def __init__(self, settings: Dict[str, Any], stop_event: threading.Event):
        self.settings = settings
        self._stop_event = stop_event
        self.executor = None

    def shutdown(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        logging.info("Cropper has been shut down.")

    def crop_all(self, files_to_crop: List[Tuple[Path, Path]]) -> Tuple[int, int]:
        """Manages the parallel cropping of a list of files."""
        successful_crops = 0
        if not files_to_crop: return 0, 0
        
        self.executor = ThreadPoolExecutor(max_workers=self.settings['workers'], thread_name_prefix='Cropper')
        future_to_file = {
            self.executor.submit(crop_single_file, in_path, out_path, self.settings['bounds'], self._stop_event): in_path
            for in_path, out_path in files_to_crop
        }
        
        total_files = len(files_to_crop)
        try:
            for i, future in enumerate(as_completed(future_to_file)):
                logging.info(f"Progress: {i + 1}/{total_files} files processed.")
                if self._stop_event.is_set(): break
                if future.result():
                    successful_crops += 1
        finally:
            self.shutdown()
        
        return successful_crops, total_files

def run_crop_session(settings: Dict[str, Any], stop_event: threading.Event):
    """Sets up and executes a cropping session."""
    try:
        input_dir = Path(settings['input_dir'])
        output_dir = Path(settings['output_dir'])
        
        if not input_dir.exists() or not input_dir.is_dir():
            logging.error(f"Input directory not found: {input_dir}")
            sys.exit(1)
        
        # --- THIS IS THE LINE TO CHANGE ---
        # Original line: nc_files = list(input_dir.glob("*.nc"))
        # Using rglob() instead of glob() tells Pathlib to search recursively
        # through all subdirectories for files ending in .nc.
        nc_files = list(input_dir.rglob("*.nc"))
        
        if not nc_files:
            logging.warning(f"No NetCDF (.nc) files found in {input_dir} or its subdirectories.")
            sys.exit(0)
            
        logging.info(f"Found {len(nc_files)} NetCDF files to process.")
        
        # The rest of the function remains the same...
        tasks = [(p, output_dir / f"{p.stem}_cropped.nc") for p in nc_files]
        
        cropper = Cropper(settings, stop_event)
        successful_crops, total_files = cropper.crop_all(tasks)

        if stop_event.is_set():
            logging.warning("Process was stopped before completion.")
        
        logging.info(f"Completed: {successful_crops}/{total_files} files cropped successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred during the session: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    """Add NetCDF cropping arguments to the provided parser."""
    io_group = parser.add_argument_group('Input and Output')
    bounds_group = parser.add_argument_group('Cropping Bounds')
    settings_group = parser.add_argument_group('Processing Settings')

    io_group.add_argument("--input_dir", help="Directory containing input NetCDF files (required if not in demo mode).")
    io_group.add_argument("--output_dir", help="Directory to save cropped files (required if not in demo mode).")

    bounds_group.add_argument("--min_lat", type=float, help="Minimum latitude (required if not in demo mode).")
    bounds_group.add_argument("--max_lat", type=float, help="Maximum latitude (required if not in demo mode).")
    bounds_group.add_argument("--min_lon", type=float, help="Minimum longitude (required if not in demo mode).")
    bounds_group.add_argument("--max_lon", type=float, help="Maximum longitude (required if not in demo mode).")

    settings_group.add_argument("--log-dir", default="./gridflow_logs", help="Directory to save log files.")
    settings_group.add_argument("--log-level", default="verbose", choices=["minimal", "verbose", "debug"], help="Set logging verbosity for the console.")
    settings_group.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of parallel workers.")
    settings_group.add_argument("--demo", action="store_true", help="Run with a pre-defined demo crop using default folders and bounds.")

def main(args=None):
    """Main entry point for the NetCDF Cropping Tool."""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="NetCDF Cropping Tool", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()
    
    setup_logging(args.log_dir, args.log_level, prefix="crop_netcdf")
    signal.signal(signal.SIGINT, signal_handler)
    
    settings = vars(args)
    if args.demo:
        logging.info("Running in demo mode.")
        settings['input_dir'] = settings.get('input_dir') or './downloads_cmip6'
        settings['output_dir'] = settings.get('output_dir') or './cropped_cmip6'
        settings['bounds'] = {'min_lat': 25.0, 'max_lat': 50.0, 'min_lon': -125.0, 'max_lon': -65.0}
        logging.info(f"Demo mode will use input '{settings['input_dir']}' and output to '{settings['output_dir']}'.")
    else:
        if not all([args.input_dir, args.output_dir, args.min_lat, args.max_lat, args.min_lon, args.max_lon]):
            logging.error("all input/output and lat/lon arguments are required when not in --demo mode.")
            sys.exit(1)
        settings['bounds'] = {'min_lat': args.min_lat, 'max_lat': args.max_lat, 'min_lon': args.min_lon, 'max_lon': args.max_lon}

    run_crop_session(settings, stop_event)
    
    if stop_event.is_set():
        logging.warning("Execution was interrupted.")
        sys.exit(130)
    
    logging.info("Process finished.")

if __name__ == "__main__":
    main()
