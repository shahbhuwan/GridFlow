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
import warnings
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

try:
    from tqdm import tqdm
    from rich.console import Console
    HAS_UI_LIBS = True
except ImportError:
    HAS_UI_LIBS = False

# --- Local Imports ---
try:
    from gridflow.utils.logging_utils import setup_logging
except ImportError:
    # Fallback if module is run standalone
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    def setup_logging(*args, **kwargs): pass

# --- Suppress known warnings ---
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Global Stop Event ---
stop_event = threading.Event()

HDF5_LOCK = threading.Lock()

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
) -> Tuple[bool, str]:
    """Crops a single NetCDF file using a thread-safe 2-pass approach."""
    if stop_event.is_set(): 
        return False, "Interrupted"
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with HDF5_LOCK:
            with nc.Dataset(input_path, 'r') as src:
                lat_var_name, lon_var_name = find_coordinate_vars(src)
                if not lat_var_name or not lon_var_name:
                    return False, "Missing coordinate variables"

                lat_data = src.variables[lat_var_name][:]
                lon_data = src.variables[lon_var_name][:]
                file_format = src.file_format  

        # --- COMPUTATION (UNLOCKED / PARALLEL) ---
        # Normalize longitude to -180 to 180
        lon_data_norm = np.where(lon_data > 180, lon_data - 360, lon_data)

        lat_indices = np.where((lat_data >= bounds['min_lat']) & (lat_data <= bounds['max_lat']))[0]
        lon_indices = np.where((lon_data_norm >= bounds['min_lon']) & (lon_data_norm <= bounds['max_lon']))[0]

        if lat_indices.size == 0 or lon_indices.size == 0:
            return False, "No data in bounds"
        
        lat_start, lat_end = lat_indices.min(), lat_indices.max()
        lon_start, lon_end = lon_indices.min(), lon_indices.max()

        with HDF5_LOCK:
            if stop_event.is_set(): return False, "Interrupted"
            
            # Re-open src to read variables, open dst to write
            with nc.Dataset(input_path, 'r') as src:
                with nc.Dataset(output_path, 'w', format=file_format) as dst:
                    dst.setncatts(src.__dict__)
                    
                    lat_dim_name = src.variables[lat_var_name].dimensions[0]
                    lon_dim_name = src.variables[lon_var_name].dimensions[0]
                    
                    dst.createDimension(lat_dim_name, lat_indices.size)
                    dst.createDimension(lon_dim_name, lon_indices.size)
                    
                    for name, dim in src.dimensions.items():
                        if name not in [lat_dim_name, lon_dim_name]:
                            dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

                    for name, var in src.variables.items():
                        if stop_event.is_set(): return False, "Interrupted"

                        fill_value = var.getncattr('_FillValue') if '_FillValue' in var.ncattrs() else None
                        dst_var = dst.createVariable(name, var.dtype, var.dimensions, fill_value=fill_value)
                        
                        attrs_to_copy = {k: var.getncattr(k) for k in var.ncattrs() if k != '_FillValue'}
                        dst_var.setncatts(attrs_to_copy)
                        
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
                        
        return True, output_path.name
    except Exception as e:
        if output_path.exists(): output_path.unlink(missing_ok=True)
        return False, str(e)

class Cropper:
    """Manages the parallel cropping of NetCDF files using a thread pool."""
    def __init__(self, settings: Dict[str, Any], stop_event: threading.Event):
        self.settings = settings
        self._stop_event = stop_event
        self.executor = None
        self.successful_crops = 0

    def shutdown(self, wait: bool = True):
        """Shuts down the thread pool."""
        if self.executor:
            self.executor.shutdown(wait=wait, cancel_futures=True)
            self.executor = None
        logging.info("Cropper has been shut down.")

    def crop_all(self, files_to_crop: List[Tuple[Path, Path]]) -> Tuple[int, int]:
        """Manages parallel execution with Rich and tqdm support."""
        if not files_to_crop:
            return 0, 0
        
        total_files = len(files_to_crop)
        workers = self.settings.get('workers', os.cpu_count() or 4)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='Cropper')

        is_gui_mode = self.settings.get('is_gui_mode', False)
        use_tqdm = HAS_UI_LIBS and not is_gui_mode

        future_to_file = {
            self.executor.submit(
                crop_single_file, 
                in_path, out_path, 
                self.settings['bounds'], 
                self._stop_event
            ): in_path
            for in_path, out_path in files_to_crop
        }
        
        futures_iter = as_completed(future_to_file)

        if use_tqdm:
            futures_iter = tqdm(
                futures_iter, 
                total=total_files, 
                unit="file", 
                desc="Cropping", 
                ncols=90, 
                bar_format='  {l_bar}{bar}{r_bar}'
            )

        try:
            for i, future in enumerate(futures_iter):
                if self._stop_event.is_set():
                    for f in future_to_file: f.cancel()
                    break

                if is_gui_mode:
                    logging.info(f"Progress: {i + 1}/{total_files} files processed.")

                original_path = future_to_file[future]
                try:
                    success, result_msg = future.result()
                    if success:
                        self.successful_crops += 1
                        if use_tqdm:
                            tqdm.write(f"  ✔ Cropped {result_msg}")
                        elif is_gui_mode:
                            logging.info(f"Cropped {result_msg}")
                    else:
                        if use_tqdm:
                            tqdm.write(f"  ✖ Failed {original_path.name}: {result_msg}")
                        elif is_gui_mode:
                            logging.info(f"Failed {original_path.name}: {result_msg}")
                except Exception as e:
                    logging.error(f"Error processing {original_path.name}: {e}")
        
        finally:
            self.shutdown(wait=not self._stop_event.is_set())
        
        return self.successful_crops, total_files

def run_crop_session(settings: Dict[str, Any], stop_event: threading.Event):
    """Sets up and executes a cropping session."""
    is_gui_mode = settings.get('is_gui_mode', False)
    use_rich = HAS_UI_LIBS and not is_gui_mode

    try:
        input_dir = Path(settings['input_dir'])
        output_dir = Path(settings['output_dir'])
        
        if not input_dir.is_dir():
            logging.error(f"Input directory not found: {input_dir}")
            if not is_gui_mode: sys.exit(1)
            return
        
        nc_files = list(input_dir.rglob("*.nc"))
        if not nc_files:
            logging.warning(f"No NetCDF (.nc) files found in {input_dir}.")
            if not is_gui_mode: sys.exit(0)
            return
            
        if use_rich:
            Console().print(f"[bold blue]Initialization:[/] Found {len(nc_files)} NetCDF files to process.")
        else:
            logging.info(f"Found {len(nc_files)} NetCDF files to process.")
        
        tasks = [(p, output_dir / f"{p.stem}_cropped.nc") for p in nc_files]
        
        cropper = Cropper(settings, stop_event)
        successful, total = cropper.crop_all(tasks)

        if stop_event.is_set():
            logging.warning("Process was stopped before completion.")
        
        logging.info(f"Completed: {successful}/{total} files cropped successfully.")

    except Exception as e:
        logging.info(f"Failed: A critical error occurred. See log file for details.")
        logging.debug(f"Full critical error trace: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    """Add cropping arguments to the provided parser."""
    io_group = parser.add_argument_group('Input and Output')
    bounds_group = parser.add_argument_group('Cropping Bounds')
    settings_group = parser.add_argument_group('Processing Settings')

    io_group.add_argument("-i", "--input_dir", help="Directory containing input NetCDF files.")
    io_group.add_argument("-o", "--output_dir", help="Directory to save cropped files.")

    bounds_group.add_argument("--min_lat", type=float, help="Minimum latitude.")
    bounds_group.add_argument("--max_lat", type=float, help="Maximum latitude.")
    bounds_group.add_argument("--min_lon", type=float, help="Minimum longitude.")
    bounds_group.add_argument("--max_lon", type=float, help="Maximum longitude.")

    settings_group.add_argument("-ld", "--log-dir", default="./gridflow_logs", help="Log directory.")
    settings_group.add_argument("-ll", "--log-level", default="minimal", choices=["minimal", "verbose", "debug"], help="Log verbosity.")
    settings_group.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 4, help="Parallel workers.")
    settings_group.add_argument("--demo", action="store_true", help="Run with demo defaults.")

def main(args=None):
    """Main entry point with Rich-enhanced reporting."""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="NetCDF Spatial Cropper", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()
    
    # FIX: Only set up logging if NOT in GUI mode
    if not getattr(args, 'is_gui_mode', False):
        setup_logging(args.log_dir, args.log_level, prefix="crop_netcdf")
        signal.signal(signal.SIGINT, signal_handler)

    active_stop_event = getattr(args, 'stop_event', stop_event)
    
    settings = vars(args)
    is_gui_mode = settings.get('is_gui_mode', False)

    if args.demo:
        settings['input_dir'] = './downloads_cmip6'
        settings['output_dir'] = './cropped_cmip6'
        settings['bounds'] = {'min_lat': 25.0, 'max_lat': 50.0, 'min_lon': -125.0, 'max_lon': -65.0}
        
        demo_cmd = "gridflow crop -i ./downloads_cmip6 -o ./cropped_cmip6 --min_lat 25 --max_lat 50 --min_lon -125 --max_lon -65"

        if HAS_UI_LIBS and not is_gui_mode:
            console = Console()
            console.print(f"[bold yellow]Running in demo mode.[/]")
            console.print(f"Demo Command:\n  [dim]{demo_cmd}[/dim]\n")
        else:
            logging.info(f"Running in demo mode.\nDemo Command: {demo_cmd}")
    else:
        required = ['input_dir', 'output_dir', 'min_lat', 'max_lat', 'min_lon', 'max_lon']
        if not all(settings.get(arg) is not None for arg in required):
            logging.error("Required arguments missing. Use --demo or provide all I/O and bound arguments.")
            if not is_gui_mode: sys.exit(1)
            return
        settings['bounds'] = {
            'min_lat': args.min_lat, 'max_lat': args.max_lat, 
            'min_lon': args.min_lon, 'max_lon': args.max_lon
        }

    run_crop_session(settings, active_stop_event)
    
    if active_stop_event.is_set():
        logging.info("Execution was interrupted.")
        if not is_gui_mode: sys.exit(130)
    
    logging.info("Process finished.")

if __name__ == "__main__":
    main()