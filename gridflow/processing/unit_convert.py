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
from typing import List, Dict, Tuple, Any, Callable

# --- Dependency Check ---
try:
    import netCDF4 as nc
    import numpy as np
except ImportError as e:
    print(f"FATAL: Missing required library: {e.name}. Please install it using 'pip install {e.name}'.")
    sys.exit(1)

# --- Local Imports ---
try:
    from gridflow.utils.logging_utils import setup_logging
except ImportError:
    print("FATAL: logging_utils.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

# --- Suppress known warnings ---
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Global Stop Event for Graceful Shutdown ---
stop_event = threading.Event()

def signal_handler(sig, frame):
    """Handles SIGINT (Ctrl+C) to gracefully shut down the application."""
    logging.warning("Stop signal received! Gracefully shutting down...")
    stop_event.set()
    logging.info("Please wait for ongoing tasks to complete.")

# --- Conversion Logic ---

def k_to_c(data):
    """Kelvin to Celsius"""
    return data - 273.15

def flux_to_mm_day(data):
    """kg m-2 s-1 to mm/day"""
    return data * 86400

def m_s_to_km_h(data):
    """m/s to km/h"""
    return data * 3.6

CONVERSIONS: Dict[str, Dict[str, Tuple[Callable, str]]] = {
    'tas': {'C': (k_to_c, 'K')},
    'tmin': {'C': (k_to_c, 'K')},
    'tmax': {'C': (k_to_c, 'K')},
    'pr': {'mm/day': (flux_to_mm_day, 'kg m-2 s-1')},
    'sfcWind': {'km/h': (m_s_to_km_h, 'm s-1')},
}

def convert_single_file(
    input_path: Path,
    output_path: Path,
    variable: str,
    target_unit: str,
    stop_event: threading.Event
) -> bool:
    """Converts the units of a variable in a single NetCDF file."""
    if stop_event.is_set(): return False
    
    try:
        # Check if a valid conversion exists
        conversion_info = CONVERSIONS.get(variable, {}).get(target_unit)
        if not conversion_info:
            logging.error(f"No conversion defined for variable '{variable}' to target unit '{target_unit}'. Skipping {input_path.name}.")
            return False

        conversion_func, source_unit_pattern = conversion_info
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with nc.Dataset(input_path, 'r') as src:
            if variable not in src.variables:
                logging.warning(f"Variable '{variable}' not found in {input_path.name}. Skipping.")
                return False
            
            src_var = src.variables[variable]
            current_unit = getattr(src_var, 'units', '').strip()

            if source_unit_pattern not in current_unit:
                logging.error(f"Source unit mismatch in {input_path.name}. Expected '{source_unit_pattern}', found '{current_unit}'. Skipping.")
                return False

            with nc.Dataset(output_path, 'w', format=src.file_format) as dst:
                # Copy metadata and dimensions
                dst.setncatts(src.__dict__)
                dst.setncattr("unit_conversion_details", f"Variable '{variable}' converted from '{current_unit}' to '{target_unit}'.")
                for name, dim in src.dimensions.items():
                    dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

                # Copy variables, converting the target one
                for name, var in src.variables.items():
                    if stop_event.is_set(): return False
                    fill_value = var.getncattr('_FillValue') if '_FillValue' in var.ncattrs() else None
                    dst_var = dst.createVariable(name, var.dtype, var.dimensions, fill_value=fill_value)
                    attrs_to_copy = {k: var.getncattr(k) for k in var.ncattrs() if k != '_FillValue'}
                    dst_var.setncatts(attrs_to_copy)
                    
                    if name == variable:
                        logging.debug(f"Applying conversion to '{name}' in {input_path.name}.")
                        converted_data = conversion_func(var[:])
                        dst_var[:] = converted_data
                        dst_var.units = target_unit
                    else:
                        dst_var[:] = var[:]
                        
        logging.info(f"Successfully converted '{variable}' in {input_path.name} to {output_path.name}")
        return True
    except Exception as e:
        logging.error(f"Failed to convert {input_path.name}: {e}", exc_info=True)
        if output_path.exists(): output_path.unlink(missing_ok=True)
        return False

class UnitConverter:
    """Manages the parallel unit conversion of NetCDF files."""
    def __init__(self, settings: Dict[str, Any], stop_event: threading.Event):
        self.settings = settings
        self._stop_event = stop_event
        self.executor = None

    def shutdown(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        logging.info("Converter has been shut down.")

    def convert_all(self, files_to_convert: List[Tuple[Path, Path]]) -> Tuple[int, int]:
        """Manages the parallel conversion of a list of files."""
        successful_conversions = 0
        if not files_to_convert: return 0, 0
        
        self.executor = ThreadPoolExecutor(max_workers=self.settings['workers'], thread_name_prefix='Converter')
        future_to_file = {
            self.executor.submit(convert_single_file, in_path, out_path, self.settings['variable'], self.settings['target_unit'], self._stop_event): in_path
            for in_path, out_path in files_to_convert
        }
        
        total_files = len(files_to_convert)
        try:
            for i, future in enumerate(as_completed(future_to_file)):
                logging.info(f"Progress: {i + 1}/{total_files} files processed.")
                if self._stop_event.is_set(): break
                if future.result():
                    successful_conversions += 1
        finally:
            self.shutdown()
        
        return successful_conversions, total_files

def run_conversion_session(settings: Dict[str, Any], stop_event: threading.Event):
    """Sets up and executes a unit conversion session."""
    try:
        input_dir = Path(settings['input_dir'])
        output_dir = Path(settings['output_dir'])
        
        if not input_dir.is_dir():
            logging.error(f"Input directory not found: {input_dir}")
            sys.exit(1)
        
        nc_files = list(input_dir.rglob("*.nc"))
        if not nc_files:
            logging.warning(f"No NetCDF (.nc) files found in {input_dir} or its subdirectories.")
            sys.exit(0)
            
        logging.info(f"Found {len(nc_files)} NetCDF files to process for unit conversion.")
        
        tasks = [(p, output_dir / f"{p.stem}_converted.nc") for p in nc_files]
        
        converter = UnitConverter(settings, stop_event)
        successful_conversions, total_files = converter.convert_all(tasks)

        if stop_event.is_set():
            logging.warning("Process was stopped before completion.")
        
        logging.info(f"Completed: {successful_conversions}/{total_files} files converted successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred during the session: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    """Add unit conversion arguments to the provided parser."""
    io_group = parser.add_argument_group('Input and Output')
    conv_group = parser.add_argument_group('Conversion Parameters')
    settings_group = parser.add_argument_group('Processing Settings')

    io_group.add_argument("--input_dir", help="Directory containing input NetCDF files.")
    io_group.add_argument("--output_dir", help="Directory to save converted files.")
    
    conv_group.add_argument("--variable", help="Name of the variable to convert (e.g., 'tas', 'pr').")
    conv_group.add_argument("--target_unit", help="Target unit to convert to (e.g., 'C', 'mm/day', 'km/h').")
    
    settings_group.add_argument("--log-dir", default="./gridflow_logs", help="Directory to save log files.")
    settings_group.add_argument("--log-level", default="verbose", choices=["minimal", "verbose", "debug"], help="Set logging verbosity.")
    settings_group.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of parallel workers.")
    settings_group.add_argument("--demo", action="store_true", help="Run with demo defaults.")

def main(args=None):
    """Main entry point for the Unit Converter CLI."""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="Unit Converter for NetCDF Files", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()
    
    setup_logging(args.log_dir, args.log_level, prefix="unit_converter")
    signal.signal(signal.SIGINT, signal_handler)
    
    settings = vars(args)
    if args.demo:
        logging.info("Running in demo mode.")
        settings['input_dir'] = './downloads_cmip6'
        settings['output_dir'] = './unit-converted_cmip6'
        settings['variable'] = 'tas'
        settings['target_unit'] = 'C'
        logging.info(f"Demo will use input '{settings['input_dir']}', output to '{settings['output_dir']}', and convert '{settings['variable']}' to '{settings['target_unit']}'.")
    else:
        required_args = ['input_dir', 'output_dir', 'variable', 'target_unit']
        if not all(settings.get(arg) for arg in required_args):
            logging.error("the following arguments are required when not in --demo mode: --input_dir, --output_dir, --variable, --target_unit")
            sys.exit(1)

    run_conversion_session(settings, stop_event)
    
    if stop_event.is_set():
        logging.warning("Execution was interrupted.")
        sys.exit(130)
    
    logging.info("Process finished.")

if __name__ == "__main__":
    main()

