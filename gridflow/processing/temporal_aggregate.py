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
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Any

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

def find_time_variable(ds: nc.Dataset) -> Optional[str]:
    """Finds the name of the time variable in a NetCDF dataset."""
    for name, var in ds.variables.items():
        if hasattr(var, 'axis') and var.axis.lower() == 't':
            return name
        if hasattr(var, 'standard_name') and var.standard_name == 'time':
            return name
    for name in ['time', 't', 'datetime']:
        if name in ds.variables:
            return name
    return None

def aggregate_single_file(
    input_path: Path,
    output_path: Path,
    variable: str,
    output_frequency: str,
    agg_method: str,
    stop_event: threading.Event
) -> bool:
    """Aggregates a single NetCDF file to a specified temporal frequency."""
    if stop_event.is_set(): return False
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with nc.Dataset(input_path, 'r') as src:
            time_var_name = find_time_variable(src)
            if not time_var_name or variable not in src.variables:
                logging.error(f"Required time or variable '{variable}' not found in {input_path.name}. Skipping.")
                return False

            time_var = src.variables[time_var_name]
            time_units = time_var.units
            time_calendar = getattr(time_var, 'calendar', 'standard')
            
            dates = nc.num2date(time_var[:], units=time_units, calendar=time_calendar)
            data = src.variables[variable][:]

            grouped_data = defaultdict(list)
            for i, dt in enumerate(dates):
                if output_frequency == 'monthly':
                    key = dt.strftime('%Y-%m')
                elif output_frequency == 'annual':
                    key = dt.strftime('%Y')
                elif output_frequency == 'seasonal':
                    season = (dt.month % 12 + 3) // 3
                    year = dt.year if dt.month > 2 else dt.year - 1
                    key = f"{year}-S{season}"
                else:
                    logging.error(f"Unsupported frequency '{output_frequency}'.")
                    return False
                # Ensure we handle masked arrays correctly
                grouped_data[key].append(data[i, :, :])

            agg_results, new_dates = [], []
            agg_func = getattr(np, f"nan{agg_method}", None)
            if not agg_func:
                logging.error(f"Unsupported aggregation method '{agg_method}'. Use mean, sum, min, or max.")
                return False

            for key, group in sorted(grouped_data.items()):
                agg_results.append(agg_func(np.ma.array(group), axis=0))
                if output_frequency == 'monthly':
                    new_dates.append(nc.date2num(datetime.strptime(f"{key}-15", "%Y-%m-%d"), units=time_units, calendar=time_calendar))
                elif output_frequency == 'annual':
                    new_dates.append(nc.date2num(datetime.strptime(f"{key}-07-01", "%Y-%m-%d"), units=time_units, calendar=time_calendar))
                elif output_frequency == 'seasonal':
                    year, season_num = int(key.split('-S')[0]), int(key.split('-S')[1])
                    month = [2, 5, 8, 11][season_num-1]
                    new_dates.append(nc.date2num(datetime(year, month, 15), units=time_units, calendar=time_calendar))

            agg_data = np.ma.array(agg_results)

            with nc.Dataset(output_path, 'w', format=src.file_format) as dst:
                dst.setncatts(src.__dict__)
                dst.setncattr("aggregation_details", f"Aggregated from {input_path.name} using method '{agg_method}' to '{output_frequency}' frequency.")

                for name, dim in src.dimensions.items():
                    size = len(new_dates) if name == time_var_name else len(dim)
                    dst.createDimension(name, size if not dim.isunlimited() else None)

                for name, var in src.variables.items():
                    if stop_event.is_set(): return False
                    dst_var = dst.createVariable(name, var.dtype, var.dimensions, fill_value=getattr(var, '_FillValue', None))
                    dst_var.setncatts({k: v for k, v in var.__dict__.items() if k != '_FillValue'})
                    
                    if name == time_var_name:
                        dst_var[:] = new_dates
                    elif name == variable:
                        dst_var[:] = agg_data
                    elif time_var_name not in var.dimensions:
                        dst_var[:] = var[:]

        logging.info(f"Successfully aggregated {input_path.name} to {output_path.name}")
        return True
    except Exception as e:
        logging.error(f"Failed to aggregate {input_path.name}: {e}", exc_info=True)
        if output_path.exists(): output_path.unlink(missing_ok=True)
        return False

class TemporalAggregator:
    """Manages the parallel aggregation of NetCDF files."""
    def __init__(self, settings: Dict[str, Any], stop_event: threading.Event):
        self.settings = settings
        self._stop_event = stop_event
        self.executor = None

    def shutdown(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        logging.info("Aggregator has been shut down.")

    def aggregate_all(self, files_to_process: List[Tuple[Path, Path]]) -> Tuple[int, int]:
        successful_aggregations = 0
        if not files_to_process: return 0, 0
        
        self.executor = ThreadPoolExecutor(max_workers=self.settings['workers'], thread_name_prefix='Aggregator')
        future_to_file = {
            self.executor.submit(aggregate_single_file, in_path, out_path, self.settings['variable'], self.settings['output_frequency'], self.settings['method'], self._stop_event): in_path
            for in_path, out_path in files_to_process
        }
        
        total_files = len(files_to_process)
        try:
            for i, future in enumerate(as_completed(future_to_file)):
                logging.info(f"Progress: {i + 1}/{total_files} files processed.")
                if self._stop_event.is_set(): break
                if future.result():
                    successful_aggregations += 1
        finally:
            self.shutdown()
        
        return successful_aggregations, total_files

def run_aggregator_session(settings: Dict[str, Any], stop_event: threading.Event):
    """Sets up and executes an aggregation session."""
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
            
        logging.info(f"Found {len(nc_files)} NetCDF files to process.")
        
        tasks = [(p, output_dir / f"{p.stem}_{settings['output_frequency']}_{settings['method']}.nc") for p in nc_files]
        
        aggregator = TemporalAggregator(settings, stop_event)
        successful_aggregations, total_files = aggregator.aggregate_all(tasks)

        if stop_event.is_set():
            logging.warning("Process was stopped before completion.")
        
        logging.info(f"Completed: {successful_aggregations}/{total_files} files aggregated successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred during the session: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    """Add temporal aggregation arguments to the provided parser."""
    io_group = parser.add_argument_group('Input and Output')
    agg_group = parser.add_argument_group('Aggregation Parameters')
    settings_group = parser.add_argument_group('Processing Settings')

    io_group.add_argument("--input_dir", help="Directory containing input NetCDF files.")
    io_group.add_argument("--output_dir", help="Directory to save aggregated files.")
    io_group.add_argument("--variable", help="Name of the variable to aggregate (e.g., 'tas', 'pr').")
    
    agg_group.add_argument("--output_frequency", default="monthly", choices=["monthly", "seasonal", "annual"], help="Target frequency for aggregation.")
    agg_group.add_argument("--method", default="mean", choices=["mean", "sum", "min", "max"], help="Aggregation method.")
    
    settings_group.add_argument("--log-dir", default="./gridflow_logs", help="Directory to save log files.")
    settings_group.add_argument("--log-level", default="verbose", choices=["minimal", "verbose", "debug"], help="Set logging verbosity.")
    settings_group.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of parallel workers.")
    settings_group.add_argument("--demo", action="store_true", help="Run with demo defaults using daily data.")

def main(args=None):
    """Main entry point for the Temporal Aggregator CLI."""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="Temporal Aggregator for NetCDF Files", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()
    
    setup_logging(args.log_dir, args.log_level, prefix="temporal_aggregator")
    signal.signal(signal.SIGINT, signal_handler)
    
    settings = vars(args)
    if args.demo:
        logging.info("Running in demo mode.")
        settings['input_dir'] = './downloads_cmip6'
        settings['output_dir'] = './aggregated_cmip6'
        settings['variable'] = 'tas'
        settings['output_frequency'] = 'monthly'
        settings['method'] = 'mean'
        logging.info(f"Demo will aggregate daily 'tas' data from '{settings['input_dir']}' to monthly mean in '{settings['output_dir']}'.")
    else:
        required_args = ['input_dir', 'output_dir', 'variable']
        if not all(settings.get(arg) for arg in required_args):
            logging.error("the following arguments are required when not in --demo mode: --input_dir, --output_dir, --variable")
            sys.exit(1)

    run_aggregator_session(settings, stop_event)
    
    if stop_event.is_set():
        logging.warning("Execution was interrupted.")
        sys.exit(130)
    
    logging.info("Process finished.")

if __name__ == "__main__":
    main()
