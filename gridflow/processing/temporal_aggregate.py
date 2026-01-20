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
) -> Tuple[bool, str]:
    """Aggregates a single NetCDF file using a thread-safe 2-pass approach."""
    if stop_event.is_set(): 
        return False, "Interrupted"
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with HDF5_LOCK:
            with nc.Dataset(input_path, 'r') as src:
                time_var_name = find_time_variable(src)
                if not time_var_name or variable not in src.variables:
                    return False, f"Required time or variable '{variable}' not found"

                time_var = src.variables[time_var_name]
                time_units = time_var.units
                time_calendar = getattr(time_var, 'calendar', 'standard')
                file_format = src.file_format
                
                # Read data into memory so we can release the lock
                raw_time = time_var[:]
                raw_data = src.variables[variable][:]

        # --- COMPUTATION (UNLOCKED / PARALLEL) ---
        dates = nc.num2date(raw_time, units=time_units, calendar=time_calendar)
        data = raw_data

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
                return False, f"Unsupported frequency '{output_frequency}'"
            grouped_data[key].append(data[i, :, :])

        agg_results, new_dates = [], []
        agg_func = getattr(np, f"nan{agg_method}", None)
        if not agg_func:
            return False, f"Unsupported method '{agg_method}'"

        for key, group in sorted(grouped_data.items()):
            if stop_event.is_set(): return False, "Interrupted"
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

        with HDF5_LOCK:
            if stop_event.is_set(): return False, "Interrupted"
            
            # Re-open src to copy attributes/other vars, Open dst to write
            with nc.Dataset(input_path, 'r') as src:
                with nc.Dataset(output_path, 'w', format=file_format) as dst:
                    dst.setncatts(src.__dict__)
                    dst.setncattr("aggregation_details", f"Method: {agg_method}, Frequency: {output_frequency}")

                    for name, dim in src.dimensions.items():
                        size = len(new_dates) if name == time_var_name else len(dim)
                        dst.createDimension(name, size if not dim.isunlimited() else None)

                    for name, var in src.variables.items():
                        if stop_event.is_set(): return False, "Interrupted"
                        fill_value = getattr(var, '_FillValue', None)
                        dst_var = dst.createVariable(name, var.dtype, var.dimensions, fill_value=fill_value)
                        dst_var.setncatts({k: v for k, v in var.__dict__.items() if k != '_FillValue'})
                        
                        if name == time_var_name:
                            dst_var[:] = new_dates
                        elif name == variable:
                            dst_var[:] = agg_data
                        elif time_var_name not in var.dimensions:
                            dst_var[:] = var[:]

        return True, output_path.name
    except Exception as e:
        if output_path.exists(): output_path.unlink(missing_ok=True)
        return False, str(e)

class TemporalAggregator:
    """Manages the parallel aggregation of NetCDF files."""
    def __init__(self, settings: Dict[str, Any], stop_event: threading.Event):
        self.settings = settings
        self._stop_event = stop_event
        self.executor = None
        self.successful_aggregations = 0

    def shutdown(self, wait: bool = True):
        """Shuts down the thread pool."""
        if self.executor:
            self.executor.shutdown(wait=wait, cancel_futures=True)
            self.executor = None
        logging.info("Aggregator has been shut down.")

    def aggregate_all(self, files_to_process: List[Tuple[Path, Path]]) -> Tuple[int, int]:
        """Manages the parallel aggregation with Rich and tqdm support."""
        if not files_to_process:
            return 0, 0
        
        total_files = len(files_to_process)
        workers = self.settings.get('workers', os.cpu_count() or 4)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='Aggregator')

        # Determine UI mode
        is_gui_mode = self.settings.get('is_gui_mode', False)
        use_tqdm = HAS_UI_LIBS and not is_gui_mode

        future_to_file = {
            self.executor.submit(
                aggregate_single_file, 
                in_path, out_path, 
                self.settings['variable'], 
                self.settings['output_frequency'], 
                self.settings['method'], 
                self._stop_event
            ): in_path
            for in_path, out_path in files_to_process
        }
        
        futures_iter = as_completed(future_to_file)

        if use_tqdm:
            futures_iter = tqdm(
                futures_iter, 
                total=total_files, 
                unit="file", 
                desc="Aggregating", 
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
                        self.successful_aggregations += 1
                        if use_tqdm:
                            tqdm.write(f"  ✔ Aggregated {result_msg}")
                        elif is_gui_mode:
                            logging.info(f"Aggregated {result_msg}")
                    else:
                        if use_tqdm:
                            tqdm.write(f"  ✖ Failed {original_path.name}: {result_msg}")
                        elif is_gui_mode:
                            logging.info(f"Failed {original_path.name}: {result_msg}")
                except Exception as e:
                    logging.error(f"Error processing {original_path.name}: {e}")
        
        finally:
            self.shutdown(wait=not self._stop_event.is_set())
        
        return self.successful_aggregations, total_files

def run_aggregator_session(settings: Dict[str, Any], stop_event: threading.Event):
    """Sets up and executes an aggregation session."""
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
        
        tasks = [(p, output_dir / f"{p.stem}_{settings['output_frequency']}_{settings['method']}.nc") for p in nc_files]
        
        aggregator = TemporalAggregator(settings, stop_event)
        successful, total = aggregator.aggregate_all(tasks)

        if stop_event.is_set():
            logging.warning("Process was stopped before completion.")
        
        logging.info(f"Completed: {successful}/{total} files aggregated successfully.")

    except Exception as e:
        logging.info(f"Failed: A critical error occurred. See log file for details.")
        logging.debug(f"Full critical error trace: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    """Add temporal aggregation arguments to the provided parser."""
    io_group = parser.add_argument_group('Input and Output')
    agg_group = parser.add_argument_group('Aggregation Parameters')
    settings_group = parser.add_argument_group('Processing Settings')

    io_group.add_argument("-i", "--input_dir", help="Directory containing input NetCDF files.")
    io_group.add_argument("-o", "--output_dir", help="Directory to save aggregated files.")
    io_group.add_argument("-var", "--variable", help="Variable to aggregate (e.g., 'tas', 'pr').")

    agg_group.add_argument("--output_frequency", default="monthly", choices=["monthly", "seasonal", "annual"], help="Target frequency.")
    agg_group.add_argument("--method", default="mean", choices=["mean", "sum", "min", "max"], help="Method.")
    
    settings_group.add_argument("-ld", "--log-dir", default="./gridflow_logs", help="Log directory.")
    settings_group.add_argument("-ll", "--log-level", default="minimal", choices=["minimal", "verbose", "debug"], help="Log verbosity.")
    settings_group.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 4, help="Parallel workers.")
    settings_group.add_argument("--demo", action="store_true", help="Run with demo defaults.")

def main(args=None):
    """Main entry point with Rich-enhanced demo reporting."""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="Temporal Aggregator for NetCDF Files", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()
    
    # FIX: Only set up logging if NOT in GUI mode
    if not getattr(args, 'is_gui_mode', False):
        setup_logging(args.log_dir, args.log_level, prefix="temporal_aggregator")
        signal.signal(signal.SIGINT, signal_handler)

    active_stop_event = getattr(args, 'stop_event', stop_event)
    
    settings = vars(args)
    is_gui_mode = settings.get('is_gui_mode', False)

    if args.demo:
        settings['input_dir'] = './downloads_cmip6'
        settings['output_dir'] = './aggregated_cmip6'
        settings['variable'] = 'tas'
        settings['output_frequency'] = 'monthly'
        settings['method'] = 'mean'
        
        demo_cmd = (
            "gridflow aggregate "
            "-i ./downloads_cmip6 "
            "-o ./aggregated_cmip6 "
            "-var tas "
            "--output_frequency monthly "
            "--method mean"
        )

        if HAS_UI_LIBS and not is_gui_mode:
            console = Console()
            console.print(f"[bold yellow]Running in demo mode.[/]")
            console.print(f"Demo Command:\n  [dim]{demo_cmd}[/dim]\n")
        else:
            logging.info(f"Running in demo mode.\nDemo Command: {demo_cmd}")
    else:
        required_args = ['input_dir', 'output_dir', 'variable']
        if not all(settings.get(arg) for arg in required_args):
            logging.error("Required arguments missing: --input_dir, --output_dir, --variable")
            if not is_gui_mode: sys.exit(1)
            return

    run_aggregator_session(settings, active_stop_event)
    
    if active_stop_event.is_set():
        logging.info("Execution was interrupted.")
        if not is_gui_mode: sys.exit(130)
    
    logging.info("Process finished.")

if __name__ == "__main__":
    main()