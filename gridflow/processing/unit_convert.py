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
import json
import signal
import logging
import threading
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any, Callable, Optional

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
    # Fallback for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(asctime)s: %(message)s')
    def setup_logging(*args, **kwargs): pass

# --- Constants & Conversion Logic ---
warnings.filterwarnings("ignore", category=DeprecationWarning)

def k_to_c(data): return data - 273.15
def flux_to_mm_day(data): return data * 86400
def m_s_to_km_h(data): return data * 3.6

CONVERSIONS: Dict[str, Dict[str, Tuple[Callable, str]]] = {
    'tas': {'C': (k_to_c, 'K')},
    'tmin': {'C': (k_to_c, 'K')},
    'tmax': {'C': (k_to_c, 'K')},
    'pr': {'mm/day': (flux_to_mm_day, 'kg m-2 s-1')},
    'sfcWind': {'km/h': (m_s_to_km_h, 'm s-1')},
}

# --- Global Stop Event and Threading Lock ---
stop_event = threading.Event()
NETCDF_LOCK = threading.Lock()

def signal_handler(sig, frame):
    """Handles SIGINT (Ctrl+C)."""
    logging.warning("Stop signal received! Gracefully shutting down...")
    stop_event.set()

class FileManager:
    """Handles file discovery and directory management."""
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        try:
            if not self.input_dir.exists():
                raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.error(f"File system error: {e}")
            sys.exit(1)

    def get_netcdf_files(self) -> List[Path]:
        """Recursively finds all .nc files."""
        return list(self.input_dir.rglob("*.nc"))

    def get_output_path(self, input_path: Path) -> Path:
        """Generates the output path preserving sub-structures."""
        rel_path = input_path.relative_to(self.input_dir)
        out_path = self.output_dir / rel_path.parent / f"{input_path.stem}_converted.nc"
        return out_path

class Converter:
    """Manages the NetCDF unit conversion logic and parallel execution."""
    def __init__(self, file_manager: FileManager, stop_event: threading.Event, **settings):
        self.file_manager = file_manager
        self._stop_event = stop_event
        self.settings = settings
        self.executor = None

    def convert_file(self, input_path: Path) -> Tuple[bool, str]:
        """Processes a single file with unit conversion."""
        if self._stop_event.is_set(): return False, "Interrupted"
        
        output_path = self.file_manager.get_output_path(input_path)
        variable = self.settings['variable']
        target_unit = self.settings['target_unit']

        try:
            conv_info = CONVERSIONS.get(variable, {}).get(target_unit)
            if not conv_info:
                return False, f"No conversion defined for {variable} to {target_unit}"

            conv_func, source_unit_pattern = conv_info
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with NETCDF_LOCK:
                if self._stop_event.is_set(): return False, "Interrupted"

                with nc.Dataset(input_path, 'r') as src:
                    if variable not in src.variables:
                        return False, f"Variable '{variable}' not found"
                    
                    src_var = src.variables[variable]
                    current_unit = getattr(src_var, 'units', '').strip()

                    if source_unit_pattern not in current_unit:
                        return False, f"Unit mismatch: Expected {source_unit_pattern}, found {current_unit}"

                    with nc.Dataset(output_path, 'w', format=src.file_format) as dst:
                        dst.setncatts(src.__dict__)
                        dst.setncattr("unit_conversion", f"From {current_unit} to {target_unit}")
                        
                        for name, dim in src.dimensions.items():
                            dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

                        for name, var in src.variables.items():
                            if self._stop_event.is_set(): return False, "Interrupted"
                            
                            fill_value = var.getncattr('_FillValue') if '_FillValue' in var.ncattrs() else None
                            dst_var = dst.createVariable(name, var.dtype, var.dimensions, fill_value=fill_value)
                            dst_var.setncatts({k: var.getncattr(k) for k in var.ncattrs() if k != '_FillValue'})
                            
                            if name == variable:
                                dst_var[:] = conv_func(var[:])
                                dst_var.units = target_unit
                            else:
                                dst_var[:] = var[:]
                                
            return True, f"Converted: {output_path.name}"
        except Exception as e:
            if output_path.exists(): output_path.unlink(missing_ok=True)
            return False, str(e)

    def process_all(self, files: List[Path]) -> Tuple[int, int]:
        """Executes parallel processing."""
        if not files: return 0, 0
        
        successful = 0
        total = len(files)
        workers = self.settings.get('workers', 4)
        
        start_msg = f"Starting unit conversion for {total} files..."
        is_gui_mode = self.settings.get('is_gui_mode', False)
        
        if HAS_UI_LIBS and not is_gui_mode:
            from rich.console import Console
            Console().print(f"[bold blue]{start_msg}[/]")
        else:
            logging.info(start_msg)

        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='Converter')
        future_to_file = {self.executor.submit(self.convert_file, f): f for f in files}
        
        use_tqdm = HAS_UI_LIBS and not is_gui_mode
        futures_iter = as_completed(future_to_file)

        if use_tqdm:
            futures_iter = tqdm(futures_iter, total=total, unit="file", desc="Converting Units", ncols=90)

        try:
            for i, future in enumerate(futures_iter):
                if self._stop_event.is_set(): break
                
                if is_gui_mode:
                    logging.info(f"Progress: {i + 1}/{total} files processed.")

                orig_path = future_to_file[future]
                try:
                    success, msg = future.result()
                    if success:
                        successful += 1
                        if use_tqdm: tqdm.write(f"  ✔ {msg}")
                        elif is_gui_mode: logging.info(msg)
                    else:
                        if use_tqdm: tqdm.write(f"  ✖ Failed {orig_path.name}: {msg}")
                        elif is_gui_mode: logging.warning(f"Failed {orig_path.name}: {msg}")
                except Exception as e:
                    logging.error(f"Error processing {orig_path.name}: {e}")
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
        
        return successful, total

def load_config(config_path: str) -> Dict:
    """Loads JSON configuration."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return {}

def create_conversion_session(settings: Dict[str, Any], stop_event: threading.Event):
    """Orchestrates the conversion session."""
    is_gui_mode = settings.get('is_gui_mode', False)
    use_rich = HAS_UI_LIBS and not is_gui_mode
    
    try:
        file_manager = FileManager(settings['input_dir'], settings['output_dir'])
        nc_files = file_manager.get_netcdf_files()
        
        if not nc_files:
            logging.warning(f"No NetCDF files found in {settings['input_dir']}")
            if not is_gui_mode: sys.exit(0)
            return
            
        status_msg = f"Initializing unit conversion for {len(nc_files)} files..."
        
        settings.pop('stop_event', None)
        settings.pop('stop_flag', None)

        if use_rich:
            console = Console()
            with console.status(f"[bold green]{status_msg}", spinner="dots"):
                converter = Converter(file_manager, stop_event, **settings)
        else:
            logging.info(status_msg)
            converter = Converter(file_manager, stop_event, **settings)

        successful, total = converter.process_all(nc_files)

        if stop_event.is_set():
            logging.warning("Process was interrupted.")
        
        logging.info(f"Completed: {successful}/{total} files converted successfully.")
    except Exception as e:
        logging.critical(f"Critical session error: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    """CLI Argument groups."""
    io_group = parser.add_argument_group('Input and Output')
    conv_group = parser.add_argument_group('Conversion Parameters')
    settings_group = parser.add_argument_group('Processing Settings')

    io_group.add_argument("-i", "--input_dir", help="Input directory.")
    io_group.add_argument("-o", "--output_dir", help="Output directory.")

    conv_group.add_argument("--variable", help="Variable (e.g., 'tas', 'pr').")
    conv_group.add_argument("--target_unit", help="Target unit (e.g., 'C', 'mm/day').")

    settings_group.add_argument("-ld", "--log-dir", default="./gridflow_logs", help="Log directory.")
    settings_group.add_argument("-ll", "--log-level", default="verbose", choices=["minimal", "verbose", "debug"])
    settings_group.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 4)
    settings_group.add_argument("--demo", action="store_true", help="Run with demo defaults.")
    settings_group.add_argument("-c", "--config", help="Path to JSON config.")

def main(args=None):
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="GridFlow Unit Converter", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()

    # FIX: Only set up logging if NOT in GUI mode
    if not getattr(args, "is_gui_mode", False):
        setup_logging(args.log_dir, args.log_level, prefix="unit_converter")
        signal.signal(signal.SIGINT, signal_handler)

    active_stop_event = getattr(args, "stop_event", stop_event)

    config = load_config(args.config) if args.config else {}
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    settings = {**config, **cli_args}

    if settings.get('demo'):
        settings.update({
            "input_dir": "./downloads_cmip6", 
            "output_dir": "./unit_converted", 
            "variable": "tas", 
            "target_unit": "C"
        })

        demo_cmd = (
            "gridflow convert "
            "-i ./downloads_cmip6 "
            "-o ./unit_converted "
            "--variable tas "
            "--target_unit C"
        )

        if HAS_UI_LIBS and not getattr(args, 'is_gui_mode', False):
            console = Console()
            console.print(f"[bold yellow]Running in demo mode with a pre-defined query.[/]")
            console.print(f"Demo Command:\n  [dim]{demo_cmd}[/dim]\n")
        else:
            logging.info(f"Demo Command:\n  {demo_cmd}\n")
    else:
        for req in ["input_dir", "output_dir", "variable", "target_unit"]:
            if not settings.get(req):
                logging.error(f"Argument --{req} is required.")
                if not settings.get("is_gui_mode"): sys.exit(1)
                return

    create_conversion_session(settings, active_stop_event)
    if active_stop_event.is_set() and not settings.get("is_gui_mode"): sys.exit(130)

if __name__ == "__main__":
    main()