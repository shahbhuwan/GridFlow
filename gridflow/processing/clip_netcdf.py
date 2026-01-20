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
import signal
import sys
import json
import logging
import threading
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any, Optional

# --- Dependency Check ---
try:
    import netCDF4 as nc
    import numpy as np
    import geopandas as gpd
    
    try:
        from shapely import contains_xy
        contains_func = contains_xy
    except ImportError:
        from shapely.vectorized import contains as contains_func

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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    def setup_logging(*args, **kwargs): pass

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyogrio")

# --- Global Stop Event ---
stop_event = threading.Event()

hdf5_lock = threading.Lock()

def signal_handler(sig, frame):
    logging.info("Stop signal received! Gracefully shutting down...")
    stop_event.set()

class FileManager:
    """Handles file and directory management for clipping operations."""
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
        return list(self.input_dir.rglob("*.nc"))

    def get_output_path(self, input_path: Path) -> Path:
        rel_path = input_path.relative_to(self.input_dir)
        out_path = self.output_dir / rel_path.parent / f"{input_path.stem}_clipped.nc"
        return out_path

class Clipper:
    """Manages the spatial clipping logic using Threading + HDF5 Locking."""
    def __init__(self, file_manager: FileManager, stop_event: threading.Event, **settings):
        self.file_manager = file_manager
        self._stop_event = stop_event
        self.settings = settings
        self.executor = None
        self.shapefile_geom = None

    def load_shapefile(self) -> bool:
        """Loads and prepares the shapefile geometry."""
        shapefile_path = Path(self.settings['shapefile'])
        if not shapefile_path.is_file():
            logging.error(f"Shapefile not found: {shapefile_path}")
            return False

        try:
            logging.info(f"Loading shapefile: {shapefile_path.name}")
            gdf = gpd.read_file(shapefile_path)
            
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                logging.info(f"Reprojecting shapefile from {gdf.crs} to EPSG:4326")
                gdf = gdf.to_crs("EPSG:4326")

            buffer_km = self.settings.get('buffer_km', 0)
            if buffer_km > 0:
                logging.info(f"Applying {buffer_km}km buffer...")
                gdf_meters = gdf.to_crs("EPSG:6933") 
                gdf_meters['geometry'] = gdf_meters.buffer(buffer_km * 1000)
                gdf = gdf_meters.to_crs("EPSG:4326")

            try:
                self.shapefile_geom = gdf.union_all()
            except AttributeError:
                self.shapefile_geom = gdf.unary_union
            return True

        except Exception as e:
            logging.error(f"Failed to load shapefile: {e}")
            return False

    def clip_file(self, input_path: Path) -> Tuple[bool, str]:
        """Clips a single NetCDF file using a safe 2-pass Locked I/O approach."""
        if self._stop_event.is_set(): return False, "Interrupted"
        
        output_path = self.file_manager.get_output_path(input_path)
        
        if output_path.exists() and not self.settings.get('overwrite', True):
            return True, "Skipped (Exists)"

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with hdf5_lock:
                with nc.Dataset(input_path, 'r') as src:
                    lat_vars = [v for v in src.variables if getattr(src.variables[v], 'standard_name', '') == 'latitude' or v in ['lat', 'latitude']]
                    lon_vars = [v for v in src.variables if getattr(src.variables[v], 'standard_name', '') == 'longitude' or v in ['lon', 'longitude']]
                    
                    if not lat_vars or not lon_vars:
                        return False, "No coordinates found"

                    lat_name, lon_name = lat_vars[0], lon_vars[0]
                    lat = src.variables[lat_name][:]
                    lon = src.variables[lon_name][:]
                    file_format = src.file_format # Save format for writing later
            
            # Normalize longitude to -180 to 180
            lon_norm = np.where(lon > 180, lon - 360, lon)

            min_x, min_y, max_x, max_y = self.shapefile_geom.bounds
            if (np.max(lon_norm) < min_x or np.min(lon_norm) > max_x or 
                np.max(lat) < min_y or np.min(lat) > max_y):
                return False, "Outside bounds"

            # Create 2D mask (Expensive operation - runs in parallel)
            lon2d, lat2d = np.meshgrid(lon_norm, lat)
            mask = contains_func(self.shapefile_geom, lon2d, lat2d)

            if not np.any(mask):
                return False, "No intersection"

            with hdf5_lock:
                if self._stop_event.is_set(): return False, "Interrupted"
                
                # Re-open input to read variables
                with nc.Dataset(input_path, 'r') as src:
                    with nc.Dataset(output_path, 'w', format=file_format) as dst:
                        dst.setncatts(src.__dict__)
                        for name, dim in src.dimensions.items():
                            dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

                        for name, var in src.variables.items():
                            fill_value = var.getncattr('_FillValue') if '_FillValue' in var.ncattrs() else None
                            dst_var = dst.createVariable(name, var.dtype, var.dimensions, fill_value=fill_value)
                            attrs = {k: var.getncattr(k) for k in var.ncattrs() if k != '_FillValue'}
                            dst_var.setncatts(attrs)
                            
                            data = var[:]
                            
                            # Apply mask if variable uses lat/lon dimensions
                            dims_map = {d_name: i for i, d_name in enumerate(var.dimensions)}
                            
                            if lat_name in dims_map and lon_name in dims_map:
                                shape_match = [data.shape[dims_map[lat_name]], data.shape[dims_map[lon_name]]]
                                if mask.shape == tuple(shape_match):
                                    try:
                                        full_mask = np.broadcast_to(mask, data.shape) if mask.ndim < data.ndim else mask
                                        fill_val = np.array(fill_value, dtype=data.dtype) if fill_value is not None else np.nan
                                        data = np.where(full_mask, data, fill_val)
                                    except ValueError:
                                        pass 

                            dst_var[:] = data

            return True, f"Clipped: {output_path.name}"

        except Exception as e:
            if output_path.exists(): output_path.unlink()
            return False, str(e)

    def process_all(self, files: List[Path]) -> Tuple[int, int]:
        """Processes all files in parallel threads."""
        if not files: return 0, 0
        
        successful = 0
        total = len(files)
        workers = self.settings.get('workers', 4)
        
        # Use ThreadPoolExecutor again (Safe now because of the Lock)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='Clipper')
        future_to_file = {self.executor.submit(self.clip_file, p): p for p in files}
        
        is_gui_mode = self.settings.get('is_gui_mode', False)
        use_tqdm = HAS_UI_LIBS and not is_gui_mode
        futures_iter = as_completed(future_to_file)

        if use_tqdm:
            futures_iter = tqdm(futures_iter, total=total, unit="file", desc="Clipping", ncols=90)

        try:
            for i, future in enumerate(futures_iter):
                if self._stop_event.is_set(): 
                    self.executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                if is_gui_mode:
                    logging.info(f"Progress: {i + 1}/{total} files processed.")
                    
                original_path = future_to_file[future]
                try:
                    success, msg = future.result()
                    if success:
                        successful += 1
                        if use_tqdm: tqdm.write(f"  ✔ {msg}")
                        elif is_gui_mode: logging.info(msg)
                    else:
                        if use_tqdm: tqdm.write(f"  ✖ Failed {original_path.name}: {msg}")
                        elif is_gui_mode: logging.warning(f"Failed {original_path.name}: {msg}")
                except Exception as e:
                    logging.error(f"Critical error processing {original_path.name}: {e}")

        finally:
            self.shutdown()
            
        return successful, total

    def shutdown(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        logging.info("Clipper shutdown complete.")

def load_config(config_path: str) -> Dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        return {}

def create_clipping_session(settings: Dict[str, Any], stop_event: threading.Event) -> None:
    is_gui_mode = settings.get('is_gui_mode', False)
    
    try:
        if not all([settings.get('input_dir'), settings.get('output_dir'), settings.get('shapefile')]):
            logging.error("Missing required arguments: input_dir, output_dir, shapefile.")
            if not is_gui_mode: sys.exit(1)
            return

        file_manager = FileManager(settings['input_dir'], settings['output_dir'])
        nc_files = file_manager.get_netcdf_files()
        
        if not nc_files:
            logging.warning(f"No NetCDF files found in {settings['input_dir']}")
            if not is_gui_mode: sys.exit(0)
            return

        # Clean settings to remove duplicate object
        settings.pop('stop_event', None)
        settings.pop('stop_flag', None)

        clipper = Clipper(file_manager, stop_event, **settings)
        
        if HAS_UI_LIBS and not is_gui_mode:
            with Console().status("[bold green]Loading shapefile...", spinner="dots"):
                loaded = clipper.load_shapefile()
        else:
            loaded = clipper.load_shapefile()

        if not loaded:
            if not is_gui_mode: sys.exit(1)
            return

        logging.info(f"Starting clipping for {len(nc_files)} files...")
        success_count, total_count = clipper.process_all(nc_files)
        
        if stop_event.is_set():
            logging.warning("Process was stopped before completion.")
        
        logging.info(f"Completed: {success_count}/{total_count} files processed successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    io_group = parser.add_argument_group('Input and Output')
    settings_group = parser.add_argument_group('Processing Settings')

    io_group.add_argument("-i", "--input_dir", help="Directory containing input NetCDF files.")
    io_group.add_argument("-o", "--output_dir", help="Directory to save clipped files.")
    io_group.add_argument("-s", "--shapefile", help="Path to the shapefile (.shp).")

    settings_group.add_argument("--buffer_km", type=float, default=0, help="Buffer distance in kilometers.")
    settings_group.add_argument("-ld", "--log-dir", default="./gridflow_logs", help="Log directory.")
    settings_group.add_argument("-ll", "--log-level", default="verbose", choices=["minimal", "verbose", "debug"], help="Log verbosity.")
    settings_group.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 4, help="Parallel workers.")
    settings_group.add_argument("--demo", action="store_true", help="Run with demo settings.")
    settings_group.add_argument("-c", "--config", help="Path to JSON config file.")

def main(args=None):
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="NetCDF Clipping Tool", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()

    if not getattr(args, 'is_gui_mode', False):
        setup_logging(args.log_dir, args.log_level, prefix="clip_netcdf")
        signal.signal(signal.SIGINT, signal_handler)

    active_stop_event = getattr(args, 'stop_event', stop_event)

    config = load_config(args.config) if args.config else {}
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    settings = {**config, **cli_args}

    if settings.get('demo'):
        default_shp = Path("./conus_border/conus.shp")
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            default_shp = Path(sys._MEIPASS) / "conus_border" / "conus.shp"

        settings.update({
            'input_dir': settings.get('input_dir') or './downloads_cmip6',
            'output_dir': settings.get('output_dir') or './clipped_cmip6',
            'shapefile': str(default_shp)
        })

        demo_cmd = (
            "gridflow clip "
            f"-i {settings['input_dir']} "
            f"-o {settings['output_dir']} "
            f"-s {settings['shapefile']}"
        )

        if HAS_UI_LIBS and not getattr(args, 'is_gui_mode', False):
            console = Console()
            console.print(f"\n[bold yellow]Running in demo mode.[/]")
            console.print(f"Demo Command:\n  [dim]{demo_cmd}[/dim]\n")
        else:
            logging.info(f"Running in demo mode.\nDemo Command: {demo_cmd}")

    create_clipping_session(settings, active_stop_event)
    
    if active_stop_event.is_set():
        logging.warning("Execution was interrupted.")
        if not getattr(args, 'is_gui_mode', False): sys.exit(130)

    logging.info("Process finished.")
    
if __name__ == "__main__":
    main()