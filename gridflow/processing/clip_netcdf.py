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
from typing import List, Dict, Tuple, Any

# --- Dependency Check ---
try:
    import netCDF4 as nc
    import numpy as np
    import geopandas as gpd
    from shapely.prepared import prep
    
    # --- THIS IS THE FIX ---
    # Handle shapely version differences for the vectorized 'contains' function.
    # The call signature for both versions will be `contains_func(geom, x, y)`.
    try:
        # For Shapely >= 2.0
        from shapely import contains_xy
        contains_func = contains_xy
    except ImportError:
        # For older Shapely versions
        from shapely.vectorized import contains as contains_func

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

def clip_single_file(
    input_path: Path,
    output_path: Path,
    clip_geom, # The unioned geometry from the shapefile
    stop_event: threading.Event
) -> bool:
    """Clips a single NetCDF file using a shapely geometry."""
    if stop_event.is_set(): return False
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with nc.Dataset(input_path, 'r') as src:
            lat_vars = [v for v in src.variables if getattr(src.variables[v], 'standard_name', '') == 'latitude' or v in ['lat', 'latitude']]
            lon_vars = [v for v in src.variables if getattr(src.variables[v], 'standard_name', '') == 'longitude' or v in ['lon', 'longitude']]
            
            if not lat_vars or not lon_vars:
                logging.error(f"Could not find coordinate variables in {input_path.name}.")
                return False

            lat_var_name, lon_var_name = lat_vars[0], lon_vars[0]
            lat = src.variables[lat_var_name][:]
            lon = src.variables[lon_var_name][:]

            lon_norm = np.where(lon > 180, lon - 360, lon)
            
            min_lon_geom, min_lat_geom, max_lon_geom, max_lat_geom = clip_geom.bounds
            logging.debug(f"Shapefile bounds (EPSG:4326): Lon({min_lon_geom:.2f}, {max_lon_geom:.2f}), Lat({min_lat_geom:.2f}, {max_lat_geom:.2f})")
            logging.debug(f"NetCDF grid bounds: Lon({np.min(lon_norm):.2f}, {np.max(lon_norm):.2f}), Lat({np.min(lat):.2f}, {np.max(lat):.2f})")

            lon2d, lat2d = np.meshgrid(lon_norm, lat)
            
            # --- THIS IS THE FIX ---
            # Call the correctly imported function with the standard (geom, x, y) signature.
            # The `prep()` function is also no longer needed for performance in modern shapely.
            mask = contains_func(clip_geom, lon2d, lat2d)

            if not np.any(mask):
                logging.warning(f"No grid points from {input_path.name} intersect with the shapefile. Skipping.")
                return False

            with nc.Dataset(output_path, 'w', format=src.file_format) as dst:
                dst.setncatts(src.__dict__)
                for name, dim in src.dimensions.items():
                    dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

                for name, var in src.variables.items():
                    if stop_event.is_set(): return False
                    fill_value = var.getncattr('_FillValue') if '_FillValue' in var.ncattrs() else None
                    dst_var = dst.createVariable(name, var.dtype, var.dimensions, fill_value=fill_value)
                    attrs_to_copy = {k: var.getncattr(k) for k in var.ncattrs() if k != '_FillValue'}
                    dst_var.setncatts(attrs_to_copy)
                    
                    data = var[:]
                    if lat_var_name in var.dimensions and lon_var_name in var.dimensions:
                        shape_to_match = [d for d_name, d in zip(var.dimensions, data.shape) if d_name in [lat_var_name, lon_var_name]]
                        if mask.shape == tuple(shape_to_match):
                            mask_expanded = np.broadcast_to(mask, data.shape)
                            fill_val_typed = np.array(fill_value, dtype=data.dtype) if fill_value is not None else np.nan
                            data = np.where(mask_expanded, data, fill_val_typed)
                    
                    dst_var[:] = data
                        
        logging.info(f"Successfully clipped {input_path.name} to {output_path.name}")
        return True
    except Exception as e:
        logging.error(f"Failed to clip {input_path.name}: {e}", exc_info=False)
        if output_path.exists(): output_path.unlink(missing_ok=True)
        return False

class Clipper:
    """Manages the parallel clipping of NetCDF files."""
    def __init__(self, settings: Dict[str, Any], stop_event: threading.Event):
        self.settings = settings
        self._stop_event = stop_event
        self.executor = None

    def shutdown(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        logging.info("Clipper has been shut down.")

    def clip_all(self, files_to_clip: List[Tuple[Path, Path]], clip_geom) -> Tuple[int, int]:
        """Manages the parallel clipping of a list of files."""
        successful_clips = 0
        if not files_to_clip: return 0, 0
        
        self.executor = ThreadPoolExecutor(max_workers=self.settings['workers'], thread_name_prefix='Clipper')
        future_to_file = {
            self.executor.submit(clip_single_file, in_path, out_path, clip_geom, self._stop_event): in_path
            for in_path, out_path in files_to_clip
        }
        
        total_files = len(files_to_clip)
        try:
            for i, future in enumerate(as_completed(future_to_file)):
                logging.info(f"Progress: {i + 1}/{total_files} files processed.")
                if self._stop_event.is_set(): break
                if future.result():
                    successful_clips += 1
        finally:
            self.shutdown()
        
        return successful_clips, total_files

def run_clip_session(settings: Dict[str, Any], stop_event: threading.Event):
    """Sets up and executes a clipping session."""
    try:
        input_dir = Path(settings['input_dir'])
        output_dir = Path(settings['output_dir'])
        shapefile_path = Path(settings['shapefile'])
        buffer_km = settings['buffer_km']

        if not input_dir.is_dir():
            logging.error(f"Input directory not found: {input_dir}")
            sys.exit(1)
        if not shapefile_path.is_file():
            logging.error(f"Shapefile not found: {shapefile_path}")
            sys.exit(1)
        
        logging.info(f"Loading and preparing shapefile: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)

        logging.info(f"Original shapefile CRS: {gdf.crs}. Standardizing to EPSG:4326.")
        gdf = gdf.to_crs("EPSG:4326")

        if buffer_km > 0:
            logging.info(f"Applying {buffer_km}km buffer to shapefile...")
            gdf_meters = gdf.to_crs("EPSG:6933") 
            gdf_meters['geometry'] = gdf_meters.buffer(buffer_km * 1000)
            gdf = gdf_meters.to_crs("EPSG:4326")
        
        union_geom = gdf.unary_union
        
        nc_files = list(input_dir.rglob("*.nc"))
        if not nc_files:
            logging.warning(f"No NetCDF (.nc) files found in {input_dir} or its subdirectories.")
            sys.exit(0)
            
        logging.info(f"Found {len(nc_files)} NetCDF files to process.")
        
        tasks = [(p, output_dir / f"{p.stem}_clipped.nc") for p in nc_files]
        
        clipper = Clipper(settings, stop_event)
        successful_clips, total_files = clipper.clip_all(tasks, union_geom)

        if stop_event.is_set():
            logging.warning("Process was stopped before completion.")
        
        logging.info(f"Completed: {successful_clips}/{total_files} files clipped successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred during the session: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    """Add NetCDF clipping arguments to the provided parser."""
    io_group = parser.add_argument_group('Input and Output')
    settings_group = parser.add_argument_group('Processing Settings')

    io_group.add_argument("--input_dir", help="Directory containing input NetCDF files (required if not in demo mode).")
    io_group.add_argument("--output_dir", help="Directory to save clipped files (required if not in demo mode).")
    io_group.add_argument("--shapefile", help="Path to the shapefile for clipping (required if not in demo mode).")

    settings_group.add_argument("--buffer_km", type=float, default=0, help="Buffer distance in kilometers to apply to the shapefile.")
    settings_group.add_argument("--log-dir", default="./gridflow_logs", help="Directory to save log files.")
    settings_group.add_argument("--log-level", default="verbose", choices=["minimal", "verbose", "debug"], help="Set logging verbosity for the console.")
    settings_group.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of parallel workers.")
    settings_group.add_argument("--demo", action="store_true", help="Run with a pre-defined demo using default folders and a built-in shapefile.")

def main(args=None):
    """Main entry point for the NetCDF Clipping Tool."""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="NetCDF Clipping Tool", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()
    
    setup_logging(args.log_dir, args.log_level, prefix="clip_netcdf")
    signal.signal(signal.SIGINT, signal_handler)
    
    settings = vars(args)
    if args.demo:
        logging.info("Running in demo mode.")
        default_shapefile = Path("./iowa_border/iowa_border.shp")
        if getattr(sys.stderr, 'frozen', False) and hasattr(sys.stderr, '_MEIPASS'):
            default_shapefile = Path(sys.stderr._MEIPASS) / "iowa_border" / "iowa_border.shp"

        settings['input_dir'] = settings.get('input_dir') or './downloads_cmip6'
        settings['output_dir'] = settings.get('output_dir') or './clipped_cmip6'
        settings['shapefile'] = settings.get('shapefile') or str(default_shapefile)
        logging.info(f"Demo will use input '{settings['input_dir']}', output to '{settings['output_dir']}', and clip with '{settings['shapefile']}'.")
    else:
        if not all([args.input_dir, args.output_dir, args.shapefile]):
            logging.error("all --input_dir, --output_dir, and --shapefile arguments are required when not in --demo mode.")
            sys.exit(1)

    run_clip_session(settings, stop_event)
    
    if stop_event.is_set():
        logging.warning("Execution was interrupted.")
        sys.exit(130)
    
    logging.info("Process finished.")

if __name__ == "__main__":
    main()
