# gridflow/download/era5_downloader.py
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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

# --- Dependency Check ---
try:
    import cdsapi
except ImportError as e:
    print(f"FATAL: Missing required library: {e.name}. Please install it using 'pip install {e.name}'.")
    sys.exit(1)

# --- Local Imports ---
try:
    from gridflow.utils.logging_utils import setup_logging
except ImportError:
    print("FATAL: gridflow/utils/logging_utils.py not found. Please ensure it is in the correct directory.")
    sys.exit(1)

# --- Global Stop Event for Graceful Shutdown ---
stop_event = threading.Event()

def signal_handler(sig, frame):
    """Handles SIGINT (Ctrl+C) to gracefully shut down the application."""
    logging.warning("Stop signal received! Gracefully shutting down...")
    stop_event.set()
    logging.info("Please wait for ongoing tasks to complete.")

# --- Area of Interest Definitions [North, West, South, East] ---
AOI_BOUNDS = {
    "global": [90, -180, -90, 180],
    "north_america": [84, -168, 5, -52],
    "conus": [50, -125, 24, -66],
    "corn_belt": [49.5, -104.5, 35.8, -80.4],
}

# --- ERA5 Variables List ---
ERA5_VARIABLES = [
    "2m_dewpoint_temperature", "2m_temperature", "skin_temperature", "soil_temperature_level_1",
    "soil_temperature_level_2", "soil_temperature_level_3", "soil_temperature_level_4",
    "lake_bottom_temperature", "lake_ice_depth", "lake_ice_temperature", "lake_mix_layer_depth",
    "lake_mix_layer_temperature", "lake_shape_factor", "lake_total_layer_temperature", "snow_albedo",
    "snow_cover", "snow_density", "snow_depth", "snow_depth_water_equivalent", "snowfall",
    "snowmelt", "temperature_of_snow_layer", "skin_reservoir_content", "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2", "volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4",
    "forecast_albedo", "surface_latent_heat_flux", "surface_net_solar_radiation",
    "surface_net_thermal_radiation", "surface_sensible_heat_flux", "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards", "evaporation_from_bare_soil",
    "evaporation_from_open_water_surfaces_excluding_oceans", "evaporation_from_the_top_of_canopy",
    "evaporation_from_vegetation_transpiration", "potential_evaporation", "runoff",
    "snow_evaporation", "sub_surface_runoff", "surface_runoff", "total_evaporation",
    "10m_u_component_of_wind", "10m_v_component_of_wind", "surface_pressure", "total_precipitation",
    "leaf_area_index_high_vegetation", "leaf_area_index_low_vegetation", "high_vegetation_cover",
    "glacier_mask", "lake_cover", "low_vegetation_cover", "lake_total_depth", "geopotential",
    "land_sea_mask", "soil_type", "type_of_high_vegetation", "type_of_low_vegetation"
]


class Downloader:
    """Manages the file download process for ERA5 data."""
    def __init__(self, settings: Dict[str, Any], stop_event: threading.Event):
        self.settings = settings
        self._stop_event = stop_event
        self.cds_client = cdsapi.Client(
            url="https://cds.climate.copernicus.eu/api/v2",
            key=self.settings['api_key']
        )
        self.executor = None

    def shutdown(self):
        """Shuts down the thread pool gracefully."""
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        logging.info("Downloader has been shut down.")

    def download_month(self, year: int, month: int, variable: str, area: List[float]) -> bool:
        """Downloads data for a single month and variable."""
        if self._stop_event.is_set():
            return False

        output_dir = Path(self.settings['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        aoi_name = self.settings.get('aoi', 'custom')
        output_file = output_dir / f"era5_land_{year}_{month:02d}_{aoi_name}_{variable}.nc"

        if output_file.exists():
            logging.info(f"Skipping {output_file.name} - already exists.")
            return True

        request = {
            "variable": [variable],
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "format": "netcdf",
            "area": area
        }
        
        try:
            logging.info(f"Starting download for {year}-{month:02d} ({variable}) to {output_file.name}")
            self.cds_client.retrieve("reanalysis-era5-land", request, output_file)
            logging.info(f"Finished download for {year}-{month:02d} ({variable})")
            return True
        except Exception as e:
            if self._stop_event.is_set():
                logging.warning(f"Download for {year}-{month:02d} ({variable}) interrupted by user.")
            else:
                logging.error(f"Failed to download {year}-{month:02d} ({variable}): {e}", exc_info=True)
            if output_file.exists():
                output_file.unlink() # Clean up failed download
            return False

    def download_all(self, tasks: List[Tuple[int, int, str]], area: List[float]) -> Tuple[int, int]:
        """Manages the download of a list of month/variable tasks."""
        successful_downloads = 0
        if not tasks:
            return 0, 0
        
        self.executor = ThreadPoolExecutor(max_workers=self.settings['workers'], thread_name_prefix='Downloader')
        future_to_task = {
            self.executor.submit(self.download_month, year, month, var, area): (year, month, var)
            for year, month, var in tasks
        }
        
        total_files = len(tasks)
        try:
            for i, future in enumerate(as_completed(future_to_task)):
                logging.info(f"Progress: {i + 1}/{total_files} files processed.")
                if self._stop_event.is_set():
                    break
                if future.result():
                    successful_downloads += 1
                if self._stop_event.is_set():
                    break
        finally:
            self.shutdown()
        
        return successful_downloads, total_files

def generate_tasks(start_date_str: str, end_date_str: str, variables: List[str]) -> List[Tuple[int, int, str]]:
    """Generates a list of (year, month, variable) tasks."""
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    tasks = []
    for var in variables:
        current = start
        while current <= end:
            tasks.append((current.year, current.month, var))
            # Move to the next month
            next_month = current.month % 12 + 1
            next_year = current.year + (current.month // 12)
            current = current.replace(year=next_year, month=next_month, day=1)
    return tasks

def run_download_session(settings: Dict[str, Any], stop_event: threading.Event):
    """Sets up and executes an ERA5 download session."""
    try:
        # Determine the area for the download
        if settings.get('bounds'):
            area = settings['bounds']
            logging.info(f"Using custom bounding box: {area}")
        else:
            area = AOI_BOUNDS[settings['aoi']]
            logging.info(f"Using predefined AOI '{settings['aoi']}': {area}")

        variables = [v.strip() for v in settings['variables'].split(',')]
        tasks = generate_tasks(settings['start_date'], settings['end_date'], variables)
        
        if not tasks:
            logging.info("No tasks to process for the given date range and variables.")
            return

        logging.info(f"Generated {len(tasks)} download tasks.")
        
        downloader = Downloader(settings, stop_event)
        successful, total = downloader.download_all(tasks, area)

        if stop_event.is_set():
            logging.warning("Process was stopped before completion.")
        
        logging.info(f"Completed: {successful}/{total} files downloaded successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred during the session: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    """Add ERA5 downloader arguments to the provided parser."""
    io_group = parser.add_argument_group('Input and Output')
    dl_group = parser.add_argument_group('Download Parameters')
    settings_group = parser.add_argument_group('Processing Settings')

    io_group.add_argument("--output_dir", help="Directory to save downloaded files.", default="./downloads_era5")
    
    dl_group.add_argument("--api_key", help="CDS API key (UID:KEY).", required=True)
    dl_group.add_argument("--start_date", default="2020-01-01", help="Start date (YYYY-MM-DD).")
    dl_group.add_argument("--end_date", default="2020-01-31", help="End date (YYYY-MM-DD).")
    dl_group.add_argument("--variables", default="2m_temperature,total_precipitation", help="Comma-separated list of variables.")
    
    # Mutually exclusive group for AOI or custom bounds
    area_group = dl_group.add_mutually_exclusive_group()
    area_group.add_argument("--aoi", default="corn_belt", choices=list(AOI_BOUNDS.keys()), help="Predefined Area of Interest.")
    area_group.add_argument("--bounds", nargs=4, type=float, metavar=('N', 'W', 'S', 'E'), help="Custom bounding box: North West South East.")
    
    settings_group.add_argument("--log-dir", default="./gridflow_logs", help="Directory to save log files.")
    settings_group.add_argument("--log-level", default="verbose", choices=["minimal", "verbose", "debug"], help="Set logging verbosity.")
    settings_group.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    settings_group.add_argument("--demo", action="store_true", help="Run with demo defaults.")

def main(args=None):
    """Main entry point for the ERA5 Downloader CLI."""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="ERA5-Land Data Downloader", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()
    
    setup_logging(args.log_dir, args.log_level, prefix="era5_downloader")
    signal.signal(signal.SIGINT, signal_handler)
    
    settings = vars(args)
    if args.demo:
        logging.info("Running in demo mode.")
        settings['start_date'] = '2023-01-01'
        settings['end_date'] = '2023-01-02'
        settings['variables'] = '2m_temperature'
        settings['aoi'] = 'corn_belt'
        settings['bounds'] = None # Ensure bounds is not used in demo
        if not settings.get('api_key'):
            logging.error("API key is required for demo mode. Please provide with --api_key.")
            sys.exit(1)
        logging.info(f"Demo will download '{settings['variables']}' for {settings['start_date']} to {settings['end_date']}.")
    
    if not settings.get('api_key'):
        logging.error("An API key is required. Please provide it with the --api_key argument.")
        sys.exit(1)

    run_download_session(settings, stop_event)
    
    if stop_event.is_set():
        logging.warning("Execution was interrupted.")
        sys.exit(130)
    
    logging.info("Process finished.")

if __name__ == "__main__":
    main()
