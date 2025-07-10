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

import sys
import signal
import logging
import threading
import requests
from pathlib import Path
from typing import Dict, Any

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

class DEMDownloader:
    """
    Handles the download of Digital Elevation Models from the OpenTopography API.
    """
    API_URL = "https://portal.opentopography.org/API/globaldem"

    def __init__(self, settings: Dict[str, Any], stop_event: threading.Event):
        self.settings = settings
        self._stop_event = stop_event

    def download(self) -> bool:
        """
        Constructs the API request and downloads the requested DEM.
        """
        if self._stop_event.is_set():
            return False

        dem_type = self.settings['dem_type']
        bounds = self.settings['bounds']
        output_file = Path(self.settings['output_file'])
        
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # --- FIX: Changed parameter name from 'key' to 'API_Key' ---
        params = {
            'demtype': dem_type,
            'south': bounds['south'],
            'north': bounds['north'],
            'west': bounds['west'],
            'east': bounds['east'],
            'outputFormat': 'GTiff',
            'API_Key': self.settings['api_key'] # Corrected parameter name
        }

        logging.info(f"Requesting DEM '{dem_type}' for bounds N:{params['north']} S:{params['south']} E:{params['east']} W:{params['west']}")
        logging.debug(f"Request URL: {self.API_URL} with params: {params}")

        try:
            response = requests.get(self.API_URL, params=params, stream=True, timeout=300)
            response.raise_for_status()

            logging.info(f"Downloading data to {output_file}...")
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self._stop_event.is_set():
                        logging.warning("Download interrupted by user.")
                        if output_file.exists():
                            output_file.unlink()
                        return False
                    f.write(chunk)
            
            logging.info(f"Successfully downloaded DEM to {output_file}")
            return True

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logging.error("HTTP Error 401: Unauthorized. Your API key is likely invalid or missing. Please get a free key from portal.opentopography.org.")
            elif e.response.status_code == 400:
                logging.error(f"Bad Request: The server rejected the request. This may be due to invalid bounds. Server message: {e.response.text}")
            else:
                logging.error(f"HTTP Error while downloading DEM: {e}")
            return False
        except requests.exceptions.RequestException as e:
            logging.error(f"A network error occurred: {e}")
            return False
        except Exception as e:
            logging.critical(f"An unexpected error occurred during download: {e}", exc_info=True)
            return False

def run_dem_download_session(settings: Dict[str, Any], stop_event: threading.Event):
    """Sets up and executes a DEM download session."""
    try:
        downloader = DEMDownloader(settings, stop_event)
        downloader.download()

    except Exception as e:
        logging.critical(f"A critical error occurred during the session: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    """Add DEM downloader arguments to the provided parser."""
    req_group = parser.add_argument_group('Required Parameters')
    opt_group = parser.add_argument_group('Optional Parameters')
    settings_group = parser.add_argument_group('Processing Settings')

    req_group.add_argument("--api_key", help="Your personal API key from opentopography.org (required if not using demo with a pre-set key).")
    req_group.add_argument("--bounds", type=float, nargs=4, metavar=('NORTH', 'SOUTH', 'EAST', 'WEST'), help="Bounding box coordinates in decimal degrees.")
    req_group.add_argument("--output_file", help="Path to save the output GeoTIFF file.")
    
    opt_group.add_argument("--dem_type", default="COP30", choices=['COP30', 'SRTMGL1', 'SRTMGL3', 'AW3D30'], help="Type of DEM to download.")

    settings_group.add_argument("--log-dir", default="./gridflow_logs", help="Directory to save log files.")
    settings_group.add_argument("--log-level", default="verbose", choices=["minimal", "verbose", "debug"], help="Set logging verbosity.")
    settings_group.add_argument("--demo", action="store_true", help="Run with a pre-defined demo query for Iowa (requires API key).")

def main(args=None):
    """Main entry point for the DEM Downloader."""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="Digital Elevation Model (DEM) Downloader via OpenTopography", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()
    
    setup_logging(args.log_dir, args.log_level, prefix="dem_downloader")
    signal.signal(signal.SIGINT, signal_handler)
    
    settings = vars(args)
    if args.demo:
        if not args.api_key:
            logging.error("--api_key is required even for --demo mode.")
            sys.exit(1)
        logging.info("Running in demo mode.")
        settings['bounds'] = {'north': 43.5, 'south': 40.3, 'east': -90.1, 'west': -96.7}
        settings['output_file'] = './downloads_dem/dem_iowa_COP30.tif'
        settings['dem_type'] = 'COP30'
        logging.info(f"Demo will download '{settings['dem_type']}' for Iowa to '{settings['output_file']}'.")
    else:
        if not all([args.api_key, args.bounds, args.output_file]):
            logging.error("--api_key, --bounds, and --output_file are required when not in --demo mode.")
            sys.exit(1)
        b = args.bounds
        if b[0] <= b[1]:
            logging.error("Invalid bounds: North coordinate must be greater than South coordinate.")
            sys.exit(1)
        settings['bounds'] = {'north': b[0], 'south': b[1], 'east': b[2], 'west': b[3]}

    run_dem_download_session(settings, stop_event)
    
    if stop_event.is_set():
        logging.warning("Execution was interrupted.")
        sys.exit(130)
    
    logging.info("Process finished.")
