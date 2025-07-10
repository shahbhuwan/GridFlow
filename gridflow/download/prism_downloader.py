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
import json
import signal
import logging
import requests
import threading
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import List, Dict, Optional, Tuple, Any

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
    logging.info("Please wait for ongoing tasks to complete or timeout.")

class FileManager:
    """Handles file and directory management for PRISM data."""
    def __init__(self, download_dir: str, metadata_dir: str, metadata_prefix: str = ""):
        self.download_dir = Path(download_dir)
        self.metadata_dir = Path(metadata_dir)
        self.metadata_prefix = metadata_prefix
        try:
            self.download_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create directories: {e}")
            sys.exit(1)

    def get_output_path(self, variable: str, resolution: str, filename: str) -> Path:
        """Constructs a logical output path based on PRISM parameters."""
        subdir = self.download_dir / variable / resolution
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / filename

    def save_metadata(self, files: List[Dict], filename: str) -> None:
        metadata_path = self.metadata_dir / f"{self.metadata_prefix}{filename}"
        serializable_files = [{k: str(v) if isinstance(v, Path) else v for k, v in f.items()} for f in files]
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_files, f, indent=2)
            logging.debug(f"Saved metadata to {metadata_path}")
        except IOError as e:
            logging.error(f"Failed to save metadata to {metadata_path}: {e}")

class AvailabilityChecker:
    """Checks for file availability on the PRISM server in parallel."""
    def __init__(self, stop_event: threading.Event, workers: int, timeout: int):
        self._stop_event = stop_event
        self.workers = workers
        self.timeout = timeout

    def check_url(self, url: str) -> bool:
        """Checks if a single URL is available using a HEAD request."""
        if self._stop_event.is_set(): return False
        try:
            response = requests.head(url, timeout=self.timeout, allow_redirects=True)
            if response.status_code == 200:
                logging.debug(f"Data available at {url}")
                return True
            else:
                logging.debug(f"Data unavailable at {url} (HTTP {response.status_code})")
                return False
        except requests.RequestException:
            return False

    def find_available_files(self, potential_files: List[Dict]) -> List[Dict]:
        """Filters a list of potential files to find ones that actually exist on the server."""
        available_files = []
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="Checker") as executor:
            future_to_file = {executor.submit(self.check_url, f['url']): f for f in potential_files}
            for future in as_completed(future_to_file):
                if self._stop_event.is_set(): break
                if future.result():
                    available_files.append(future_to_file[future])
        return available_files

class Downloader:
    """Manages the file download process using a thread pool."""
    def __init__(self, stop_event: threading.Event, workers: int, timeout: int):
        self._stop_event = stop_event
        self.workers = workers
        self.timeout = timeout
        self.successful_downloads = 0
        self.executor = None

    def shutdown(self):
        """Shuts down the thread pool gracefully."""
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        logging.info("Downloader has been shut down.")

    def download_file(self, file_info: Dict) -> Optional[Dict]:
        """Downloads a single file."""
        url = file_info['url']
        output_path = file_info['output_path']
        thread_id = threading.get_ident()

        if self._stop_event.is_set(): return None
        
        try:
            logging.debug(f"[Worker-{thread_id}] Starting download of {output_path.name}")
            response = requests.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self._stop_event.is_set():
                        raise requests.exceptions.RequestException("Download interrupted by user.")
                    f.write(chunk)
            
            logging.info(f"Downloaded {output_path.name}")
            return file_info
        except requests.RequestException as e:
            logging.warning(f"[Worker-{thread_id}] Download failed for {output_path.name}: {e}")
            if output_path.exists(): output_path.unlink(missing_ok=True)
            return None

    def download_all(self, files: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Manages the download of a list of files using a thread pool."""
        downloaded, failed = [], []
        if not files: return [], []

        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='Downloader')
        future_to_file = {self.executor.submit(self.download_file, f): f for f in files}

        try:
            for i, future in enumerate(as_completed(future_to_file)):
                logging.info(f"Progress: {i + 1}/{len(files)} files processed.")
                if self._stop_event.is_set(): break
                original_file = future_to_file[future]
                result = future.result()
                if result:
                    downloaded.append(result)
                else:
                    failed.append(original_file)
        finally:
            self.shutdown()
        
        self.successful_downloads = len(downloaded)
        return downloaded, failed

def create_download_session(settings: Dict[str, Any], stop_event: threading.Event) -> None:
    """Sets up and executes a PRISM download session."""
    try:
        variable, resolution, time_step = settings['variable'], settings['resolution'], settings['time_step']
        start_dt = datetime.strptime(settings['start_date'], '%Y-%m-%d')
        end_dt = datetime.strptime(settings['end_date'], '%Y-%m-%d')

        potential_files = []
        current_dt = start_dt
        file_manager = FileManager(settings['output_dir'], settings['metadata_dir'], "gridflow_prism_")
        
        base_url = "https://data.prism.oregonstate.edu"
        
        while current_dt <= end_dt:
            if stop_event.is_set(): break
            year_str = current_dt.strftime('%Y')
            date_str = current_dt.strftime('%Y%m%d')
            
            # --- Correctly format the PRISM filename and URL ---
            resolution_label = '4kmD2' if resolution == '4km' else '800mD2'
            filename = f"PRISM_{variable}_stable_{resolution_label}_{date_str}_bil.zip"
            url = f"{base_url}/{time_step}/{variable}/{year_str}/{filename}"
            
            file_info = {
                'url': url,
                'output_path': file_manager.get_output_path(variable, resolution, filename),
                'date': current_dt.strftime('%Y-%m-%d')
            }
            potential_files.append(file_info)
            current_dt += relativedelta(days=1)

        logging.info(f"Generated {len(potential_files)} potential file URLs for the specified date range.")
        if stop_event.is_set(): return

        checker = AvailabilityChecker(stop_event, settings['workers'], settings['timeout'])
        available_files = checker.find_available_files(potential_files)
        
        logging.info(f"Found {len(available_files)} available files on the server.")
        if not available_files: sys.exit(0)
        if stop_event.is_set(): return

        files_to_download = [f for f in available_files if not f['output_path'].exists()]
        existing_file_count = len(available_files) - len(files_to_download)
        if existing_file_count > 0:
            logging.info(f"{existing_file_count} files already exist and will be skipped.")
        
        file_manager.save_metadata(available_files, "query_results.json")

        if settings.get('dry_run'):
            logging.info(f"Dry run: Would attempt to download {len(files_to_download)} files.")
            return

        downloader = Downloader(stop_event, settings['workers'], settings['timeout'])
        downloaded, failed = downloader.download_all(files_to_download)

        if stop_event.is_set(): logging.warning("Process was stopped before completion.")
        if failed:
            file_manager.save_metadata(failed, "failed_downloads.json")
            logging.warning(f"{len(failed)} downloads failed. Check 'failed_downloads.json' for details.")
        
        total_processed = downloader.successful_downloads + existing_file_count
        logging.info(f"Completed: {total_processed}/{len(available_files)} files processed successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred during the session: {e}", exc_info=True)
        stop_event.set()


def add_arguments(parser):
    """Add PRISM downloader arguments to the provided parser."""
    query_group = parser.add_argument_group('Query Parameters')
    settings_group = parser.add_argument_group('Download & Output Settings')

    query_group.add_argument("--variable", choices=['ppt', 'tmin', 'tmax', 'tmean', 'tdmean', 'vpdmin', 'vpdmax'], help="Variable to download (required if not in demo mode).")
    query_group.add_argument("--resolution", default='4km', choices=['4km', '800m'], help="Data resolution.")
    query_group.add_argument("--time_step", default='daily', choices=['daily'], help="Time step of the data.")
    query_group.add_argument("--start_date", help="Start date in YYYY-MM-DD format (required if not in demo mode).")
    query_group.add_argument("--end_date", help="End date in YYYY-MM-DD format (required if not in demo mode).")

    settings_group.add_argument("--output-dir", default="./downloads_prism", help="Directory to save downloaded files.")
    settings_group.add_argument("--metadata-dir", default="./metadata_prism", help="Directory to save metadata files.")
    settings_group.add_argument("--log-dir", default="./gridflow_logs", help="Directory to save log files.")
    settings_group.add_argument("--log-level", default="verbose", choices=["minimal", "verbose", "debug"], help="Set logging verbosity for the console.")
    settings_group.add_argument("--workers", type=int, default=8, help="Number of parallel workers for checking and downloading.")
    settings_group.add_argument("--timeout", type=int, default=30, help="Network request timeout in seconds.")
    settings_group.add_argument("--dry-run", action="store_true", help="Find available files but do not download them.")
    settings_group.add_argument("--demo", action="store_true", help="Run with a small, pre-defined demo query for PRISM.")

def main(args=None):
    """Main entry point for the PRISM Downloader."""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="PRISM Data Downloader", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()

    setup_logging(args.log_dir, args.log_level, prefix="prism_downloader")
    signal.signal(signal.SIGINT, signal_handler)

    if args.demo:
        settings = {
            **vars(args),
            'variable': 'tmean', 'resolution': '4km', 'time_step': 'daily',
            'start_date': '2020-01-01', 'end_date': '2020-01-05'
        }
        logging.info("Running in demo mode with a pre-defined PRISM query.")
    else:
        if not all([args.variable, args.start_date, args.end_date]):
            logging.error("the following arguments are required when not in --demo mode: --variable, --start_date, --end_date")
            sys.exit(1)
        settings = vars(args)

    try:
        datetime.strptime(settings['start_date'], '%Y-%m-%d')
        datetime.strptime(settings['end_date'], '%Y-%m-%d')
    except (ValueError, TypeError):
        logging.error(f"Invalid date format. Please use YYYY-MM-DD.")
        sys.exit(1)

    create_download_session(settings, stop_event)
    
    if stop_event.is_set():
        logging.warning("Execution was interrupted.")
        sys.exit(130)
    
    logging.info("Process finished.")

if __name__ == "__main__":
    main()