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

    def find_available_files(self, potential_files: List[Dict], is_gui_mode: bool = False) -> List[Dict]:
        """Filters a list of potential files to find ones that actually exist on the server."""
        available_files = []
        
        use_rich = HAS_UI_LIBS and not is_gui_mode
        status_context = None

        if use_rich:
            console = Console()
            status_context = console.status(f"[bold green]Checking availability for {len(potential_files)} files (PRISM server)...", spinner="dots")
            status_context.start()
        else:
            logging.info(f"Checking availability for {len(potential_files)} files...")

        try:
            with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="Checker") as executor:
                future_to_file = {executor.submit(self.check_url, f['url']): f for f in potential_files}
                
                for future in as_completed(future_to_file):
                    if self._stop_event.is_set(): break
                    if future.result():
                        available_files.append(future_to_file[future])
            
            if status_context: status_context.stop()
            return available_files
            
        finally:
            if status_context: status_context.stop()

class Downloader:
    """Manages the file download process using a thread pool."""
    def __init__(self, stop_event: threading.Event, workers: int, timeout: int, is_gui_mode: bool = False):
        self._stop_event = stop_event
        self.workers = workers
        self.timeout = timeout
        self.is_gui_mode = is_gui_mode
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
            
            logging.debug(f"Downloaded {output_path.name}")
            return file_info
        except requests.RequestException as e:
            logging.warning(f"[Worker-{thread_id}] Download failed for {output_path.name}: {e}")
            if output_path.exists(): output_path.unlink(missing_ok=True)
            return None

    def download_all(self, files: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Manages the download of a list of files using a thread pool."""
        downloaded, failed = [], []
        if not files:
            return [], []

        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='Downloader')

        future_to_file = {}
        for f in files:
            ft = self.executor.submit(self.download_file, f)
            setattr(ft, "_original_file", f)
            future_to_file[ft] = f

        use_tqdm = HAS_UI_LIBS and not self.is_gui_mode
        
        futures_iter = as_completed(future_to_file)
        
        if use_tqdm:
            futures_iter = tqdm(
                futures_iter, 
                total=len(files), 
                unit="file", 
                desc="Downloading", 
                ncols=90, 
                bar_format='  {l_bar}{bar}{r_bar}'
            )

        try:
            for i, future in enumerate(futures_iter):
                if self.is_gui_mode:
                    logging.info(f"Progress: {i + 1}/{len(files)} files processed.")
                
                if self._stop_event.is_set():
                    break

                original_file = future_to_file.get(future, getattr(future, "_original_file", None))
                file_name = original_file['output_path'].name if original_file else "unknown_file"

                try:
                    result = future.result()
                    if result:
                        downloaded.append(result)
                        if use_tqdm:
                            short_name = (file_name[:50] + '..') if len(file_name) > 50 else file_name
                            tqdm.write(f"  ✔ Downloaded {short_name}")
                        elif self.is_gui_mode:
                            logging.info(f"Downloaded {file_name}")
                    else:
                        failed.append(original_file if original_file else {"file": "unknown"})
                        if use_tqdm:
                            tqdm.write(f"  ✖ Failed: {file_name}")
                            
                except Exception as e:
                    failed.append(original_file if original_file else {"file": "unknown"})
                    if use_tqdm:
                        tqdm.write(f"  ✖ Failed: {file_name}")
                    logging.debug(f"Error in future result: {e}")

        except Exception as e:
            print(f"\nCRITICAL ERROR in download loop: {e}", file=sys.stderr)
            raise e
            
        finally:
            self.shutdown()

        self.successful_downloads = len(downloaded)
        return downloaded, failed

def load_config(config_path: str) -> Dict:
    """Loads a JSON configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
        logging.error(f"Failed to load config file {config_path}: {e}")
        sys.exit(1)

def create_download_session(settings: Dict[str, Any], stop_event: threading.Event) -> None:
    """Sets up and executes a PRISM download session handling both DAILY and MONTHLY intervals."""
    is_gui_mode = settings.get('is_gui_mode', False)
    try:
        variables = settings['variable']
        if isinstance(variables, str):
            variables = [variables]
            
        resolution = settings['resolution']
        time_step = settings['time_step']
        
        start_dt = datetime.strptime(settings['start_date'], '%Y-%m-%d')
        end_dt = datetime.strptime(settings['end_date'], '%Y-%m-%d')

        if time_step == 'monthly':
            # Monthly: Format YYYYMM, Step 1 Month
            date_fmt = '%Y%m'
            step_delta = relativedelta(months=1)
        else:
            # Daily: Format YYYYMMDD, Step 1 Day
            date_fmt = '%Y%m%d'
            step_delta = relativedelta(days=1)

        potential_files = []
        current_dt = start_dt
        file_manager = FileManager(settings['output_dir'], settings['metadata_dir'], "gridflow_prism_")
        
        base_url = "https://data.prism.oregonstate.edu/time_series/us/an"
        
        while current_dt <= end_dt:
            if stop_event.is_set(): break
            
            year_str = current_dt.strftime('%Y')
            date_str = current_dt.strftime(date_fmt) 
            
            # 800m folders -> '30s' in filename
            # 4km folders  -> '25m' in filename
            if resolution == '800m':
                res_path = '800m'
                res_file_code = '30s'
            else:
                res_path = '4km'
                res_file_code = '25m'

            for variable in variables:
                filename = f"prism_{variable}_us_{res_file_code}_{date_str}.zip"
                
                # URL Construction
                url = f"{base_url}/{res_path}/{variable}/{time_step}/{year_str}/{filename}"
                
                file_info = {
                    'url': url,
                    'output_path': file_manager.get_output_path(variable, resolution, filename),
                    'date': current_dt.strftime('%Y-%m-%d')
                }
                potential_files.append(file_info)
            
            current_dt += step_delta 

        logging.info(f"Generated {len(potential_files)} potential file URLs.")
        if stop_event.is_set(): return

        workers = settings.get('workers', 8)
        timeout = settings.get('timeout', 30)

        checker = AvailabilityChecker(stop_event, workers, timeout)
        available_files = checker.find_available_files(potential_files, is_gui_mode=is_gui_mode)
        
        if not available_files:
            logging.info("No files were found on the server for the given criteria.")
            if not is_gui_mode: sys.exit(0)
            return
        
        if HAS_UI_LIBS and not is_gui_mode:
            from rich.console import Console
            Console().print(f"[bold blue]Query complete![/] Found {len(available_files)} available files.")
        else:
            logging.info(f"Query complete! Found {len(available_files)} available files.")

        if stop_event.is_set(): return

        files_to_download = [f for f in available_files if not f['output_path'].exists()]
        existing_file_count = len(available_files) - len(files_to_download)
        if existing_file_count > 0:
            logging.info(f"{existing_file_count} files already exist and will be skipped.")
        
        file_manager.save_metadata(available_files, "query_results.json")

        if settings.get('dry_run'):
            logging.info(f"Dry run: Would attempt to download {len(files_to_download)} files.")
            return

        downloader = Downloader(stop_event, workers, timeout, is_gui_mode=is_gui_mode)
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

    query_group.add_argument("-var", "--variable", nargs='+', choices=['ppt', 'tmin', 'tmax', 'tmean', 'tdmean', 'vpdmin', 'vpdmax'], help="Variable(s) to download (required if not in demo mode).")
    query_group.add_argument("-r", "--resolution", default='4km', choices=['4km', '800m'], help="Data resolution.")
    query_group.add_argument("-ts", "--time-step", default='daily', choices=['daily', 'monthly'], help="Time step of the data.")
    query_group.add_argument("-sd", "--start-date", help="Start date in YYYY-MM-DD format (required if not in demo mode).")
    query_group.add_argument("-ed", "--end-date", help="End date in YYYY-MM-DD format (required if not in demo mode).")

    query_group.add_argument("-c", "--config", help="Path to JSON config file to pre-fill arguments.")

    settings_group.add_argument("-o", "--output-dir", default="./downloads_prism", help="Directory to save downloaded files.")
    settings_group.add_argument("-md", "--metadata-dir", default="./metadata_prism", help="Directory to save metadata files.")
    settings_group.add_argument("-ld", "--log-dir", default="./gridflow_logs", help="Directory to save log files.")
    settings_group.add_argument("-ll", "--log-level", default="minimal", choices=["minimal", "verbose", "debug"], help="Set logging verbosity for the console.")
    settings_group.add_argument("-w", "--workers", type=int, default=8, help="Number of parallel workers for checking and downloading.")
    settings_group.add_argument("-t", "--timeout", type=int, default=30, help="Network request timeout in seconds.")
    settings_group.add_argument("--dry-run", action="store_true", help="Find available files but do not download them.")
    settings_group.add_argument("--demo", action="store_true", help="Run with a small, pre-defined demo query for PRISM.")

def main(args=None):
    """Main entry point for the PRISM Downloader."""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="PRISM Data Downloader", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()

    # FIX: Only set up logging if NOT in GUI mode
    if not getattr(args, 'is_gui_mode', False):
        setup_logging(args.log_dir, args.log_level, prefix="prism_downloader")
        signal.signal(signal.SIGINT, signal_handler)

    active_stop_event = getattr(args, 'stop_event', stop_event)

    config = load_config(args.config) if args.config else {}
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    settings = {**config, **cli_args}

    if settings.get('demo'):
        settings.update({
            'variable': ['tmean'], 
            'resolution': '4km', 
            'time_step': 'daily',
            'start_date': '2020-01-01', 
            'end_date': '2020-01-05'
        })

        demo_cmd = (
            "gridflow prism "
            "--variable tmean "
            "--resolution 4km "
            "--start-date 2020-01-01 "
            "--end-date 2020-01-05"
        )

        if HAS_UI_LIBS and not getattr(args, 'is_gui_mode', False):
            console = Console()
            console.print(f"[bold yellow]Running in demo mode with a pre-defined PRISM query.[/]")
            console.print(f"Demo Command:\n  [dim]{demo_cmd}[/dim]\n")
        else:
            logging.info(f"Running in demo mode.\nDemo Command: {demo_cmd}")

    else:
        if not all([settings.get('variable'), settings.get('start_date'), settings.get('end_date')]):
            logging.error("The following arguments are required when not in --demo mode: --variable, --start-date, --end-date")
            if not getattr(args, 'is_gui_mode', False):
                sys.exit(1)
            return

    try:
        if settings.get('time_step') == 'monthly':
             datetime.strptime(settings['start_date'], '%Y-%m-%d')
             datetime.strptime(settings['end_date'], '%Y-%m-%d')
        else:
             datetime.strptime(settings['start_date'], '%Y-%m-%d')
             datetime.strptime(settings['end_date'], '%Y-%m-%d')
    except (ValueError, TypeError):
        logging.error(f"Invalid date format. Please use YYYY-MM-DD.")
        if not getattr(args, 'is_gui_mode', False):
            sys.exit(1)
        return

    create_download_session(settings, active_stop_event)
    
    if active_stop_event.is_set():
        logging.warning("Execution was interrupted.")
        if not getattr(args, 'is_gui_mode', False):
            sys.exit(130)
    
    logging.info("Process finished.")

if __name__ == "__main__":
    main()