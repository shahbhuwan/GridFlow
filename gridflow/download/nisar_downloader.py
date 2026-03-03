# gridflow/download/nisar_downloader.py
# Copyright (c) 2025 Bhuwan Shah
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import sys
import json
import signal
import logging
import requests
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
    from rich.console import Console
    HAS_UI_LIBS = True
except ImportError:
    HAS_UI_LIBS = False

try:
    from gridflow.utils.logging_utils import setup_logging
except ImportError:
    print("FATAL: gridflow/utils/logging_utils.py not found.")
    sys.exit(1)

# Constants
ASF_API_URL = "https://api.daac.asf.alaska.edu/services/search/param"
stop_event = threading.Event()

def signal_handler(sig, frame):
    logging.warning("Stop signal received! Gracefully shutting down...")
    stop_event.set()

class QueryHandler:
    def __init__(self, stop_event: threading.Event):
        self._stop_event = stop_event
        
    def _format_date(self, date_str: str) -> str:
        from dateutil import parser as date_parser
        try:
            dt = date_parser.parse(date_str)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            logging.error(f"Failed to parse date '{date_str}': {e}")
            return date_str
    
    def fetch_datasets(self, params: Dict[str, Any], is_gui_mode: bool = False) -> List[Dict]:
        """Queries the ASF DAAC API for NISAR data."""
        use_rich = HAS_UI_LIBS and not is_gui_mode
        status_context = None

        if use_rich:
            console = Console()
            status_context = console.status("[bold green]Querying ASF DAAC for NISAR...", spinner="dots")
            status_context.start()
        else:
            logging.info("Querying ASF DAAC for NISAR datasets...")

        try:
            # Construct API parameters
            api_params = {
                "platform": "NISAR",
                "output": "geojson"
            }
            if params.get('bounds'):
                # API expects point, polygon, or bbox
                # bounds is [north, south, east, west] (if we use that standard)
                b = params['bounds']
                api_params['bbox'] = f"{b['west']},{b['south']},{b['east']},{b['north']}"
            
            if params.get('start_date'):
                api_params['start'] = self._format_date(params['start_date'])
            if params.get('end_date'):
                api_params['end'] = self._format_date(params['end_date'])
            if params.get('flight_direction'):
                api_params['flightDirection'] = params['flight_direction']

            logging.debug(f"Query URL: {ASF_API_URL}")
            logging.debug(f"Query Params: {api_params}")

            # Implement robust retry strategy for 502/504 errors
            session = requests.Session()
            retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))

            logging.info("Sending request to NASA ASF servers (may take up to 30s)...")
            response = session.get(ASF_API_URL, params=api_params, timeout=45)
            response.raise_for_status()
            
            data = response.json()
            features = data.get("features", [])
            
            results = []
            for f in features:
                props = f.get("properties", {})
                url = props.get("url")
                filename = props.get("fileName", url.split('/')[-1] if url else "unknown.h5")
                if url:
                    results.append({
                        "title": filename,
                        "url": url,
                        "source": "ASF_DAAC",
                        "size_bytes": props.get("bytes", 0)
                    })

            if status_context: status_context.stop()
            
            if use_rich:
                console.print(f"[bold blue]Query complete![/] Found {len(results)} datasets.")
            else:
                logging.info(f"Query complete. Found {len(results)} datasets.")
                
            return results

        except Exception as e:
            if status_context: status_context.stop()
            logging.error(f"Error querying ASF API: {e}")
            return []

class FileManager:
    def __init__(self, download_dir: str, metadata_dir: str, metadata_prefix: str = ""):
        self.download_dir = Path(download_dir)
        self.metadata_dir = Path(metadata_dir)
        self.metadata_prefix = metadata_prefix
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def get_output_path(self, file_info: Dict) -> Path:
        return self.download_dir / file_info['title']

    def save_metadata(self, files: List[Dict], filename: str) -> None:
        metadata_path = self.metadata_dir / f"{self.metadata_prefix}{filename}"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(files, f, indent=2)

class Downloader:
    def __init__(self, file_manager: FileManager, stop_event: threading.Event, **settings):
        self.file_manager = file_manager
        self._stop_event = stop_event
        self.settings = settings
        self.successful_downloads = 0

    def download_file(self, file_info: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        if self._stop_event.is_set(): return None, file_info
        
        output_path = self.file_manager.get_output_path(file_info)
        url = file_info['url']
        
        if output_path.exists():
            logging.debug(f"Skipping {output_path.name} - already exists.")
            return str(output_path), None
            
        try:
            logging.debug(f"Downloading {url}")
            # Simplified downloading
            with open(output_path, 'wb') as f:
                with requests.get(url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=8192):
                        if self._stop_event.is_set():
                            raise Exception("Interrupted")
                        f.write(chunk)
            return str(output_path), None
        except Exception as e:
            logging.warning(f"Download failed for {file_info['title']}: {e}")
            if output_path.exists(): output_path.unlink()
            return None, file_info

    def download_all(self, files: List[Dict]) -> Tuple[List[str], List[Dict]]:
        downloaded, failed = [], []
        if not files: return [], []
        
        is_gui_mode = self.settings.get('is_gui_mode', False)
        use_tqdm = HAS_UI_LIBS and not is_gui_mode
        
        max_workers = self.settings.get('workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.download_file, f): f for f in files}
            
            futures_iter = as_completed(future_to_file)
            if use_tqdm:
                futures_iter = tqdm(futures_iter, total=len(files), desc="Downloading", bar_format='  {l_bar}{bar}{r_bar}')

            for i, future in enumerate(futures_iter):
                if self._stop_event.is_set():
                    break
                    
                if is_gui_mode:
                    logging.info(f"Progress: {i + 1}/{len(files)} files processed.")
                    
                info = future_to_file[future]
                try:
                    path, f_info = future.result()
                    if path:
                        downloaded.append(path)
                        if use_tqdm: tqdm.write(f"  ✔ Downloaded {Path(path).name}")
                        elif is_gui_mode: logging.info(f"Downloaded {Path(path).name}")
                    if f_info:
                        failed.append(info)
                except Exception:
                    failed.append(info)
                    
        self.successful_downloads = len(downloaded)
        return downloaded, failed

def create_download_session(settings: Dict[str, Any], stop_event: threading.Event) -> None:
    is_gui_mode = settings.get('is_gui_mode', False)
    
    try:
        # Validate Bounds if presented
        if settings.get('bounds') and isinstance(settings['bounds'], list):
            b = settings['bounds']
            settings['bounds'] = {"north": b[0], "south": b[1], "east": b[2], "west": b[3]}
            
        handler = QueryHandler(stop_event)
        files = handler.fetch_datasets(settings, is_gui_mode=is_gui_mode)
        
        if not files:
            logging.info("No NISAR datasets found.")
            if not is_gui_mode: sys.exit(0)
            return
            
        if settings.get('max_downloads'):
            files = files[:settings['max_downloads']]
            
        manager = FileManager(settings['output_dir'], settings['metadata_dir'], "gridflow_nisar_")
        manager.save_metadata(files, "query_results.json")
        
        if settings.get('dry_run'):
            logging.info(f"Dry run: Would attempt to download {len(files)} files.")
            return
            
        # Remove stop_event from settings if present to prevent multiple values exception from **settings unpacking
        settings.pop('stop_event', None)
        settings.pop('stop_flag', None)    
            
        downloader = Downloader(manager, stop_event, **settings)
        downloader.download_all(files)
        
        logging.info(f"Completed: {downloader.successful_downloads}/{len(files)} files downloaded successfully.")
        
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    query_group = parser.add_argument_group('Query Parameters')
    settings_group = parser.add_argument_group('Download Settings')

    query_group.add_argument("--bounds", type=float, nargs=4, metavar=('NORTH', 'SOUTH', 'EAST', 'WEST'), help="Bounding box coordinates in decimal degrees.")
    query_group.add_argument("--start-date", help="Start date (e.g., '2024-01-01', 'Jan 1 2024')")
    query_group.add_argument("--end-date", help="End date (e.g., '2024-12-31')")
    query_group.add_argument("--flight-direction", choices=['ASCENDING', 'DESCENDING'], help="Orbit direction")

    settings_group.add_argument("-o", "--output-dir", default="./downloads_nisar", help="Output directory.")
    settings_group.add_argument("-md", "--metadata-dir", default="./metadata_nisar", help="Metadata directory.")
    settings_group.add_argument("-ld", "--log-dir", default="./gridflow_logs", help="Log directory.")
    settings_group.add_argument("-ll", "--log-level", default="minimal", choices=["minimal", "verbose", "debug"], help="Log verbosity.")
    settings_group.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel workers.")
    settings_group.add_argument("--max-downloads", type=int, help="Maximum number of files to download.")
    settings_group.add_argument("--dry-run", action="store_true", help="Find files but do not download.")
    settings_group.add_argument("--demo", action="store_true", help="Run with demo settings.")

def main(args=None):
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="NISAR Downloader")
        add_arguments(parser)
        args = parser.parse_args()

    if not getattr(args, 'is_gui_mode', False):
        setup_logging(args.log_dir, args.log_level, prefix="nisar_downloader")
        signal.signal(signal.SIGINT, signal_handler)

    active_stop_event = getattr(args, 'stop_event', stop_event)
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    
    if cli_args.get('demo'):
        cli_args['bounds'] = [42.0, 41.0, -91.0, -92.0]
        cli_args['start_date'] = "2024-01-01T00:00:00Z"
        cli_args['max_downloads'] = 1
        if HAS_UI_LIBS and not getattr(args, 'is_gui_mode', False):
            Console().print("[bold yellow]Running in Demo Mode for NISAR![/]")
    
    create_download_session(cli_args, active_stop_event)
    
if __name__ == "__main__":
    main()
