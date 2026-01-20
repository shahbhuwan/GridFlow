# gridflow/download/dem_downloader.py
# Copyright (c) 2025 Bhuwan Shah

import sys
import json
import math
import time
import signal
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Dependency Check ---
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError as e:
    print(f"FATAL: Missing required library: {e.name}. Please install it using 'pip install boto3'.")
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

# --- Constants ---
# Copernicus (Global 30m)
AWS_BUCKET_COP30 = 'copernicus-dem-30m'
AWS_REGION_COP30 = 'eu-central-1'

# USGS 3DEP (US 10m / 1/3 arc-second)
AWS_BUCKET_USGS = 'prd-tnm'
AWS_REGION_USGS = 'us-west-2'

# --- Global Stop Event ---
stop_event = threading.Event()

def signal_handler(sig, frame):
    """Handles SIGINT (Ctrl+C)."""
    logging.info("Stop signal received! Gracefully shutting down...")
    stop_event.set()

class FileManager:
    """Handles file and directory management for DEM data."""
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

    def get_output_path(self, file_info: Dict) -> Path:
        """Constructs output path."""
        # Organize by source type (COP30 vs USGS10m) to keep things clean
        source = file_info.get('source', 'dem')
        subdir = self.download_dir / source
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / file_info['filename']

    def save_metadata(self, files: List[Dict], filename: str) -> None:
        metadata_path = self.metadata_dir / f"{self.metadata_prefix}{filename}"
        serializable_files = []
        for f in files:
            item = f.copy()
            if 'output_path' in item and isinstance(item['output_path'], Path):
                item['output_path'] = str(item['output_path'])
            serializable_files.append(item)
            
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_files, f, indent=2)
            logging.debug(f"Saved metadata to {metadata_path}")
        except IOError as e:
            logging.error(f"Failed to save metadata to {metadata_path}: {e}")

class QueryHandler:
    """Generates S3 keys for DEM tiles based on bounds and type."""
    def __init__(self, stop_event: threading.Event):
        self._stop_event = stop_event
        # We initialize clients lazily or maintain multiple if regions differ
        self.s3_cop = boto3.client('s3', region_name=AWS_REGION_COP30, config=Config(signature_version=UNSIGNED))
        self.s3_usgs = boto3.client('s3', region_name=AWS_REGION_USGS, config=Config(signature_version=UNSIGNED))

    def _format_cop_coord(self, val: float, is_lat: bool) -> str:
        """Formats lat/lon for Copernicus tiles (N50, W090)."""
        abs_val = abs(int(math.floor(val)))
        if is_lat:
            prefix = 'N' if val >= 0 else 'S'
            return f"{prefix}{abs_val:02d}"
        else:
            prefix = 'E' if val >= 0 else 'W'
            return f"{prefix}{abs_val:03d}"

    def generate_potential_files(self, bounds: Dict[str, float], dem_type: str = 'COP30') -> List[Dict]:
        """Calculates tiles covering the bounding box for the specified DEM source."""
        potential_files = []
        
        if dem_type == 'COP30':
            # Logic for Copernicus GLO-30
            lat_min = int(math.floor(bounds['south']))
            lat_max = int(math.floor(bounds['north']))
            lon_min = int(math.floor(bounds['west']))
            lon_max = int(math.floor(bounds['east']))

            for lat in range(lat_min, lat_max + 1):
                for lon in range(lon_min, lon_max + 1):
                    lat_str = self._format_cop_coord(lat, is_lat=True)
                    lon_str = self._format_cop_coord(lon, is_lat=False)
                    base_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
                    filename = f"{base_name}.tif"
                    s3_key = f"{base_name}/{filename}"
                    
                    potential_files.append({
                        'title': filename,
                        'filename': filename,
                        's3_key': s3_key,
                        's3_bucket': AWS_BUCKET_COP30,
                        'source': 'COP30',
                        'region': AWS_REGION_COP30
                    })

        elif dem_type == 'USGS10m':
            # Logic for USGS 3DEP 10m (1/3 arc-second)
            # USGS naming (nXXwYYY) is based on the upper-left corner usually.
            # Coverage logic: lat range [ceil(south), ceil(north)], lon range [floor(west), floor(east)]
            lat_start = int(math.ceil(bounds['south']))
            lat_end = int(math.ceil(bounds['north']))
            lon_start = int(math.floor(bounds['west']))
            lon_end = int(math.floor(bounds['east']))

            for lat in range(lat_start, lat_end + 1):
                for lon in range(lon_start, lon_end + 1):
                    # Naming: n42w093 covers 41-42N, 92-93W roughly
                    lat_str = f"n{lat:02d}"
                    lon_str = f"w{abs(lon):03d}"
                    base_name = f"USGS_13_{lat_str}{lon_str}"
                    filename = f"{base_name}.tif"
                    
                    # Key prefix - we don't know if it's in 'current' or 'historical' yet
                    # We'll set a placeholder key and resolve it in validate_files
                    potential_files.append({
                        'title': filename,
                        'filename': filename,
                        'tile_id': f"{lat_str}{lon_str}", # Helper to find key later
                        's3_bucket': AWS_BUCKET_USGS,
                        'source': 'USGS10m',
                        'region': AWS_REGION_USGS
                    })

        return potential_files

    def validate_files_on_s3(self, potential_files: List[Dict], is_gui_mode: bool = False) -> List[Dict]:
        """Verifies files exist. For USGS, resolves 'current' vs 'historical' path."""
        valid_files = []
        use_rich = HAS_UI_LIBS and not is_gui_mode
        status_context = None

        if use_rich:
            console = Console()
            status_context = console.status(f"[bold green]Querying AWS S3 for {len(potential_files)} tiles...", spinner="dots")
            status_context.start()
        else:
            logging.info(f"Querying AWS S3 for {len(potential_files)} tiles...")

        try:
            with ThreadPoolExecutor(max_workers=10, thread_name_prefix="S3Checker") as executor:
                future_to_task = {executor.submit(self._resolve_s3_key, p): p for p in potential_files}
                
                for future in as_completed(future_to_task):
                    if self._stop_event.is_set(): break
                    result = future.result()
                    if result:
                        valid_files.append(result)
        
        except Exception as e:
            logging.error(f"Error during S3 query: {e}")
        finally:
            if status_context: status_context.stop()
        
        if use_rich:
            Console().print(f"[bold blue]Query complete![/] Found {len(valid_files)} available tiles.")
        else:
            logging.info(f"Query complete! Found {len(valid_files)} available tiles.")
            
        return valid_files

    def _resolve_s3_key(self, file_info: Dict) -> Optional[Dict]:
        """Checks existence. For USGS, searches subfolders."""
        if self._stop_event.is_set(): return None
        
        source = file_info['source']
        bucket = file_info['s3_bucket']
        
        # client selection
        s3 = self.s3_cop if source == 'COP30' else self.s3_usgs

        if source == 'COP30':
            try:
                s3.head_object(Bucket=bucket, Key=file_info['s3_key'])
                file_info['size_bytes'] = 0 
                return file_info
            except ClientError:
                return None

        elif source == 'USGS10m':
            tile_id = file_info['tile_id']
            # Try 'current' first
            keys_to_try = [
                f"StagedProducts/Elevation/13/TIFF/current/{tile_id}/USGS_13_{tile_id}.tif",
                f"StagedProducts/Elevation/13/TIFF/historical/{tile_id}/USGS_13_{tile_id}.tif"
            ]
            
            for key in keys_to_try:
                try:
                    s3.head_object(Bucket=bucket, Key=key)
                    file_info['s3_key'] = key # Found it!
                    file_info['size_bytes'] = 0
                    return file_info
                except ClientError:
                    continue
            
            return None # Not found in either

class Downloader:
    """Manages the download process from S3."""
    def __init__(self, file_manager: FileManager, stop_event: threading.Event, **settings):
        self.file_manager = file_manager
        self._stop_event = stop_event
        self.settings = settings
        # Maintain separate clients for regions
        self.s3_cop = boto3.client('s3', region_name=AWS_REGION_COP30, config=Config(signature_version=UNSIGNED))
        self.s3_usgs = boto3.client('s3', region_name=AWS_REGION_USGS, config=Config(signature_version=UNSIGNED))

    def shutdown(self):
        logging.info("Downloader has been shut down.")

    def download_file(self, file_info: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        if self._stop_event.is_set(): return None, file_info
        
        output_path = self.file_manager.get_output_path(file_info)
        file_info['output_path'] = output_path 
        
        if output_path.exists():
            logging.debug(f"Skipping {output_path.name} - already exists.")
            return str(output_path), None
        
        try:
            # Select correct client
            s3 = self.s3_cop if file_info['source'] == 'COP30' else self.s3_usgs
            
            logging.debug(f"Downloading {file_info['s3_key']}")
            s3.download_file(
                file_info['s3_bucket'], 
                file_info['s3_key'], 
                str(output_path)
            )
            return str(output_path), None
        except Exception as e:
            if self._stop_event.is_set(): return None, file_info
            logging.warning(f"Failed to download {file_info['title']}: {e}")
            if output_path.exists(): output_path.unlink()
            return None, {**file_info, "error": str(e)}

    def download_all(self, files_to_download: List[Dict]) -> Tuple[List[str], List[Dict]]:
        downloaded, failed = [], []
        if not files_to_download: return [], []

        total_files = len(files_to_download)
        self.executor = ThreadPoolExecutor(
            max_workers=self.settings.get('workers', 4),
            thread_name_prefix='Downloader'
        )
        
        future_to_file = {
            self.executor.submit(self.download_file, f): f 
            for f in files_to_download
        }

        is_gui_mode = self.settings.get('is_gui_mode', False)
        use_tqdm = HAS_UI_LIBS and not is_gui_mode
        futures_iter = as_completed(future_to_file)
        
        if use_tqdm:
            futures_iter = tqdm(
                futures_iter, total=total_files, unit="file", 
                desc="Downloading", ncols=90, bar_format='  {l_bar}{bar}{r_bar}'
            )

        try:
            for i, future in enumerate(futures_iter):
                if self._stop_event.is_set():
                    for f in future_to_file: f.cancel()
                    break

                if is_gui_mode:
                    logging.info(f"Progress: {i + 1}/{total_files} files processed.")

                original_info = future_to_file[future]
                try:
                    path, failed_info = future.result()
                    if path:
                        downloaded.append(path)
                        if use_tqdm:
                            short_name = (Path(path).name[:40] + '..') if len(Path(path).name) > 40 else Path(path).name
                            tqdm.write(f"  ✔ Downloaded {short_name}")
                        elif is_gui_mode:
                            logging.info(f"Downloaded {Path(path).name}")
                    if failed_info:
                        failed.append(failed_info)
                        if use_tqdm:
                            tqdm.write(f"  ✖ Failed: {original_info['title']}")
                        elif is_gui_mode:
                            logging.info(f"Failed: {original_info['title']}")
                except Exception as e:
                    failed.append(original_info)
                    logging.error(f"Error processing {original_info['title']}: {e}")
        except Exception as e:
            print(f"\nCRITICAL ERROR in download loop: {e}", file=sys.stderr)
            raise e
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)

        self.successful_downloads = len(downloaded)
        return downloaded, failed

def load_config(config_path: str) -> Dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config file {config_path}: {e}")
        return {}

def create_download_session(settings: Dict[str, Any], stop_event: threading.Event) -> None:
    is_gui_mode = settings.get('is_gui_mode', False)
    try:
        if settings.get('retry_failed_path'):
            files_to_process = load_config(settings['retry_failed_path'])
        else:
            # 1. Check Bounds
            if not settings.get('bounds') and not settings.get('demo'):
                logging.error("Bounds (North, South, East, West) are required.")
                if not is_gui_mode: sys.exit(1)
                return

            dem_type = settings.get('dem_type', 'COP30')
            
            # 2. Query S3
            query_handler = QueryHandler(stop_event=stop_event)
            potential_files = query_handler.generate_potential_files(settings['bounds'], dem_type)
            
            if not potential_files:
                logging.error("Could not generate tile keys for the given bounds.")
                if not is_gui_mode: sys.exit(1)
                return

            files_to_process = query_handler.validate_files_on_s3(potential_files, is_gui_mode=is_gui_mode)

        if not files_to_process:
            logging.info(f"No available tiles found on S3 for {settings.get('dem_type')}.")
            if not is_gui_mode: sys.exit(0)
            return

        file_manager = FileManager(settings['output_dir'], settings['metadata_dir'], "gridflow_dem_")
        file_manager.save_metadata(files_to_process, "query_results.json")

        if settings.get('dry_run'):
            logging.info(f"Dry run: Would attempt to download {len(files_to_process)} tiles.")
            return

        settings.pop('stop_event', None)
        settings.pop('stop_flag', None)

        downloader = Downloader(file_manager, stop_event, **settings)
        downloaded, failed = downloader.download_all(files_to_process)

        if stop_event.is_set(): logging.warning("Process was stopped before completion.")
        if failed:
            file_manager.save_metadata(failed, "failed_downloads.json")
            logging.warning(f"{len(failed)} downloads failed. Check 'failed_downloads.json' for details.")
        
        logging.info(f"Completed: {downloader.successful_downloads}/{len(files_to_process)} tiles downloaded successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    query_group = parser.add_argument_group('Query Parameters')
    settings_group = parser.add_argument_group('Download Settings')

    query_group.add_argument("--bounds", type=float, nargs=4, metavar=('NORTH', 'SOUTH', 'EAST', 'WEST'), help="Bounding box coordinates in decimal degrees.")
    query_group.add_argument("--dem_type", default="COP30", choices=['COP30', 'USGS10m'], help="Source: 'COP30' (Global 30m) or 'USGS10m' (US 10m).")
    
    settings_group.add_argument("-o", "--output-dir", default="./downloads_dem", help="Output directory.")
    settings_group.add_argument("-md", "--metadata-dir", default="./metadata_dem", help="Metadata directory.")
    settings_group.add_argument("-ld", "--log-dir", default="./gridflow_logs", help="Log directory.")
    settings_group.add_argument("-ll", "--log-level", default="minimal", choices=["minimal", "verbose", "debug"], help="Log verbosity.")
    settings_group.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel workers.")
    settings_group.add_argument("--dry-run", action="store_true", help="Find tiles but do not download.")
    settings_group.add_argument("--demo", action="store_true", help="Run with demo settings (Iowa).")
    settings_group.add_argument("-c", "--config", help="Path to JSON config file.")
    settings_group.add_argument("--retry-failed", help="Path to failed_downloads.json to retry.")

def main(args=None):
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="DEM Downloader (Supports Copernicus GLO-30 & USGS 3DEP)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()

    if not getattr(args, 'is_gui_mode', False):
        setup_logging(args.log_dir, args.log_level, prefix="dem_downloader")
        signal.signal(signal.SIGINT, signal_handler)

    active_stop_event = getattr(args, 'stop_event', stop_event)

    config = load_config(args.config) if args.config else {}
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    settings = {**config, **cli_args}

    if settings.get('demo'):
        # Iowa demo bounds 
        settings['bounds'] = {'north': 42.9, 'south': 41.1, 'east': -91.1, 'west': -92.9}
        
        demo_cmd = (
            "gridflow dem --bounds 42.9 41.1 -91.1 -92.9 --dem_type COP30\n"
            "  gridflow dem --bounds 42.9 41.1 -91.1 -92.9 --dem_type USGS10m"
        )

        if HAS_UI_LIBS and not getattr(args, 'is_gui_mode', False):
            console = Console()
            console.print(f"[bold yellow]Running in demo mode (Iowa Region - 4 Tiles).[/]")
            console.print(f"Demo Commands:\n  [dim]{demo_cmd}[/dim]\n")
        else:
            logging.info("Running in demo mode (Iowa Region).")
            logging.info(f"Demo Commands:\n  {demo_cmd}")

        print("--- Phase 1: Downloading Copernicus Global 30m ---")
        phase1_settings = settings.copy() 
        phase1_settings['dem_type'] = 'COP30'
        create_download_session(phase1_settings, active_stop_event)
        
        if not active_stop_event.is_set():
            print("\n--- Phase 2: Downloading USGS 3DEP 10m (High Res) ---")
            phase2_settings = settings.copy() 
            phase2_settings['dem_type'] = 'USGS10m'
            create_download_session(phase2_settings, active_stop_event)
            
        logging.info("Demo complete.")
        return

    else:
        if not settings.get('bounds') and not settings.get('retry_failed_path'):
            logging.error("Bounds are required when not in --demo mode.")
            if not getattr(args, 'is_gui_mode', False): sys.exit(1)
            return
        
        if isinstance(settings.get('bounds'), list):
            b = settings['bounds']
            if b[0] <= b[1]:
                logging.error("Invalid bounds: North must be greater than South.")
                if not getattr(args, 'is_gui_mode', False): sys.exit(1)
                return
            settings['bounds'] = {'north': b[0], 'south': b[1], 'east': b[2], 'west': b[3]}

    create_download_session(settings, active_stop_event)
    
    if active_stop_event.is_set():
        logging.warning("Execution was interrupted.")
        if not getattr(args, 'is_gui_mode', False): sys.exit(130)

    logging.info("Process finished.")

if __name__ == "__main__":
    main()