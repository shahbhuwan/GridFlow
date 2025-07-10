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
import time
import signal
import logging
import requests
import threading
from pathlib import Path
from datetime import datetime
from hashlib import md5, sha256
from urllib.parse import urlencode
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Local Imports
try:
    from gridflow.utils.logging_utils import setup_logging
except ImportError:
    print("FATAL: gridflow/utils/logging_utils.py not found. Please ensure it is in the correct directory.")
    sys.exit(1)

# Constants
ESGF_NODES = [
    "https://esgf-node.llnl.gov/esg-search/search",
    "https://esgf-node.ipsl.upmc.fr/esg-search/search",
    "https://esgf-data.dkrz.de/esg-search/search",
    "https://esgf-index1.ceda.ac.uk/esg-search/search"
]

# Global Stop Event for Graceful Shutdown
stop_event = threading.Event()

def signal_handler(sig, frame):
    """Handles SIGINT (Ctrl+C) to gracefully shut down the application."""
    logging.warning("Stop signal received! Gracefully shutting down...")
    stop_event.set()
    logging.info("Please wait for ongoing tasks to complete or timeout.")

class InterruptibleSession(requests.Session):
    """A requests.Session that checks a threading.Event before making a request."""
    def __init__(self, stop_event: threading.Event, cert_path: Optional[str] = None):
        super().__init__()
        self.stop_event = stop_event
        self.cert = cert_path  # Path to certificate for OpenID authentication
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.mount("http://", adapter)
        self.mount("https://", adapter)

    def get(self, url, **kwargs):
        if self.stop_event.is_set():
            raise requests.exceptions.RequestException("Download interrupted by user.")
        kwargs.setdefault("timeout", (10, 5))
        if self.cert:
            kwargs["cert"] = self.cert
        return super().get(url, **kwargs)

class FileManager:
    """Handles file and directory management."""
    def __init__(self, download_dir: str, metadata_dir: str, save_mode: str, prefix: str = "", metadata_prefix: str = ""):
        self.download_dir = Path(download_dir)
        self.metadata_dir = Path(metadata_dir)
        self.save_mode = save_mode.lower()
        self.prefix = prefix
        self.metadata_prefix = metadata_prefix
        try:
            self.download_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create directories: {e}")
            sys.exit(1)

    def get_output_path(self, file_info: Dict) -> Path:
        filename = file_info.get('title', 'unknown_file.nc')
        activity = file_info.get('activity_id', ['unknown'])[0].replace('/', '_')
        variable = file_info.get('variable_id', ['unknown'])[0].replace('/', '_')
        nominal_resolution = file_info.get('nominal_resolution', ['unknown_resolution'])[0]
        resolution = nominal_resolution.replace(' ', '')

        if self.save_mode == 'flat':
            return self.download_dir / f"{self.prefix}{activity}_{resolution}_{filename}"
        else:  # structured
            subdir = self.download_dir / variable / resolution / activity
            subdir.mkdir(parents=True, exist_ok=True)
            return subdir / filename

    def save_metadata(self, files: List[Dict], filename: str) -> None:
        metadata_path = self.metadata_dir / f"{self.metadata_prefix}{filename}"
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(files, f, indent=2)
            logging.debug(f"Saved metadata to {metadata_path}")
        except IOError as e:
            logging.error(f"Failed to save metadata to {metadata_path}: {e}")

class QueryHandler:
    """Handles querying ESGF nodes for dataset files."""
    def __init__(self, nodes: List[str] = ESGF_NODES, stop_event: Optional[threading.Event] = None):
        self.nodes = nodes
        self._stop_event = stop_event or threading.Event()
        self.session = InterruptibleSession(self._stop_event)

    def build_query(self, base_url: str, params: Dict[str, str]) -> str:
        query_params = {
            'type': 'File', 'project': 'CMIP6', 'format': 'application/solr+json',
            'limit': '1000', 'distrib': 'true', **params
        }
        return f"{base_url}?{urlencode(query_params, safe='/')}"

    def fetch_datasets(self, params: Dict[str, str], timeout: int) -> List[Dict]:
        all_files, seen_ids = [], set()
        for node in self.nodes:
            if self._stop_event.is_set():
                logging.info("Querying stopped by user.")
                break
            try:
                logging.info(f"Querying node: {node}")
                node_files = self._fetch_from_node(node, params, timeout)
                unique_files = [f for f in node_files if f.get('id') not in seen_ids]
                for f in unique_files: seen_ids.add(f.get('id'))
                all_files.extend(unique_files)
                if all_files or not any(params.values()):
                    logging.debug(f"Retrieved {len(all_files)} unique files from {node}.")
                    return all_files
            except requests.exceptions.Timeout:
                logging.warning(f"The request to node {node} timed out. Trying next node.")
            except requests.exceptions.ConnectionError:
                logging.warning(f"Could not connect to node {node}. Please check your network. Trying next node.")
            except requests.RequestException as e:
                logging.warning(f"An error occurred while connecting to {node}: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred while querying {node}: {e}", exc_info=True)
        
        if not all_files:
            logging.error("All nodes failed or no files were found for the given query.")
        return all_files

    def _fetch_from_node(self, node: str, params: Dict[str, str], timeout: int) -> List[Dict]:
        files, offset = [], 0
        while not self._stop_event.is_set():
            query_url = self.build_query(node, {**params, 'offset': str(offset)})
            logging.debug(f"Querying URL: {query_url}")
            response = self.session.get(query_url, timeout=timeout)
            response.raise_for_status()
            data = response.json().get('response', {})
            docs = data.get('docs', [])
            if not docs:
                break
            files.extend(docs)
            if offset + len(docs) >= int(data.get('numFound', 0)):
                break
            offset += len(docs)
        return files

class Downloader:
    """Manages the file download process using a thread pool."""
    def __init__(self, file_manager: FileManager, stop_event: threading.Event, **settings):
        self.file_manager = file_manager
        self._stop_event = stop_event
        self.settings = settings
        self.cert_path = None
        self.log_lock = threading.Lock()
        self.successful_downloads = 0
        self.executor = None
        self.pending_futures: Dict[Future, Dict] = {}
        self.worker_status: Dict[int, str] = {}

        # Initialize session with OpenID certificate if provided
        openid, username, password = settings.get('openid'), settings.get('username'), settings.get('password')
        if openid and username and password:
            self._fetch_esgf_certificate(openid, username, password)
        self.session = InterruptibleSession(self._stop_event, cert_path=self.cert_path)

        if username and password and not openid:
            self.session.auth = (username, password)

    def _fetch_esgf_certificate(self, openid: str, username: str, password: str) -> None:
        """Fetches an ESGF certificate using OpenID credentials."""
        cert_dir = Path.home() / ".esg"
        cert_dir.mkdir(exist_ok=True)
        self.cert_path = cert_dir / "credentials.pem"

        if self.cert_path.exists():
            try:
                # Check certificate validity (simplified check)
                with open(self.cert_path, 'r') as f:
                    cert_content = f.read()
                if "BEGIN CERTIFICATE" in cert_content:
                    logging.debug(f"Using existing certificate at {self.cert_path}")
                    return
            except Exception:
                logging.debug("Existing certificate invalid or corrupted. Fetching new one.")

        try:
            # Simulate OpenID login to fetch certificate (ESGF nodes vary, using a generic approach)
            login_url = openid.rsplit('/', 1)[0] + "/esgf-idp/openid/"  # Extract base URL
            auth_data = {
                "openid": openid,
                "username": username,
                "password": password
            }
            response = requests.post(login_url, data=auth_data, timeout=30)
            response.raise_for_status()

            # Assume response contains certificate (simplified; actual flow may involve redirects)
            cert_content = response.text
            if "BEGIN CERTIFICATE" not in cert_content:
                raise ValueError("No valid certificate received from ESGF.")

            with open(self.cert_path, 'w') as f:
                f.write(cert_content)
            logging.info(f"Fetched ESGF certificate and saved to {self.cert_path}")
        except requests.RequestException as e:
            logging.error(f"Failed to fetch ESGF certificate: {e}")
            self.cert_path = None
        except Exception as e:
            logging.error(f"Unexpected error during certificate fetch: {e}")
            self.cert_path = None

    def shutdown(self):
        """Shuts down the thread pool gracefully."""
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        self.session.close()
        logging.info("Downloader has been shut down.")

    def verify_checksum(self, file_path: Path, file_info: Dict) -> bool:
        checksum = file_info.get('checksum', [''])[0]
        checksum_type = file_info.get('checksum_type', ['sha256'])[0].lower()
        if not checksum: return True
        try:
            hasher = md5() if checksum_type == 'md5' else sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            file_hash = hasher.hexdigest()
            if file_hash == checksum: return True
            logging.error(f"Checksum mismatch for {file_path.name}: expected {checksum}, got {file_hash}")
            return False
        except Exception as e:
            logging.error(f"Checksum verification failed for {file_path.name}: {e}")
            return False

    def download_file(self, file_info: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        """Downloads a single file with robust URL parsing and retries."""
        thread_id = threading.get_ident()
        filename = file_info.get('title', 'unknown_file')

        try:
            with self.log_lock:
                self.worker_status[thread_id] = filename
                logging.debug(f"[Worker-{thread_id}] Starting download of: {filename}")

            if self._stop_event.is_set(): return None, file_info

            output_path = self.file_manager.get_output_path(file_info)
            if output_path.exists() and self.verify_checksum(output_path, file_info):
                logging.info(f"Downloaded {filename} (already exists)")
                return str(output_path), None
            
            urls_data = file_info.get('url', [])
            download_urls = [p[0] for p in (u.split('|') for u in urls_data) if len(p) == 3 and p[2] == 'HTTPServer']
            
            if not download_urls:
                logging.error(f"No 'HTTPServer' URL for {filename}")
                return None, file_info

            for url in download_urls:
                if self._stop_event.is_set(): break
                temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
                try:
                    logging.debug(f"[Worker-{thread_id}] Trying URL {url}")
                    response = self.session.get(url, stream=True, verify=not self.settings.get('no_verify_ssl'))
                    response.raise_for_status()

                    with open(temp_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if self._stop_event.is_set():
                                raise requests.exceptions.RequestException("Download interrupted by user.")
                            f.write(chunk)

                    if self.verify_checksum(temp_path, file_info):
                        temp_path.rename(output_path)
                        logging.info(f"Downloaded {filename}")
                        return str(output_path), None
                except requests.RequestException as e:
                    logging.warning(f"[Worker-{thread_id}] Download failed for {filename} from {url}: {e}")
                finally:
                    if temp_path.exists(): temp_path.unlink(missing_ok=True)
            
            return None, file_info
        finally:
            with self.log_lock:
                if thread_id in self.worker_status:
                    del self.worker_status[thread_id]
                    logging.debug(f"[Worker-{thread_id}] Finished task for {filename}.")

    def download_all(self, files_to_download: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Manages the download of a list of files using a thread pool."""
        downloaded, failed = [], []
        if not files_to_download: return [], []
        
        total_files = len(files_to_download)
        self.executor = ThreadPoolExecutor(max_workers=self.settings.get('workers'), thread_name_prefix='Downloader')
        self.pending_futures = {self.executor.submit(self.download_file, f): f for f in files_to_download}

        try:
            for i, future in enumerate(as_completed(self.pending_futures)):
                logging.info(f"Progress: {i + 1}/{total_files} files processed.")
                if self._stop_event.is_set(): break
                original_file_info = self.pending_futures[future]
                try:
                    path, failed_info = future.result()
                    if path: downloaded.append(path)
                    if failed_info: failed.append(failed_info)
                except Exception as e:
                    failed.append(original_file_info)
                    logging.error(f"A critical error occurred while processing {original_file_info.get('title')}: {e}", exc_info=True)
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

def create_download_session(params: Dict[str, Any], settings: Dict[str, Any], stop_event: threading.Event) -> None:
    """Sets up and executes a download session."""
    try:
        if settings.get('retry_failed_path'):
            files_to_process = load_config(settings['retry_failed_path'])
            if not files_to_process:
                logging.info("Retry file is empty. Nothing to do.")
                sys.exit(0)
            logging.info(f"Retrying {len(files_to_process)} files from {settings['retry_failed_path']}")
        else:
            if not any(params.values()) and not settings.get('demo'):
                 logging.error("No search parameters provided. Please specify criteria like --variable or --model.")
                 sys.exit(1)
            
            query_handler = QueryHandler(stop_event=stop_event)
            all_found_files = query_handler.fetch_datasets(params, settings['timeout'])
            
            if not all_found_files: sys.exit(1)
            if stop_event.is_set(): return
            
            files_by_title = {f['title']: f for f in all_found_files if 'title' in f}
            unique_files = list(files_by_title.values())
            logging.info(f"Found {len(unique_files)} unique files.")

            max_downloads = settings.get('max_downloads')
            files_to_process = unique_files[:max_downloads] if max_downloads else unique_files

        metadata_prefix = f"gridflow_{params.get('project', 'cmip6').lower()}_"
        file_manager = FileManager(
            settings['output_dir'], settings['metadata_dir'], settings['save_mode'],
            metadata_prefix=metadata_prefix
        )
        file_manager.save_metadata(files_to_process, "query_results_to_download.json")

        if settings.get('dry_run'):
            logging.info(f"Dry run: Would attempt to download {len(files_to_process)} files.")
            return

        downloader = Downloader(file_manager, stop_event, **settings)
        downloaded, failed = downloader.download_all(files_to_process)
        
        if stop_event.is_set():
            logging.warning("Process was stopped before completion.")
        
        if failed:
            file_manager.save_metadata(failed, "failed_downloads.json")
            logging.warning(f"{len(failed)} downloads failed. Check 'failed_downloads.json' for details.")
        
        logging.info(f"Completed: {downloader.successful_downloads}/{len(files_to_process)} files downloaded successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred during the session: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    """Add CMIP6 downloader arguments to the provided parser."""
    query_group = parser.add_argument_group('Query Parameters (for filtering data)')
    settings_group = parser.add_argument_group('Download & Output Settings')
    mode_group = parser.add_argument_group('Authentication & Special Modes')

    query_group.add_argument("--config", help="Path to JSON config file to pre-fill arguments.")
    query_group.add_argument("--project", default="CMIP6", help="Project name (e.g., CMIP6).")
    query_group.add_argument("--activity", help="Activity ID (e.g., HighResMIP, ScenarioMIP).")
    query_group.add_argument("--experiment", help="Experiment ID (e.g., hist-1950, ssp585).")
    query_group.add_argument("--model", help="Source ID / Model name (e.g., HadGEM3-GC31-LL).")
    query_group.add_argument("--variable", help="Variable ID (e.g., tas, pr).")
    query_group.add_argument("--frequency", help="Time frequency (e.g., day, Amon).")
    query_group.add_argument("--resolution", help="Nominal resolution (e.g., '250 km', '50 km').")
    query_group.add_argument("--ensemble", help="Variant label / ensemble member (e.g., r1i1p1f1).")
    query_group.add_argument("--grid_label", help="Grid label (e.g., gn).")
    query_group.add_argument("--latest", action='store_true', help="Only get the latest version of files.")

    settings_group.add_argument("--output-dir", default="./downloads_cmip6", help="Directory to save downloaded files.")
    settings_group.add_argument("--metadata-dir", default="./metadata_cmip6", help="Directory to save metadata files.")
    settings_group.add_argument("--log-dir", default="./gridflow_logs", help="Directory to save log files.")
    settings_group.add_argument("--log-level", default="verbose", choices=["minimal", "verbose", "debug"], help="Set logging verbosity for the console.")
    settings_group.add_argument("--save-mode", default="structured", choices=["structured", "flat"], help="File saving mode: 'structured' (subdirs) or 'flat' (one dir).")
    settings_group.add_argument("--workers", type=int, default=4, help="Number of parallel download workers.")
    settings_group.add_argument("--timeout", type=int, default=30, help="Network request timeout in seconds.")
    settings_group.add_argument("--max-downloads", type=int, help="Maximum number of files to download in this session.")
    settings_group.add_argument("--no-verify-ssl", action="store_true", help="Disable SSL certificate verification (use with caution).")

    mode_group.add_argument("--openid", help="ESGF OpenID (e.g., https://esgf-node.llnl.gov/esgf-idp/openid/username).")
    mode_group.add_argument("--id", help="ESGF username for authentication.")
    mode_group.add_argument("--password", help="ESGF password for authentication.")
    mode_group.add_argument("--retry-failed", help="Path to a 'failed_downloads.json' file to retry.")
    mode_group.add_argument("--dry-run", action="store_true", help="Query for files but do not download them.")
    mode_group.add_argument("--demo", action="store_true", help="Run with a small, pre-defined demo query.")

def main(args=None):
    """Main entry point for the CMIP6 Downloader."""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="CMIP6 Data Downloader", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()

    setup_logging(args.log_dir, args.log_level)
    signal.signal(signal.SIGINT, signal_handler)

    config = load_config(args.config) if args.config else {}
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    settings = {**config, **cli_args}

    params = {
        'project': settings.get('project'),
        'activity_id': settings.get('activity'),
        'experiment_id': settings.get('experiment'),
        'frequency': settings.get('frequency'),
        'variable_id': settings.get('variable'),
        'source_id': settings.get('model'),
        'variant_label': settings.get('ensemble'),
        'grid_label': settings.get('grid_label'),
        'nominal_resolution': settings.get('resolution')
    }
    
    if settings.get('latest'):
        params['latest'] = 'true'
    
    if settings.get('demo'):
        params = {
            'project': 'CMIP6', 'activity_id': 'HighResMIP', 'variable_id': 'tas', 
            'source_id': 'HadGEM3-GC31-LL', 'experiment_id': 'hist-1950', 'frequency': 'day',
            'variant_label': 'r1i1p1f1', 'nominal_resolution': '250 km', 'limit': '5'
        }
        settings['max_downloads'] = 5
        logging.info("Running in demo mode with a pre-defined query.")

    # Validate OpenID authentication
    if settings.get('openid') and not all([settings.get('id'), settings.get('password')]):
        logging.error("Both --id and --password are required when using --openid.")
        sys.exit(1)

    params = {k: v for k, v in params.items() if v}
    settings['username'] = settings.get('id')
    
    create_download_session(params, settings, stop_event)
    
    if stop_event.is_set():
        logging.warning("Execution was interrupted.")
        sys.exit(130)
    
    logging.info("Process finished.")

if __name__ == "__main__":
    main()