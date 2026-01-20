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
import time
import requests
import threading
from pathlib import Path
from bs4 import BeautifulSoup
from hashlib import md5, sha256
from urllib.parse import urlencode
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

try:
    from tqdm import tqdm
    from rich.console import Console
    HAS_UI_LIBS = True
except ImportError:
    HAS_UI_LIBS = False

# Local Imports
try:
    from gridflow.utils.logging_utils import setup_logging
except ImportError:
    print("FATAL: gridflow/utils/logging_utils.py not found. Please ensure it is in the correct directory.")
    sys.exit(1)

# Constants
ALL_ESGF_NODES = {
    "LLNL": "https://esgf-node.llnl.gov/esg-search/search",
    "IPSL": "https://esgf-node.ipsl.upmc.fr/esg-search/search",
    "DKRZ": "https://esgf-data.dkrz.de/esg-search/search",
    "CEDA": "https://esgf-index1.ceda.ac.uk/esg-search/search",
    "ANL": "https://esgf.anl.gov/esg-search/search",
    "ORNL": "https://esgf.ornl.gov/esg-search/search",
    "NCI": "https://esgf.nci.org.au/esg-search/search",
    "LIU": "https://esg-dn1.nsc.liu.se/esg-search/search"
}
NODES_STATUS_URL = "https://esgf.github.io/nodes.html"
DKRZ_NODE_KEY = "DKRZ" # Priority

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
    """Handles file and directory management for CMIP5 data."""
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
        """Constructs a logical output path based on CMIP5 metadata."""
        filename = file_info.get('title', 'unknown_file.nc')
        variable = file_info.get('variable', ['unknown'])[0]
        model = file_info.get('model', ['unknown'])[0]
        experiment = file_info.get('experiment', ['unknown'])[0]
        ensemble = file_info.get('ensemble', ['unknown'])[0]

        if self.save_mode == 'flat':
            prefixed_filename = f"{self.prefix}{model}_{experiment}_{filename}"
            return self.download_dir / prefixed_filename
        else:  # structured
            subdir = self.download_dir / variable / model / experiment / ensemble
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
    """Handles querying ESGF nodes for CMIP5 dataset files in parallel."""
    def __init__(self, stop_event: Optional[threading.Event] = None):
        self.all_nodes_map = ALL_ESGF_NODES
        self._stop_event = stop_event or threading.Event()
        self.session = InterruptibleSession(self._stop_event)
        self.nodes = self._get_available_nodes()

    def _get_available_nodes(self) -> List[str]:
        """Fetches the list of active ESGF nodes from the status page and prioritizes them."""
        logging.info(f"Checking node status from: {NODES_STATUS_URL}")
        try:
            response = requests.get(NODES_STATUS_URL, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            list_items = soup.find_all('li')
            found_node_keys = set()
            for item in list_items:
                item_text = item.get_text()
                for key in self.all_nodes_map:
                    if key in item_text:
                        found_node_keys.add(key)
                        break
            
            if not found_node_keys:
                logging.warning("Could not parse any known nodes from the status page. Falling back to the hardcoded list.")
                available_urls = list(self.all_nodes_map.values())
            else:
                logging.debug(f"Nodes found on status page: {sorted(list(found_node_keys))}")
                available_urls = [self.all_nodes_map[key] for key in found_node_keys if key in self.all_nodes_map]

            prioritized_nodes = []
            dkrz_url = self.all_nodes_map.get(DKRZ_NODE_KEY)

            if dkrz_url and dkrz_url in available_urls:
                prioritized_nodes.append(dkrz_url)
                for url in available_urls:
                    if url != dkrz_url:
                        prioritized_nodes.append(url)
            else:
                prioritized_nodes = available_urls

            logging.debug(f"Final hierarchical node list created ({len(prioritized_nodes)} available nodes): {prioritized_nodes}")
            if not prioritized_nodes:
                logging.error("No available ESGF nodes could be determined. Please check your network or the status page.")

            return prioritized_nodes
            
        except requests.RequestException as e:
            logging.error(f"Failed to fetch node status page: {e}. Using the full hardcoded list of nodes as a fallback.")
            return list(self.all_nodes_map.values())
        except Exception as e:
            logging.error(f"An unexpected error occurred while checking node status: {e}. Using fallback list.", exc_info=True)
            return list(self.all_nodes_map.values())

    def build_query(self, base_url: str, params: Dict[str, str]) -> str:
        query_params = {
            'type': 'File',
            'project': 'CMIP5',
            'format': 'application/solr+json',
            'limit': '1000',
            'distrib': 'true',
            **params
        }
        return f"{base_url}?{urlencode(query_params, safe='/')}"

    def fetch_datasets(self, params: Dict[str, str], timeout: int, is_gui_mode: bool = False) -> List[Dict]:
        """
        Queries all available ESGF nodes in parallel and merges the results.
        """
        if not self.nodes:
            logging.error("No ESGF nodes available to query.")
            return []

        merged_files: Dict[str, Dict] = {}
        num_workers = len(self.nodes)

        # UI Logic: Use Rich spinner ONLY if we have libs AND we are not in GUI mode
        use_rich = HAS_UI_LIBS and not is_gui_mode
        status_context = None

        if use_rich:
            # Use default Console (no padding)
            console = Console()
            # Spinner that shows work is happening
            status_context = console.status(f"[bold green]Querying {num_workers} ESGF nodes (this may take a minute)...", spinner="dots")
            status_context.start()
        else:
            logging.info(f"Querying {num_workers} node(s) in parallel...")

        try:
            with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='QueryWorker') as executor:
                future_to_node = {
                    executor.submit(self._fetch_from_node, node, params, timeout): node 
                    for node in self.nodes
                }

                for future in as_completed(future_to_node):
                    node = future_to_node[future]
                    try:
                        files_from_node = future.result()
                        if not files_from_node:
                            logging.debug(f"No results returned from node: {node}")
                            continue
                        
                        logging.debug(f"Received {len(files_from_node)} file results from {node}")
                        
                        for file_info in files_from_node:
                            # FIX: Use 'title' for robust deduplication in CMIP5
                            file_id = file_info.get('title')
                            if not file_id:
                                continue

                            if file_id not in merged_files:
                                merged_files[file_id] = file_info
                            else:
                                # File already found, merge the URLs
                                existing_urls = set(merged_files[file_id].get('url', []))
                                new_urls = set(file_info.get('url', []))
                                merged_files[file_id]['url'] = sorted(list(existing_urls.union(new_urls)))
                    
                    except Exception as e:
                        logging.error(f"An error occurred while querying node {node}: {e}")

            if not merged_files:
                if status_context: status_context.stop()
                logging.info("All nodes failed or no files were found for the given query.")
                return []
            
            final_list = list(merged_files.values())
            
            if status_context: status_context.stop()
            
            # Print success message
            if use_rich:
                console.print(f"[bold blue]Query complete![/] Found {len(final_list)} unique files.")
            else:
                logging.info(f"Parallel query complete. Found {len(final_list)} unique files across all nodes.")
            
            return final_list

        finally:
            if status_context:
                status_context.stop()

    def _fetch_from_node(self, node: str, params: Dict[str, str], timeout: int) -> List[Dict]:
        files, offset = [], 0
        logging.debug(f"Worker started for node: {node}")
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

        self.dead_urls: set[str] = set()
        self.dead_lock = threading.Lock()

        self.per_host_limit = int(self.settings.get('per_host_limit', 0))
        self._host_semaphores: Dict[str, threading.Semaphore] = {}
        self._host_lock = threading.Lock()

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
                # Construct the standard certificate endpoint from the OpenID's hostname.
                # This is a much more reliable method that works for CEDA, LLNL, and others.
                from urllib.parse import urlparse
                parsed_uri = urlparse(openid)
                hostname = parsed_uri.netloc
                # The standard endpoint for certificate retrieval via username/password
                login_url = f"https://{hostname}/esgf-idp/openid/login"
                
                logging.debug(f"Attempting to fetch certificate from standard endpoint: {login_url}")
                
                auth_data = {
                    "openid_identifier": openid,
                    "username": username,
                    "password": password
                }
                
                response = requests.post(login_url, data=auth_data, timeout=30)
                response.raise_for_status()

                cert_content = response.text
                if "BEGIN CERTIFICATE" not in cert_content:
                    raise ValueError("No valid certificate received from ESGF. Please check your credentials.")

                with open(self.cert_path, 'w') as f:
                    f.write(cert_content)
                logging.info(f"Fetched ESGF certificate and saved to {self.cert_path}")
            except requests.RequestException as e:
                logging.error(f"Failed to fetch ESGF certificate: {e}. Please check your credentials and network connection.")
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
                for chunk in iter(lambda: f.read(8192), b""): hasher.update(chunk)
            file_hash = hasher.hexdigest()
            if file_hash == checksum: return True
            logging.error(f"Checksum mismatch for {file_path.name}: expected {checksum}, got {file_hash}")
            return False
        except Exception as e:
            logging.error(f"Checksum verification failed for {file_path.name}: {e}")
            return False

    def download_file(self, file_info: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        """Downloads a single file with robust URL parsing and retries."""
        import random
        from urllib.parse import urlparse

        thread_id = threading.get_ident()
        filename = file_info.get('title', 'unknown_file')

        try:
            with self.log_lock:
                self.worker_status[thread_id] = filename
                logging.debug(f"[Worker-{thread_id}] Starting download of: {filename}")

            if self._stop_event.is_set():
                return None, file_info

            output_path = self.file_manager.get_output_path(file_info)
            # Fast path: already present and valid
            if output_path.exists() and self.verify_checksum(output_path, file_info):
                # FIX: Debug only, so it doesn't spam console
                logging.debug(f"Downloaded {filename} (already exists)")
                return str(output_path), None
            
            urls_data = file_info.get('url', [])
            download_urls = [p[0] for p in (u.split('|') for u in urls_data) if len(p) == 3 and p[2] == 'HTTPServer']
            if not download_urls:
                logging.error(f"No 'HTTPServer' URL for {filename}")
                return None, file_info

            # Optional preferred nodes ordering
            prefer = [h.strip() for h in (self.settings.get('prefer_nodes') or "").split(',') if h.strip()]
            if prefer:
                def pref_key(u: str) -> int:
                    host = urlparse(u).netloc
                    for i, frag in enumerate(prefer):
                        if frag in host:
                            return i
                    return len(prefer) + 1
                download_urls.sort(key=pref_key)

            # Resume settings
            resume_enabled = bool(self.settings.get('resume', False))
            temp_path = output_path.with_suffix(output_path.suffix + '.part')  # persistent partial for resume
            existing_bytes = temp_path.stat().st_size if resume_enabled and temp_path.exists() else 0

            # Timeouts tuned for fast stop responsiveness
            connect_timeout = 5
            read_timeout = max(5, int(self.settings.get('timeout', 10)))

            for url in download_urls:
                if self._stop_event.is_set():
                    return None, file_info

                with self.dead_lock:
                    if url in self.dead_urls:
                        logging.debug(f"[Worker-{thread_id}] Skipping known-dead mirror {url}")
                        continue

                host = urlparse(url).netloc
                host_sem = None
                if getattr(self, "per_host_limit", 0):
                    with self._host_lock:
                        host_sem = self._host_semaphores.setdefault(host, threading.Semaphore(self.per_host_limit))
                    host_sem.acquire()

                try:
                    # Probe with HEAD if resuming from a partial
                    use_range = False
                    expected_size = None
                    if resume_enabled and existing_bytes > 0:
                        try:
                            h = self.session.head(url, timeout=(connect_timeout, 5), allow_redirects=True,
                                                verify=not self.settings.get('no_verify_ssl'))
                            h.raise_for_status()
                            accept_ranges = h.headers.get("Accept-Ranges", "").lower() == "bytes"
                            cl_header = h.headers.get("Content-Length")
                            expected_size = int(cl_header) if cl_header and cl_header.isdigit() else None
                            if accept_ranges and (expected_size is None or existing_bytes < expected_size):
                                use_range = True
                        except requests.RequestException as e:
                            logging.debug(f"[Worker-{thread_id}] HEAD probe failed for {url}: {e}")
                            use_range = existing_bytes > 0

                    headers = {}
                    mode = 'wb'
                    start_from = 0
                    if resume_enabled and existing_bytes > 0 and use_range:
                        headers["Range"] = f"bytes={existing_bytes}-"
                        mode = 'ab'
                        start_from = existing_bytes

                    logging.debug(f"[Worker-{thread_id}] Trying URL {url} (resume={resume_enabled}, offset={start_from})")

                    resp = self.session.get(
                        url, stream=True, verify=not self.settings.get('no_verify_ssl'),
                        timeout=(connect_timeout, read_timeout), headers=headers or None,
                    )

                    if resp.status_code == 416 and resume_enabled and existing_bytes > 0:
                        logging.debug(f"[Worker-{thread_id}] 416 for {url}; retrying from start.")
                        resp.close()
                        headers, mode, start_from = {}, 'wb', 0
                        resp = self.session.get(url, stream=True, verify=not self.settings.get('no_verify_ssl'),
                                              timeout=(connect_timeout, read_timeout))

                    resp.raise_for_status()

                    bytes_written = 0
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, mode) as f:
                        for chunk in resp.iter_content(chunk_size=1 << 15):
                            if self._stop_event.is_set():
                                raise requests.exceptions.RequestException("Download interrupted by user.")
                            if chunk:
                                f.write(chunk)
                                bytes_written += len(chunk)

                    final_size = (start_from + bytes_written)
                    if expected_size is not None and final_size != expected_size:
                        logging.warning(f"[Worker-{thread_id}] Size mismatch for {filename}: {final_size} vs {expected_size}")

                    if self.verify_checksum(temp_path, file_info):
                        temp_path.replace(output_path)
                        # FIX: Debug only. 'download_all' will handle the pretty print.
                        logging.debug(f"Downloaded {filename}")
                        return str(output_path), None
                    else:
                        logging.warning(f"[Worker-{thread_id}] Checksum mismatch for {filename} from {url}")

                except requests.RequestException as e:
                    if self._stop_event.is_set():
                        return None, file_info
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    if status in (404, 410):
                        with self.dead_lock:
                            self.dead_urls.add(url)
                    logging.warning(f"[Worker-{thread_id}] Download failed for {filename} from {url}: {e}")

                finally:
                    if not resume_enabled and temp_path.exists():
                        try: temp_path.unlink() 
                        except: pass
                    if host_sem: host_sem.release()

                if not self._stop_event.is_set():
                    time.sleep(0.2 + 0.4 * random.random())
                
                existing_bytes = temp_path.stat().st_size if resume_enabled and temp_path.exists() else 0

            return None, file_info

        finally:
            with self.log_lock:
                if thread_id in self.worker_status:
                    del self.worker_status[thread_id]

    def download_all(self, files_to_download: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Manages the download of a list of files using a thread pool."""
        import sys
        
        downloaded, failed = [], []
        if not files_to_download:
            return [], []

        total_files = len(files_to_download)
        self.executor = ThreadPoolExecutor(
            max_workers=self.settings.get('workers'),
            thread_name_prefix='Downloader'
        )
        self.pending_futures = {
            self.executor.submit(self.download_file, f): f for f in files_to_download
        }

        # Determine if we should use TQDM or Standard Logging
        is_gui_mode = self.settings.get('is_gui_mode', False)
        use_tqdm = HAS_UI_LIBS and not is_gui_mode

        futures_iter = as_completed(self.pending_futures)
        
        # Initialize the progress bar correctly
        if use_tqdm:
            futures_iter = tqdm(
                futures_iter, 
                total=total_files, 
                unit="file", 
                desc="Downloading", 
                ncols=90, 
                bar_format='  {l_bar}{bar}{r_bar}'
            )

        try:
            for i, future in enumerate(futures_iter):
                if self._stop_event.is_set():
                    for f in self.pending_futures: f.cancel()
                    break

                # GUI NEEDS these logs to update its progress bar
                if is_gui_mode:
                    logging.info(f"Progress: {i + 1}/{total_files} files processed.")

                original_file_info = self.pending_futures[future]
                try:
                    path, failed_info = future.result()
                    
                    # Handle success INSIDE the loop, where 'path' is defined
                    if path:
                        downloaded.append(path)
                        if use_tqdm:
                            short_name = (Path(path).name[:50] + '..') if len(Path(path).name) > 50 else Path(path).name
                            # Use static tqdm.write to avoid crashing iterator objects
                            tqdm.write(f"  ✔ Downloaded {short_name}")
                        elif is_gui_mode:
                            logging.info(f"Downloaded {Path(path).name}")
                            
                    if failed_info:
                        failed.append(failed_info)
                        
                except Exception as e:
                    failed.append(original_file_info)
                    title = original_file_info.get('title', 'a file')
                    
                    if is_gui_mode:
                        logging.info(f"Failed: An error occurred while processing {title}.")
                    elif use_tqdm:
                        tqdm.write(f"  ✖ Failed: {title}")
                    
                    logging.debug(f"Full error trace for {title}: {e}", exc_info=True)
        
        except Exception as e:
            # Print critical errors to stderr so you can see them even if logs fail
            print(f"\nCRITICAL ERROR in download loop: {e}", file=sys.stderr)
            raise e
            
        finally:
            if self._stop_event.is_set():
                self.shutdown()
            else:
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
    """Sets up and executes a download session (GUI-safe behavior)."""
    is_gui_mode = settings.get('is_gui_mode', False)
    try:
        if settings.get('retry_failed_path'):
            files_to_process = load_config(settings['retry_failed_path'])
            if not files_to_process:
                logging.info("Retry file is empty. Nothing to do.")
                if not is_gui_mode: sys.exit(0)
                return
            logging.info(f"Retrying {len(files_to_process)} files from {settings['retry_failed_path']}")
        else:
            user_intent_params = {
                k: v for k, v in params.items() 
                if k not in ['project', 'latest'] and v
            }
            
            if not user_intent_params and not settings.get('demo'):
                logging.error("No specific search parameters provided. Please specify criteria like --variable, --model, or --experiment.")
                if not is_gui_mode: sys.exit(1)
                return
            
            query_handler = QueryHandler(stop_event=stop_event)
            
            unique_files = query_handler.fetch_datasets(
                params, 
                settings.get('timeout', 30), 
                is_gui_mode=is_gui_mode
            )
            
            if not unique_files:
                if not is_gui_mode: sys.exit(1)
                return
            if stop_event.is_set(): return

            max_downloads = settings.get('max_downloads')
            files_to_process = unique_files[:max_downloads] if max_downloads else unique_files

        metadata_prefix = f"gridflow_{params.get('project', 'cmip5').lower()}_"
        file_manager = FileManager(
            settings['output_dir'], settings['metadata_dir'], settings['save_mode'],
            metadata_prefix=metadata_prefix
        )
        file_manager.save_metadata(files_to_process, "query_results_to_download.json")

        if settings.get('dry_run'):
            logging.info(f"Dry run: Would attempt to download {len(files_to_process)} files.")
            return

        settings.pop('stop_event', None)
        downloader = Downloader(file_manager, stop_event, **settings)
        downloaded, failed = downloader.download_all(files_to_process)
        
        if stop_event.is_set(): logging.warning("Process was stopped before completion.")
        if failed:
            file_manager.save_metadata(failed, "failed_downloads.json")
            logging.warning(f"{len(failed)} downloads failed. Check 'failed_downloads.json' for details.")
        
        logging.info(f"Completed: {downloader.successful_downloads}/{len(files_to_process)} files downloaded successfully.")

    except Exception as e:
        logging.info(f"Failed: A critical error stopped the session. See log file for details.")
        logging.error(f"Full critical error trace for session: {e}", exc_info=True)
        stop_event.set()


def add_arguments(parser):
    """Add CMIP5 downloader arguments to the provided parser."""
    query_group = parser.add_argument_group('Query Parameters (for filtering data)')
    settings_group = parser.add_argument_group('Download & Output Settings')
    mode_group = parser.add_argument_group('Authentication & Special Modes')

    query_group.add_argument("-p", "--project", default="CMIP5", help="Project name (e.g., CMIP5).")
    query_group.add_argument("-m", "--model", help="Model name (e.g., CanCM4, GFDL-CM3).")
    query_group.add_argument("-i", "--institute", help="Modeling institute (e.g., CCCMA).")
    query_group.add_argument("-e", "--experiment", help="Experiment name (e.g., historical, rcp85).")
    query_group.add_argument("--experiment_family", help="Experiment family (e.g., Decadal, RCP).")
    query_group.add_argument("-var", "--variable", help="Variable name (e.g., tas, pr).")
    query_group.add_argument("-f", "--frequency", help="Time frequency (e.g., mon, day). Maps to 'time_frequency'.")
    query_group.add_argument("-r", "--realm", help="Modeling realm (e.g., atmos, ocean).")
    query_group.add_argument("-en", "--ensemble", help="Ensemble member (e.g., r1i1p1).")
    query_group.add_argument("-l", "--latest", action='store_true', help="Only get the latest version of files.")
    query_group.add_argument("-c", "--config", help="Path to JSON config file to pre-fill arguments.")
    
    settings_group.add_argument("-o", "--output-dir", default="./downloads_cmip5", help="Directory to save downloaded files.")
    settings_group.add_argument("-md", "--metadata-dir", default="./metadata_cmip5", help="Directory to save metadata files.")
    settings_group.add_argument("-ld", "--log-dir", default="./gridflow_logs", help="Directory to save log files.")
    settings_group.add_argument("-ll", "--log-level", default="minimal", choices=["minimal", "verbose", "debug"], help="Set logging verbosity for the console.")
    settings_group.add_argument("-sm", "--save-mode", default="structured", choices=["structured", "flat"], help="File saving mode: 'structured' (subdirs) or 'flat' (one dir).")
    settings_group.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel download workers.")
    settings_group.add_argument("-t", "--timeout", type=int, default=30, help="Network request timeout in seconds.")
    settings_group.add_argument("--max-downloads", type=int, help="Maximum number of files to download in this session.")
    settings_group.add_argument("--no-verify-ssl", action="store_true", help="Disable SSL certificate verification (use with caution).")

    mode_group.add_argument("--openid", help="ESGF OpenID (e.g., https://esgf-node.llnl.gov/esgf-idp/openid/username).")
    mode_group.add_argument("--id", help="ESGF username for authentication.")
    mode_group.add_argument("--password", help="ESGF password for authentication.")
    mode_group.add_argument("--retry-failed", help="Path to a 'failed_downloads.json' file to retry.")
    mode_group.add_argument("--dry-run", action="store_true", help="Query for files but do not download them.")
    mode_group.add_argument("--demo", action="store_true", help="Run with a small, pre-defined demo query for CMIP5.")

def main(args=None):
    """Main entry point for the CMIP5 Downloader."""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="CMIP5 Data Downloader", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()

    # FIX: Only set up logging if NOT in GUI mode
    if not getattr(args, 'is_gui_mode', False):
        setup_logging(args.log_dir, args.log_level, prefix="cmip5_downloader")
        signal.signal(signal.SIGINT, signal_handler)

    active_stop_event = getattr(args, 'stop_event', stop_event)

    config = load_config(args.config) if args.config else {}
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    settings = {**config, **cli_args}

    params = {
        'project': settings.get('project'),
        'model': settings.get('model'),
        'institute': settings.get('institute'),
        'experiment': settings.get('experiment'),
        'experiment_family': settings.get('experiment_family'),
        'time_frequency': settings.get('frequency'), 
        'realm': settings.get('realm'),
        'variable': settings.get('variable'),
        'ensemble': settings.get('ensemble'),
    }
    
    if settings.get('latest'): params['latest'] = 'true'
    
    if settings.get('demo'):
        params = {
            'project': 'CMIP5', 'variable': 'tas', 'model': 'CanESM2',
            'experiment': 'historical', 'time_frequency': 'mon',
            'realm': 'atmos', 'ensemble': 'r1i1p1', 'limit': '5'
        }
        settings['max_downloads'] = 5

        demo_cmd = (
            "gridflow cmip5 "
            "-var tas "
            "-m CanESM2 "
            "-e historical "
            "-f mon "
            "-en r1i1p1 "
            "--max-downloads 5"
        )

        if HAS_UI_LIBS and not getattr(args, 'is_gui_mode', False):
            console = Console()
            console.print(f"[bold yellow]Running in demo mode with a pre-defined CMIP5 query.[/]")
            console.print(f"Demo Command:\n  [dim]{demo_cmd}[/dim]\n")
        else:
            logging.info(f"Running in demo mode.\nDemo Command: {demo_cmd}")

    # Validate OpenID authentication
    if settings.get('openid') and not all([settings.get('id'), settings.get('password')]):
        logging.error("Both --id and --password are required when using --openid.")
        if not getattr(args, 'is_gui_mode', False):
            sys.exit(1)
        return

    params = {k: v for k, v in params.items() if v}
    settings['username'] = settings.get('id')
    
    create_download_session(params, settings, active_stop_event)
    
    if active_stop_event.is_set():
        logging.warning("Execution was interrupted.")
        if not getattr(args, 'is_gui_mode', False):
            sys.exit(130)
    
    logging.info("Process finished.")

if __name__ == "__main__":
    main()