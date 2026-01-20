# gridflow/download/era5_downloader.py
# Copyright (c) 2025 Bhuwan Shah

import sys
import json
import time
import signal
import logging
import threading
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import List, Dict, Optional, Tuple, Any, Union

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
    from rich.table import Table
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
AWS_BUCKET_NAME = 'nsf-ncar-era5'
AWS_REGION = 'us-west-2'

# --- Variable Mapping ---
# Maps User Short Codes -> AWS S3 Folder & File Codes

VARIABLE_MAP = {
    # ==============================
    # 1. TEMPERATURE & HEAT
    # ==============================
    "t2m": {"folder": "e5.oper.an.sfc", "code": "128_167_2t", "desc": "2m Air Temperature"},
    "2m_temperature": {"folder": "e5.oper.an.sfc", "code": "128_167_2t", "desc": "2m Air Temperature"},
    "d2m": {"folder": "e5.oper.an.sfc", "code": "128_168_2d", "desc": "2m Dewpoint Temperature"},
    "skt": {"folder": "e5.oper.an.sfc", "code": "128_235_skt", "desc": "Skin Temperature"},
    "sst": {"folder": "e5.oper.an.sfc", "code": "128_034_sstk", "desc": "Sea Surface Temperature"},
    
    # Soil & Ice Temps
    "stl1": {"folder": "e5.oper.an.sfc", "code": "128_139_stl1", "desc": "Soil Temperature Level 1"},
    "stl2": {"folder": "e5.oper.an.sfc", "code": "128_170_stl2", "desc": "Soil Temperature Level 2"},
    "stl3": {"folder": "e5.oper.an.sfc", "code": "128_183_stl3", "desc": "Soil Temperature Level 3"},
    "stl4": {"folder": "e5.oper.an.sfc", "code": "128_236_stl4", "desc": "Soil Temperature Level 4"},
    "istl1": {"folder": "e5.oper.an.sfc", "code": "128_035_istl1", "desc": "Ice Temperature Layer 1"},
    "istl2": {"folder": "e5.oper.an.sfc", "code": "128_036_istl2", "desc": "Ice Temperature Layer 2"},
    "istl3": {"folder": "e5.oper.an.sfc", "code": "128_037_istl3", "desc": "Ice Temperature Layer 3"},
    "istl4": {"folder": "e5.oper.an.sfc", "code": "128_038_istl4", "desc": "Ice Temperature Layer 4"},
    "tsn": {"folder": "e5.oper.an.sfc", "code": "128_238_tsn", "desc": "Temperature of Snow Layer"},

    # ==============================
    # 2. PRECIPITATION & WATER
    # ==============================
    # Composite: Total Precip (Convective + Large Scale)
    "precip": [
        {"folder": "e5.oper.fc.sfc.accumu", "code": "128_143_cp", "desc": "Convective Precipitation (1/2)"},
        {"folder": "e5.oper.fc.sfc.accumu", "code": "128_142_lsp", "desc": "Large Scale Precipitation (2/2)"}
    ],
    "total_precipitation": [
        {"folder": "e5.oper.fc.sfc.accumu", "code": "128_143_cp", "desc": "Convective Precipitation (1/2)"},
        {"folder": "e5.oper.fc.sfc.accumu", "code": "128_142_lsp", "desc": "Large Scale Precipitation (2/2)"}
    ],
    
    "cp": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_143_cp", "desc": "Convective Precipitation"},
    "lsp": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_142_lsp", "desc": "Large Scale Precipitation"},
    "csf": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_239_csf", "desc": "Convective Snowfall"},
    "lsf": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_240_lsf", "desc": "Large Scale Snowfall"},
    "sf": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_144_sf", "desc": "Snowfall"},
    
    # Water Vapor & Clouds
    "tcw": {"folder": "e5.oper.an.sfc", "code": "128_136_tcw", "desc": "Total Column Water"},
    "tcwv": {"folder": "e5.oper.an.sfc", "code": "128_137_tcwv", "desc": "Total Column Water Vapour"},
    "tclw": {"folder": "e5.oper.an.sfc", "code": "128_078_tclw", "desc": "Total Column Cloud Liquid Water"},
    "tciw": {"folder": "e5.oper.an.sfc", "code": "128_079_tciw", "desc": "Total Column Cloud Ice Water"},
    "tcrw": {"folder": "e5.oper.an.sfc", "code": "228_089_tcrw", "desc": "Total Column Rain Water"},
    "tcsw": {"folder": "e5.oper.an.sfc", "code": "228_090_tcsw", "desc": "Total Column Snow Water"},
    "tcc": {"folder": "e5.oper.an.sfc", "code": "128_164_tcc", "desc": "Total Cloud Cover"},
    "lcc": {"folder": "e5.oper.an.sfc", "code": "128_186_lcc", "desc": "Low Cloud Cover"},
    "mcc": {"folder": "e5.oper.an.sfc", "code": "128_187_mcc", "desc": "Medium Cloud Cover"},
    "hcc": {"folder": "e5.oper.an.sfc", "code": "128_188_hcc", "desc": "High Cloud Cover"},

    # Evaporation & Runoff
    "e": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_182_e", "desc": "Evaporation"},
    "pev": {"folder": "e5.oper.fc.sfc.accumu", "code": "228_251_pev", "desc": "Potential Evaporation"},
    "es": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_044_es", "desc": "Snow Evaporation"},
    "smlt": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_045_smlt", "desc": "Snowmelt"},
    "ro": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_205_ro", "desc": "Runoff"},
    "sro": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_008_sro", "desc": "Surface Runoff"},
    "ssro": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_009_ssro", "desc": "Sub-surface Runoff"},

    # Soil Water
    "swvl1": {"folder": "e5.oper.an.sfc", "code": "128_039_swvl1", "desc": "Volumetric Soil Water Layer 1"},
    "swvl2": {"folder": "e5.oper.an.sfc", "code": "128_040_swvl2", "desc": "Volumetric Soil Water Layer 2"},
    "swvl3": {"folder": "e5.oper.an.sfc", "code": "128_041_swvl3", "desc": "Volumetric Soil Water Layer 3"},
    "swvl4": {"folder": "e5.oper.an.sfc", "code": "128_042_swvl4", "desc": "Volumetric Soil Water Layer 4"},

    # ==============================
    # 3. WIND & PRESSURE
    # ==============================
    "u10": {"folder": "e5.oper.an.sfc", "code": "128_165_10u", "desc": "10m U-Component of Wind"},
    "v10": {"folder": "e5.oper.an.sfc", "code": "128_166_10v", "desc": "10m V-Component of Wind"},
    "100u": {"folder": "e5.oper.an.sfc", "code": "228_246_100u", "desc": "100m U-Component of Wind"},
    "100v": {"folder": "e5.oper.an.sfc", "code": "228_247_100v", "desc": "100m V-Component of Wind"},
    "sp": {"folder": "e5.oper.an.sfc", "code": "128_134_sp", "desc": "Surface Pressure"},
    "msl": {"folder": "e5.oper.an.sfc", "code": "128_151_msl", "desc": "Mean Sea Level Pressure"},
    
    # Stress (Instantaneous/Accumulated)
    "iews": {"folder": "e5.oper.an.sfc", "code": "128_229_iews", "desc": "Instantaneous Eastward Wind Stress"},
    "inss": {"folder": "e5.oper.an.sfc", "code": "128_230_inss", "desc": "Instantaneous Northward Wind Stress"},
    "ewss": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_180_ewss", "desc": "Eastward Turbulent Surface Stress"},
    "nsss": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_181_nsss", "desc": "Northward Turbulent Surface Stress"},

    # ==============================
    # 4. RADIATION & FLUXES
    # ==============================
    # Solar (Shortwave)
    "ssrd": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_169_ssrd", "desc": "Surface Solar Radiation Downwards"},
    "ssrdc": {"folder": "e5.oper.fc.sfc.accumu", "code": "228_129_ssrdc", "desc": "Surface Solar Radiation Downwards, Clear Sky"},
    "ssr": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_176_ssr", "desc": "Surface Net Solar Radiation"},
    "ssrc": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_210_ssrc", "desc": "Surface Net Solar Radiation, Clear Sky"},
    "tsr": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_178_tsr", "desc": "Top Net Solar Radiation"},
    "tisr": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_212_tisr", "desc": "TOA Incident Solar Radiation"},
    "fdir": {"folder": "e5.oper.fc.sfc.accumu", "code": "228_021_fdir", "desc": "Total Sky Direct Solar Radiation"},
    "cdir": {"folder": "e5.oper.fc.sfc.accumu", "code": "228_022_cdir", "desc": "Clear-sky Direct Solar Radiation"},
    "fal": {"folder": "e5.oper.an.sfc", "code": "128_243_fal", "desc": "Forecast Albedo"},

    # Thermal (Longwave)
    "strd": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_175_strd", "desc": "Surface Thermal Radiation Downwards"},
    "strdc": {"folder": "e5.oper.fc.sfc.accumu", "code": "228_130_strdc", "desc": "Surface Thermal Radiation Downwards, Clear Sky"},
    "str": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_177_str", "desc": "Surface Net Thermal Radiation"},
    "strc": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_211_strc", "desc": "Surface Net Thermal Radiation, Clear Sky"},
    "ttr": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_179_ttr", "desc": "Top Net Thermal Radiation"},

    # Heat Fluxes
    "sshf": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_146_sshf", "desc": "Surface Sensible Heat Flux"},
    "slhf": {"folder": "e5.oper.fc.sfc.accumu", "code": "128_147_slhf", "desc": "Surface Latent Heat Flux"},
    "ishf": {"folder": "e5.oper.an.sfc", "code": "128_231_ishf", "desc": "Instantaneous Surface Sensible Heat Flux"},
    "ie": {"folder": "e5.oper.an.sfc", "code": "128_232_ie", "desc": "Instantaneous Moisture Flux (Evaporation)"},

    # ==============================
    # 5. OTHER SURFACE / VEGETATION
    # ==============================
    "lsm": {"folder": "e5.oper.an.sfc", "code": "128_172_lsm", "desc": "Land-Sea Mask"},
    "sd": {"folder": "e5.oper.an.sfc", "code": "128_141_sd", "desc": "Snow Depth"},
    "asn": {"folder": "e5.oper.an.sfc", "code": "128_032_asn", "desc": "Snow Albedo"},
    "rsn": {"folder": "e5.oper.an.sfc", "code": "128_033_rsn", "desc": "Snow Density"},
    "ci": {"folder": "e5.oper.an.sfc", "code": "128_031_ci", "desc": "Sea Ice Cover"},
    "lailv": {"folder": "e5.oper.an.sfc", "code": "128_066_lailv", "desc": "Leaf Area Index, Low Vegetation"},
    "laihv": {"folder": "e5.oper.an.sfc", "code": "128_067_laihv", "desc": "Leaf Area Index, High Vegetation"},
    "cape": {"folder": "e5.oper.an.sfc", "code": "128_059_cape", "desc": "Convective Available Potential Energy"},
    "blh": {"folder": "e5.oper.an.sfc", "code": "128_159_blh", "desc": "Boundary Layer Height"},
    "tco3": {"folder": "e5.oper.an.sfc", "code": "128_206_tco3", "desc": "Total Column Ozone"},
    "fsr": {"folder": "e5.oper.an.sfc", "code": "128_244_fsr", "desc": "Forecast Surface Roughness"},
}

ERA5_VARIABLES = list(VARIABLE_MAP.keys())

# Predefined Areas of Interest (N, S, E, W)
AOI_BOUNDS = {
    "corn_belt": [49.5, 35.8, -80.4, -104.5], 
    "iowa": [43.5, 40.3, -90.1, -96.7],
    "conus": [49.38, 24.52, -66.93, -124.78]
}

# --- Global Stop Event ---
stop_event = threading.Event()

def signal_handler(sig, frame):
    """Handles SIGINT (Ctrl+C)."""
    logging.info("Stop signal received! Gracefully shutting down...")
    stop_event.set()

class FileManager:
    """Handles file and directory management for ERA5 data."""
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
        filename = file_info.get('filename')
        return self.download_dir / filename

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
    """Handles generating and validating AWS S3 keys for ERA5."""
    def __init__(self, stop_event: threading.Event):
        self._stop_event = stop_event
        self.s3 = boto3.client('s3', region_name=AWS_REGION, config=Config(signature_version=UNSIGNED))

    def generate_potential_files(self, variables: List[str], start_date: str, end_date: str) -> List[Dict]:
        """Generates list of S3 keys based on date range and variable mapping."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        potential_files = []
        
        for var_name in variables:
            # Splits "t2m (2m Temp)" -> "t2m"
            var_clean = var_name.split('(')[0].strip().lower()
            
            if var_clean not in VARIABLE_MAP:
                logging.warning(f"Variable '{var_clean}' not found in mapping. Use --list-variables to see valid options. Skipping.")
                continue
            
            # Handle both single dict and list of dicts (for composite variables like precip)
            entry = VARIABLE_MAP[var_clean]
            metas = entry if isinstance(entry, list) else [entry]
            
            for meta in metas:
                current = start.replace(day=1)
                while current <= end:
                    year_month = current.strftime("%Y%m")
                    prefix = f"{meta['folder']}/{year_month}/{meta['folder']}.{meta['code']}."
                    
                    potential_files.append({
                        'variable_user': var_clean,
                        'description': meta.get('desc', ''),
                        'year': current.year,
                        'month': current.month,
                        'prefix': prefix,
                        's3_bucket': AWS_BUCKET_NAME
                    })
                    current += relativedelta(months=1)
                
        return potential_files

    def validate_files_on_s3(self, potential_files: List[Dict], is_gui_mode: bool = False) -> List[Dict]:
        """Checks S3 to find the exact filename (timestamps differ) and verifies existence."""
        valid_files = []
        use_rich = HAS_UI_LIBS and not is_gui_mode
        status_context = None

        if use_rich:
            console = Console()
            status_context = console.status(f"[bold green]Querying AWS S3 for {len(potential_files)} monthly files...", spinner="dots")
            status_context.start()
        else:
            logging.info(f"Querying AWS S3 for {len(potential_files)} monthly files...")

        try:
            with ThreadPoolExecutor(max_workers=10, thread_name_prefix="S3Checker") as executor:
                future_to_task = {executor.submit(self._find_s3_key, p): p for p in potential_files}
                
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
            Console().print(f"[bold blue]Query complete![/] Found {len(valid_files)} available files.")
        else:
            logging.info(f"Query complete! Found {len(valid_files)} available files.")
            
        return valid_files

    def _find_s3_key(self, task: Dict) -> Optional[Dict]:
        if self._stop_event.is_set(): return None
        try:
            # List objects with prefix to find specific file (start/end hours in filename vary)
            response = self.s3.list_objects_v2(
                Bucket=task['s3_bucket'], 
                Prefix=task['prefix'], 
                MaxKeys=1
            )
            
            if 'Contents' in response:
                s3_key = response['Contents'][0]['Key']
                file_size = response['Contents'][0]['Size']
                
                return {
                    'title': s3_key.split('/')[-1],
                    'filename': s3_key.split('/')[-1],
                    's3_key': s3_key,
                    's3_bucket': task['s3_bucket'],
                    'size_bytes': file_size,
                    'variable': task['variable_user'],
                    'year': task['year'],
                    'month': task['month']
                }
            return None
        except ClientError:
            return None

class Downloader:
    """Manages the download process from S3."""
    def __init__(self, file_manager: FileManager, stop_event: threading.Event, **settings):
        self.file_manager = file_manager
        self._stop_event = stop_event
        self.settings = settings
        self.successful_downloads = 0
        self.executor = None
        self.s3 = boto3.client('s3', region_name=AWS_REGION, config=Config(signature_version=UNSIGNED))

    def shutdown(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        logging.info("Downloader has been shut down.")

    def download_file(self, file_info: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        if self._stop_event.is_set(): return None, file_info
        
        output_path = self.file_manager.get_output_path(file_info)
        file_info['output_path'] = output_path 
        
        if output_path.exists():
            if output_path.stat().st_size == file_info['size_bytes']:
                logging.debug(f"Skipping {output_path.name} - already exists.")
                return str(output_path), None
        
        try:
            logging.debug(f"Downloading {file_info['s3_key']}")
            self.s3.download_file(file_info['s3_bucket'], file_info['s3_key'], str(output_path))
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
        self.executor = ThreadPoolExecutor(max_workers=self.settings.get('workers', 4), thread_name_prefix='Downloader')
        
        future_to_file = {self.executor.submit(self.download_file, f): f for f in files_to_download}

        is_gui_mode = self.settings.get('is_gui_mode', False)
        use_tqdm = HAS_UI_LIBS and not is_gui_mode
        futures_iter = as_completed(future_to_file)
        
        if use_tqdm:
            futures_iter = tqdm(futures_iter, total=total_files, unit="file", desc="Downloading", ncols=90, bar_format='  {l_bar}{bar}{r_bar}')

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
            self.shutdown()

        self.successful_downloads = len(downloaded)
        return downloaded, failed

def print_variables_table():
    """Prints a table of available variables."""
    if HAS_UI_LIBS:
        console = Console()
        table = Table(title="Available ERA5 Variables (AWS S3)", show_header=True, header_style="bold magenta")
        table.add_column("Short Code", style="cyan", width=15)
        table.add_column("Description", style="white")
        table.add_column("AWS File Code", style="dim")

        for key, value in VARIABLE_MAP.items():
            if isinstance(value, list):
                # Handle composite variables like precip
                desc = " & ".join([v.get('desc', '') for v in value])
                codes = " + ".join([v.get('code', '') for v in value])
                table.add_row(key, f"[Composite] {desc}", codes)
            else:
                table.add_row(key, value.get('desc', 'N/A'), value.get('code', 'N/A'))
        
        console.print(table)
    else:
        print(f"{'Short Code':<20} | {'Description':<40} | {'AWS Code'}")
        print("-" * 80)
        for key, value in VARIABLE_MAP.items():
            if isinstance(value, list):
                desc = "Composite (e.g. " + value[0].get('desc', '') + ")"
                print(f"{key:<20} | {desc:<40} | Multiple")
            else:
                print(f"{key:<20} | {value.get('desc', ''):<40} | {value.get('code', '')}")

def load_config(config_path: str) -> Dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        return {}

def create_download_session(settings: Dict[str, Any], stop_event: threading.Event) -> None:
    is_gui_mode = settings.get('is_gui_mode', False)
    try:
        if settings.get('retry_failed_path'):
            files_to_process = load_config(settings['retry_failed_path'])
        else:
            raw_vars = settings.get('variables', '')
            variables = [v.strip() for v in raw_vars.split(',') if v.strip()] if isinstance(raw_vars, str) else raw_vars

            if not variables and not settings.get('demo'):
                logging.error("No variables specified. Use -var to specify variables or --list-variables to see options.")
                if not is_gui_mode: sys.exit(1)
                return

            query_handler = QueryHandler(stop_event=stop_event)
            potential_files = query_handler.generate_potential_files(variables, settings['start_date'], settings['end_date'])
            
            if not potential_files:
                logging.error("No valid variables found in mapping. Use --list-variables to see valid codes.")
                if not is_gui_mode: sys.exit(1)
                return

            files_to_process = query_handler.validate_files_on_s3(potential_files, is_gui_mode=is_gui_mode)

        if not files_to_process:
            logging.info("No available files found on S3 for criteria.")
            if not is_gui_mode: sys.exit(0)
            return

        file_manager = FileManager(settings['output_dir'], settings['metadata_dir'], "gridflow_era5_")
        file_manager.save_metadata(files_to_process, "query_results.json")

        if settings.get('dry_run'):
            logging.info(f"Dry run: Would attempt to download {len(files_to_process)} files.")
            return

        settings.pop('stop_event', None)
        settings.pop('stop_flag', None)

        downloader = Downloader(file_manager, stop_event, **settings)
        downloaded, failed = downloader.download_all(files_to_process)

        if stop_event.is_set(): logging.warning("Process was stopped before completion.")
        if failed:
            file_manager.save_metadata(failed, "failed_downloads.json")
            logging.warning(f"{len(failed)} downloads failed. Check 'failed_downloads.json' for details.")
        
        logging.info(f"Completed: {downloader.successful_downloads}/{len(files_to_process)} files downloaded successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    query_group = parser.add_argument_group('Query Parameters')
    settings_group = parser.add_argument_group('Download Settings')

    query_group.add_argument("-var", "--variables", help="Comma-separated variables (e.g., t2m, precip, u10).")
    query_group.add_argument("-sd", "--start-date", help="Start date (YYYY-MM-DD).")
    query_group.add_argument("-ed", "--end-date", help="End date (YYYY-MM-DD).")
    query_group.add_argument("-lv", "--list-variables", action="store_true", help="List all available variables and their descriptions.")
    
    settings_group.add_argument("-o", "--output-dir", default="./downloads_era5", help="Output directory.")
    settings_group.add_argument("-md", "--metadata-dir", default="./metadata_era5", help="Metadata directory.")
    settings_group.add_argument("-ld", "--log-dir", default="./gridflow_logs", help="Log directory.")
    settings_group.add_argument("-ll", "--log-level", default="minimal", choices=["minimal", "verbose", "debug"], help="Log verbosity.")
    settings_group.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel workers.")
    settings_group.add_argument("--dry-run", action="store_true", help="Find files but do not download.")
    settings_group.add_argument("--demo", action="store_true", help="Run with demo settings.")
    settings_group.add_argument("-c", "--config", help="Path to JSON config file.")
    settings_group.add_argument("--retry-failed", help="Path to failed_downloads.json to retry.")

def main(args=None):
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="ERA5 (AWS S3) Downloader", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()

    # Handle List Variables immediately
    if getattr(args, 'list_variables', False):
        print_variables_table()
        sys.exit(0)

    # FIX: Only set up logging if NOT in GUI mode
    if not getattr(args, 'is_gui_mode', False):
        setup_logging(args.log_dir, args.log_level, prefix="era5_downloader")
        signal.signal(signal.SIGINT, signal_handler)

    active_stop_event = getattr(args, 'stop_event', stop_event)

    config = load_config(args.config) if args.config else {}
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    settings = {**config, **cli_args}

    if settings.get('demo'):
        settings.update({
            'variables': ['t2m', 'precip'],
            'start_date': '2021-01-01',
            'end_date': '2021-03-01'
        })
        
        demo_cmd = "gridflow era5 --variables t2m,precip --start-date 2021-01-01 --end-date 2021-03-01"

        if HAS_UI_LIBS and not getattr(args, 'is_gui_mode', False):
            console = Console()
            console.print(f"[bold yellow]Running in demo mode (AWS S3 source).[/]")
            console.print(f"Demo Command:\n  [dim]{demo_cmd}[/dim]\n")
        else:
            logging.info(f"Running in demo mode.\nDemo Command: {demo_cmd}")

    else:
        if not all([settings.get('variables'), settings.get('start_date'), settings.get('end_date')]):
            logging.error("Required arguments missing: --variables, --start-date, --end-date")
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