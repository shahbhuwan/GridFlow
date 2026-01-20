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
import signal
import logging
import threading
import warnings
from pathlib import Path
from typing import List, Dict, Any

# --- Dependency Check ---
try:
    import netCDF4 as nc
except ImportError as e:
    print(f"FATAL: Missing required library: {e.name}. Please install it using 'pip install {e.name}'.")
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

# --- Suppress known warnings ---
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Global Stop Event ---
stop_event = threading.Event()

def signal_handler(sig, frame):
    """Handles SIGINT (Ctrl+C) to gracefully shut down the application."""
    logging.warning("Stop signal received! Gracefully shutting down...")
    stop_event.set()
    logging.info("Please wait for ongoing tasks to complete.")

def extract_metadata_from_file(file_path: Path) -> Dict:
    """Extracts required metadata from a single NetCDF file."""
    try:
        with nc.Dataset(file_path, 'r') as ds:
            metadata = {
                "activity_id": getattr(ds, "activity_id", None),
                "source_id": getattr(ds, "source_id", None),
                "variant_label": getattr(ds, "variant_label", None),
                "variable_id": getattr(ds, "variable_id", None),
                "institution_id": getattr(ds, "institution_id", None)
            }
        return {"file_path": str(file_path), "metadata": metadata, "error": None}
    except Exception as e:
        return {"file_path": str(file_path), "metadata": {}, "error": f"Failed to read or extract metadata: {e}"}

class Cataloger:
    """Encapsulates the logic for generating a data catalog from NetCDF files."""

    def __init__(self, settings: Dict[str, Any], stop_event: threading.Event):
        self.settings = settings
        self._stop_event = stop_event
        self.catalog = {}
        self.duplicates = []
        self.skipped_count = 0
        self.included_count = 0

    def find_and_deduplicate_files(self) -> List[Path]:
        """Recursively finds all .nc files and handles duplicates with UI feedback."""
        input_dir = Path(self.settings['input_dir'])
        is_gui_mode = self.settings.get('is_gui_mode', False)
        use_rich = HAS_UI_LIBS and not is_gui_mode
        
        status_context = None
        if use_rich:
            console = Console()
            status_context = console.status(f"[bold green]Searching for NetCDF files in {input_dir}...", spinner="dots")
            status_context.start()
        else:
            logging.info(f"Searching for NetCDF files in {input_dir}...")

        try:
            all_files = list(input_dir.rglob("*.nc"))
            
            if not all_files:
                return []

            # Group files by a base name to find duplicates
            files_by_base_name = {}
            for f_path in all_files:
                base_name = self._get_base_filename(f_path.name)
                if base_name not in files_by_base_name:
                    files_by_base_name[base_name] = []
                files_by_base_name[base_name].append(f_path)
                
            unique_files = []
            for base_name, file_list in files_by_base_name.items():
                if self._stop_event.is_set(): break
                
                if len(file_list) == 1:
                    unique_files.append(file_list[0])
                else:
                    # Prefer non-prefixed filenames if multiple exist
                    preferred_file = next((f for f in file_list if self._is_non_prefixed(f.name)), file_list[0])
                    unique_files.append(preferred_file)
                    for f in file_list:
                        if f != preferred_file:
                            self.duplicates.append(str(f))
                            logging.debug(f"Duplicate found for base name '{base_name}': Using '{preferred_file.name}', ignoring '{f.name}'.")
            
            return unique_files
            
        finally:
            if status_context:
                status_context.stop()

    def generate(self):
        """Main method to orchestrate the catalog generation process."""
        unique_files = self.find_and_deduplicate_files()
        
        if not unique_files:
            msg = f"No NetCDF files found in {self.settings['input_dir']}."
            if self.settings.get('demo'):
                logging.warning(f"{msg} Run a downloader script with '--demo' to get sample files.")
            else:
                logging.warning(msg)
            return

        is_gui_mode = self.settings.get('is_gui_mode', False)
        use_tqdm = HAS_UI_LIBS and not is_gui_mode
        total_files = len(unique_files)

        if use_tqdm:
            Console().print(f"[bold blue]Cataloging:[/] Found {total_files} unique files to process.")
        else:
            logging.info(f"Found {total_files} unique files to process.")
        
        # Setup iterator
        files_iter = unique_files
        if use_tqdm:
            files_iter = tqdm(
                unique_files, 
                total=total_files, 
                unit="file", 
                desc="Processing", 
                ncols=90, 
                bar_format='  {l_bar}{bar}{r_bar}'
            )

        for i, file_path in enumerate(files_iter):
            if self._stop_event.is_set():
                logging.info("Catalog generation stopped by user.")
                break

            if is_gui_mode:
                logging.info(f"Progress: {i + 1}/{total_files} files processed.")

            result = extract_metadata_from_file(file_path)
            success = self._process_metadata_result(result)
            
            if not success and use_tqdm:
                tqdm.write(f"  ✖ Skipped {file_path.name}")
            elif not success:
                logging.warning(f"Skipped {file_path.name}")

        self._save_results()

    def _process_metadata_result(self, result: Dict) -> bool:
        """Processes a single metadata dictionary. Returns True if included, False if skipped."""
        file_path = result["file_path"]
        metadata = result["metadata"]
        error = result.get("error")

        if error:
            logging.debug(f"Skipping {Path(file_path).name}: {error}")
            self.skipped_count += 1
            return False

        required_keys = ["activity_id", "source_id", "variant_label", "variable_id"]
        if not all(metadata.get(key) for key in required_keys):
            missing = [key for key in required_keys if not metadata.get(key)]
            logging.debug(f"Skipping {Path(file_path).name}: Incomplete metadata (missing: {', '.join(missing)}).")
            self.skipped_count += 1
            return False
            
        # Build catalog key
        key = f"{metadata['activity_id']}:{metadata['source_id']}:{metadata['variant_label']}"
        
        if key not in self.catalog:
            self.catalog[key] = {
                "activity_id": metadata["activity_id"],
                "source_id": metadata["source_id"],
                "variant_label": metadata["variant_label"],
                "institution_id": metadata.get("institution_id", ""),
                "variables": {}
            }
            
        variable_id = metadata["variable_id"]

        if variable_id not in self.catalog[key]["variables"]:
            self.catalog[key]["variables"][variable_id] = {
                "file_count": 0,
                "files": []
            }
        
        self.catalog[key]["variables"][variable_id]["files"].append(str(file_path))
        self.catalog[key]["variables"][variable_id]["file_count"] += 1
        self.included_count += 1
        return True

    def _save_results(self):
        """Saves the generated catalog and duplicates list to JSON files."""
        output_dir = Path(self.settings['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        catalog_filename = "cmip6_catalog.json" if self.settings.get('demo') else "catalog.json"
        catalog_path = output_dir / catalog_filename
        
        try:
            with open(catalog_path, 'w', encoding='utf-8') as f:
                json.dump(self.catalog, f, indent=2)
            logging.info(f"Catalog saved to {catalog_path}")
        except IOError as e:
            logging.error(f"Failed to save catalog: {e}")

        if self.duplicates:
            duplicates_path = output_dir / "duplicates.json"
            try:
                with open(duplicates_path, 'w', encoding='utf-8') as f:
                    json.dump(self.duplicates, f, indent=2)
                logging.info(f"List of duplicate files saved to {duplicates_path}")
            except IOError as e:
                logging.error(f"Failed to save duplicates list: {e}")
        
        summary = f"Summary: Included {self.included_count} files in catalog. Skipped {self.skipped_count} files."
        if self.duplicates:
            summary += f" Found {len(self.duplicates)} duplicates."
        
        is_gui_mode = self.settings.get('is_gui_mode', False)
        if HAS_UI_LIBS and not is_gui_mode:
             Console().print(f"[bold green]{summary}[/]")
        else:
            logging.info(summary)

    @staticmethod
    def _is_non_prefixed(filename: str) -> bool:
        """Determines if a filename is non-prefixed."""
        cmip_vars = {'tas', 'pr', 'huss', 'psl', 'ts', 'uas', 'vas'}
        return any(filename.startswith(f"{v}_") for v in cmip_vars)

    @staticmethod
    def _get_base_filename(filename: str) -> str:
        """Extracts the base filename by removing known prefixes."""
        prefixes = ['ScenarioMIP_250km_', 'CMIP6_', 'CMIP5_']
        for prefix in prefixes:
            if filename.startswith(prefix):
                return filename[len(prefix):]
        return filename

def run_catalog_session(settings: Dict[str, Any], stop_event: threading.Event):
    """Sets up and executes a catalog generation session."""
    is_gui_mode = settings.get('is_gui_mode', False)
    
    try:
        # Validate Inputs
        input_dir = Path(settings['input_dir'])
        if not input_dir.is_dir():
             logging.error(f"Input directory not found: {input_dir}")
             if not is_gui_mode: sys.exit(1)
             return

        cataloger = Cataloger(settings, stop_event)
        cataloger.generate()

        if stop_event.is_set():
            logging.warning("Process was stopped before completion.")

    except Exception as e:
        logging.info(f"Failed: A critical error occurred. See log file for details.")
        logging.debug(f"Full critical error trace: {e}", exc_info=True)
        stop_event.set()

def add_arguments(parser):
    """Add catalog generation arguments to the provided parser."""
    io_group = parser.add_argument_group('Input and Output')
    settings_group = parser.add_argument_group('Processing Settings')

    io_group.add_argument("-i", "--input_dir", help="Directory containing NetCDF files (searched recursively).")
    io_group.add_argument("-o", "--output_dir", help="Directory to save catalog.json.")

    settings_group.add_argument("-ld", "--log-dir", default="./gridflow_logs", help="Log directory.")
    settings_group.add_argument("-ll", "--log-level", default="minimal", choices=["minimal", "verbose", "debug"], help="Log verbosity.")
    settings_group.add_argument("--demo", action="store_true", help="Run with demo defaults.")

def main(args=None):
    """Main entry point with Rich-enhanced reporting."""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="NetCDF Catalog Generator", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_arguments(parser)
        args = parser.parse_args()
    
    # FIX: Only set up logging if NOT in GUI mode
    if not getattr(args, 'is_gui_mode', False):
        setup_logging(args.log_dir, args.log_level, prefix="catalog_generator")
        signal.signal(signal.SIGINT, signal_handler)

    active_stop_event = getattr(args, 'stop_event', stop_event)
    
    settings = vars(args)
    is_gui_mode = settings.get('is_gui_mode', False)

    if args.demo:
        settings['input_dir'] = './downloads_cmip6'
        settings['output_dir'] = './catalog_cmip6'
        
        demo_cmd = "gridflow catalog -i ./downloads_cmip6 -o ./catalog_cmip6"

        if HAS_UI_LIBS and not is_gui_mode:
            console = Console()
            console.print(f"[bold yellow]Running in demo mode.[/]")
            console.print(f"Demo Command:\n  [dim]{demo_cmd}[/dim]\n")
        else:
            logging.info(f"Running in demo mode.\nDemo Command: {demo_cmd}")
    else:
        if not all([settings.get('input_dir'), settings.get('output_dir')]):
            logging.error("Required arguments missing: --input_dir and --output_dir (or use --demo)")
            if not is_gui_mode: sys.exit(1)
            return

    run_catalog_session(settings, active_stop_event)
    
    if active_stop_event.is_set():
        logging.info("Execution was interrupted.")
        if not is_gui_mode: sys.exit(130)
    
    logging.info("Process finished.")

if __name__ == "__main__":
    main()