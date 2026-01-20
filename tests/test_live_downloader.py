# tests/test_live_downloaders.py
import sys
import shutil
import logging
import threading
import importlib.util
import pytest
from pathlib import Path

# --- Configuration ---
TESTS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = TESTS_DIR.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# --- Helper to Bypass __init__.py Shadowing ---
def load_module_from_file(module_name: str, file_name: str):
    """
    Loads a module directly from a file path, bypassing package __init__.py 
    shadowing issues.
    """
    file_path = PROJECT_ROOT / "gridflow" / "download" / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find module file: {file_path}")
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    
    # Register manually to sys.modules so imports inside it work if needed
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# --- Explicit Module Loading ---
try:
    cmip6_module = load_module_from_file("cmip6_downloader_mod", "cmip6_downloader.py")
    cmip5_module = load_module_from_file("cmip5_downloader_mod", "cmip5_downloader.py")
    prism_module = load_module_from_file("prism_downloader_mod", "prism_downloader.py")
    dem_module = load_module_from_file("dem_downloader_mod", "dem_downloader.py")
    era5_module = load_module_from_file("era5_downloader_mod", "era5_downloader.py")

except Exception as e:
    print(f"CRITICAL: Failed to load gridflow modules from source. Error: {e}")
    sys.exit(1)

# --- Test Configurations ---
TEST_CONFIGS = {
    "cmip6": {
        "module": cmip6_module,
        "params": {
            "project": "CMIP6",
            "source_id": "HadGEM3-GC31-LL",
            "experiment_id": "hist-1950",
            "variable_id": "tas",
            "frequency": "day",
            "variant_label": "r1i1p1f1",
            "limit": "1"
        },
        "settings": {
            "max_downloads": 1,
            "save_mode": "flat",
            "timeout": 30
        }
    },
    "cmip5": {
        "module": cmip5_module,
        "params": {
            "project": "CMIP5",
            "model": "CanESM2",
            "variable": "tas",
            "experiment": "historical",
            "time_frequency": "mon",
            "realm": "atmos",
            "ensemble": "r1i1p1",
            "limit": "1"
        },
        "settings": {
            "max_downloads": 1,
            "save_mode": "flat",
            "timeout": 30
        }
    },
    "prism": {
        "module": prism_module,
        "settings": {
            "variable": ["tmean"],
            "resolution": "4km",
            "time_step": "daily",
            "start_date": "2020-01-01",
            "end_date": "2020-01-01",
            "timeout": 30
        }
    },
    "era5": {
        "module": era5_module,
        "settings": {
            "variables": ["t2m"],
            "start_date": "2021-01-01",
            "end_date": "2021-01-01",
            "timeout": 30
        }
    },
    "dem": {
        "module": dem_module,
        "settings": {
            "bounds": {"north": 42.0, "south": 41.0, "east": -93.0, "west": -94.0},
            "dem_type": "COP30",
            "timeout": 30
        }
    }
}

def run_live_test(name: str, config: dict):
    """
    Orchestrates the setup, execution, and verification of a single downloader.
    """
    print(f"\n🔵 Starting Live Test: {name.upper()}")
    
    module = config["module"]
    output_dir = PROJECT_ROOT / f"temp_test_live_{name}"
    metadata_dir = output_dir / "metadata"
    log_dir = output_dir / "logs"

    # Aggressive pre-test cleanup
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            print(f"⚠️ Warning: Pre-test cleanup failed for {output_dir}: {e}")

    # Ensure directories exist (force absolute paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "output_dir": str(output_dir.resolve()),
        "metadata_dir": str(metadata_dir.resolve()),
        "log_dir": str(log_dir.resolve()),
        "log_level": "debug",
        "workers": 1,
        "is_gui_mode": True,  # Prevents sys.exit()
        "dry_run": False
    }
    settings.update(config.get("settings", {}))
    
    params = config.get("params", {})
    stop_event = threading.Event()

    try:
        # Route to correct function signature
        if name in ["cmip6", "cmip5"]:
            module.create_download_session(params, settings, stop_event)
        elif name in ["prism", "era5", "dem"]:
            module.create_download_session(settings, stop_event)

    except Exception as e:
        print(f"❌ CRITICAL ERROR during execution of {name}: {e}")
        return False
    finally:
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            if isinstance(h, logging.FileHandler) and str(output_dir) in h.baseFilename:
                h.close()
                root_logger.removeHandler(h)

    # --- Verification ---
    print(f"🟡 Verifying output in: {output_dir}")
    
    files = list(output_dir.rglob("*.nc")) + \
            list(output_dir.rglob("*.zip")) + \
            list(output_dir.rglob("*.tif")) 
            
    data_files = [f for f in files if f.parent != metadata_dir and "json" not in f.suffix]

    passed = True
    if not data_files:
        print(f"❌ FAILED: No data files found.")
        failed_log = metadata_dir / "failed_downloads.json"
        if failed_log.exists():
            print(f"   Found failed_downloads.json content (first 500 chars):")
            print(f"   {failed_log.read_text()[:500]}")
        passed = False
    else:
        print(f"✅ PASSED: Found {len(data_files)} file(s).")
        for f in data_files:
            print(f"   - {f.name} ({f.stat().st_size / 1024:.2f} KB)")
            if f.stat().st_size < 100:
                 print(f"⚠️ WARNING: File seems suspiciously small.")

    # --- Cleanup ---
    try:
        if output_dir.exists():
            shutil.rmtree(output_dir)
    except PermissionError:
        print("⚠️ Warning: Could not clean up test directory (files might be in use).")
    except Exception as e:
        print(f"⚠️ Warning: Cleanup error: {e}")

    rogue_meta = PROJECT_ROOT / f"gridflow_{name}_query_results.json"
    if rogue_meta.exists():
        try:
            rogue_meta.unlink()
            print(f"🧹 Cleaned up rogue file in root: {rogue_meta.name}")
        except: pass
        
    return passed

# --- Pytest Entry Points ---

def test_live_cmip6():
    assert run_live_test("cmip6", TEST_CONFIGS["cmip6"])

def test_live_cmip5():
    assert run_live_test("cmip5", TEST_CONFIGS["cmip5"])

def test_live_prism():
    assert run_live_test("prism", TEST_CONFIGS["prism"])

def test_live_era5():
    assert run_live_test("era5", TEST_CONFIGS["era5"])

def test_live_dem():
    assert run_live_test("dem", TEST_CONFIGS["dem"])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    print("="*60)
    print("RUNNING ALL LIVE DOWNLOADER TESTS")
    print("="*60)

    results = {}
    for name, config in TEST_CONFIGS.items():
        results[name] = run_live_test(name, config)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name.upper():<10} : {status}")
        if not passed: all_passed = False
    
    sys.exit(0 if all_passed else 1)