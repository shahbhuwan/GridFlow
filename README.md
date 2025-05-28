```markdown
# GridFlow

GridFlow is a Python library and GUI application for downloading and processing CMIP5, CMIP6, and PRISM climate data. It supports concurrent downloading of CMIP5 and CMIP6 datasets from ESGF nodes and PRISM data from Oregon State University, cropping and clipping NetCDF files to specific geographic regions, generating metadata catalogs, and running batch processes. Designed for researchers and data scientists, GridFlow offers a user-friendly graphical interface (GUI) and a powerful command-line interface (CLI) with error handling, parallel processing, and robust retry mechanisms.

## Features

- **Download CMIP5/CMIP6**: Retrieve climate data from ESGF nodes with customizable query parameters (e.g., institute, variable, model, experiment, activity, resolution).
- **Download PRISM**: Access daily or monthly PRISM climate data for the contiguous U.S. at 4km or 800m resolution.
- **Crop**: Crop NetCDF files to a specified latitude/longitude bounding box with optional buffering.
- **Clip**: Clip NetCDF files using a shapefile for precise geographic regions.
- **Catalog**: Generate a JSON catalog grouping NetCDF files by activity, source, and variant.
- **Batch**: Execute multiple commands from a JSON configuration file.
- **GUI Interface**: Intuitive graphical interface for configuring and running download, crop, clip, and catalog operations.
- **Dual-Mode Executable**: Run as a GUI with a double-click or as a CLI from the command line (Windows).
- **Parallel Processing**: Utilize multiple threads for efficient downloading and processing.
- **Error Handling**: Automatic retries, checksum verification, and detailed logging for robust operations.
- **Testing**: Comprehensive test suite using pytest to ensure reliability.

## Installation

### Prerequisites

- **Python Version**: Python 3.8 or higher (required for `pip` installation).
- **System Libraries** (for `pip` installation):
  ```bash
  # Ubuntu
  sudo apt-get install libgeos-dev libgdal-dev libhdf5-dev

  # macOS
  brew install geos gdal hdf5

  # Windows (use a package manager like Chocolatey)
  choco install hdf5
  ```

### Install via pip (Recommended for Developers)
Install GridFlow using the provided wheel (`whl`) file or from source for CLI and GUI access.

#### Using Pre-built Wheel
Download the wheel file from the [GitHub Releases](https://github.com/shahbhuwan/GridFlow/releases) page and install:
```bash
pip install gridflow-1.0-py3-none-any.whl
```

#### From Source
Clone the repository and install:
```bash
git clone https://github.com/shahbhuwan/GridFlow.git
cd GridFlow
python -m venv gridflow-env
source gridflow-env/bin/activate  # On Windows: gridflow-env\Scripts\activate
pip install .
```

#### Manual Dependency Installation (if needed)
```bash
pip install -r requirements.txt
pip install .
```

### Install via Standalone Executable (Windows)
For users who prefer not to install Python, download the standalone `gridflow.exe` from the [GitHub Releases](https://github.com/shahbhuwan/GridFlow/releases) page.
- **GUI**: Double-click `gridflow.exe` to launch the graphical interface.
- **CLI**: Run `gridflow.exe <command>` (e.g., `gridflow.exe crop --demo`) from a command prompt.
- No Python installation or dependencies required.

## Usage

GridFlow supports both a **Graphical User Interface (GUI)** and a **Command-Line Interface (CLI)**. The standalone executable (`gridflow.exe` on Windows) and the Python package (`gridflow`) support both modes.

### GUI Usage
1. **Launch the GUI**:
   - **Windows Executable**: Double-click `gridflow.exe` (downloaded from [GitHub Releases](https://github.com/shahbhuwan/GridFlow/releases)).
   - **Python Installation**: Run `python -m gridflow` or `gridflow` (after `pip` installation).
2. **Configure Operations**:
   - Use the intuitive interface to set parameters for downloading (CMIP5, CMIP6, PRISM), cropping, clipping, or catalog generation.
   - Select input/output directories, shapefiles, and other options via dropdowns and file pickers.
   - Configure logging level, number of workers, retries, and timeouts.
3. **Run Tasks**:
   - Click "Run" to execute the selected operation.
   - Monitor progress and logs in the GUI’s output panel.
   - Save or export results (e.g., NetCDF files, JSON catalogs) to the specified output directory.
4. **Demo Mode**:
   - Enable the "Demo" option to test with predefined settings (e.g., Iowa shapefile for clipping, sample CMIP6 data).

**Outputs**: NetCDF files, JSON catalogs, and logs in the specified directories (e.g., `./cmip6_data`, `./logs`).

### CLI Usage
Run `gridflow --help` to see available subcommands and options. The CLI is accessible via:
- **Python Installation**: Use `gridflow <command>` after installing via `pip`.
- **Windows Executable**: Use `gridflow.exe <command>` from a command prompt.

#### Download CMIP5 Data
Download CMIP5 `tas` (surface air temperature) data for HadCM3 (historical experiment) from MOHC:
```bash
gridflow download-cmip5 \
  --project CMIP5 \
  --institute MOHC \
  --variable tas \
  --model HadCM3 \
  --experiment historical \
  --frequency mon \
  --output-dir ./cmip5_data \
  --metadata-dir ./metadata \
  --log-level normal \
  --workers 4 \
  --retries 5 \
  --timeout 30 \
  --max-downloads 10 \
  --openid https://esgf-node.llnl.gov/esgf-idp/openid/your_username \
  --username your_username \
  --password your_password
```

**Outputs**: NetCDF files in `./cmip5_data` (e.g., `./cmip5_data/tas/250km/MOHC/`), metadata in `./metadata/query_results.json`, logs in `./logs`.

- Use `--demo` for a test run with predefined settings (e.g., CanESM2, historical).
- Use `--retry-failed path/to/failed_downloads.json` to retry failed downloads.
- Authentication (`--openid`, `--username`, `--password`) is required for restricted datasets.

#### Download CMIP6 Data
Download HighResMIP `tas` data at 50 km resolution with daily frequency:
```bash
gridflow download \
  --project CMIP6 \
  --activity HighResMIP \
  --variable tas \
  --resolution "50 km" \
  --frequency day \
  --output-dir ./cmip6_data \
  --metadata-dir ./metadata \
  --log-level normal \
  --workers 4 \
  --retries 5 \
  --timeout 30 \
  --max-downloads 10
```

**Outputs**: NetCDF files in `./cmip6_data`, metadata in `./metadata/query_results.json`, logs in `./logs`.

- Use `--demo` for a test run with predefined settings.

#### Download PRISM Data
Download PRISM daily precipitation data at 4km resolution for 2020:
```bash
gridflow download-prism \
  --variable ppt \
  --resolution 4km \
  --time-step daily \
  --year 2020 \
  --output-dir ./prism_data \
  --log-level normal \
  --retries 3 \
  --timeout 30 \
  --demo
```

**Options**:
- `--variable`: ppt, tmax, tmin, tmean, tdmean, vpdmin, vpdmax
- `--resolution`: 4km, 800m
- `--time-step`: daily, monthly
- `--year`: 1981-present (daily), 1895-present (monthly)
- `--output-dir`: Output directory (default: ./prism_data)
- `--log-dir`: Log directory (default: ./logs)
- `--log-level`: minimal, normal, verbose, debug
- `--retries`: Number of download retries (default: 3)
- `--timeout`: HTTP timeout in seconds (default: 30)
- `--demo`: Download one day (YYYY-01-01) or month (YYYY)

**Outputs**: ZIP files in `./prism_data` (e.g., `PRISM_ppt_daily_20200101_4km.zip`), logs in `./logs`.

#### Crop NetCDF Files
Crop NetCDF files to a geographic region (e.g., latitude 35–70, longitude -10–40):
```bash
gridflow crop \
  --input-dir ./cmip6_data \
  --output-dir ./cmip6_data_cropped \
  --min-lat 35 \
  --max-lat 70 \
  --min-lon -10 \
  --max-lon 40 \
  --buffer-km 10 \
  --log-level normal \
  --workers 4
```

**Outputs**: Cropped NetCDF files in `./cmip6_data_cropped`.

- Use `--demo` for a test run with default bounds (latitude 35–45, longitude -105–-95).

#### Clip NetCDF Files
Clip NetCDF files using a shapefile (e.g., Iowa border shapefile in `iowa_border/`):
```bash
gridflow clip \
  --input-dir ./cmip6_data \
  --output-dir ./cmip6_data_clipped \
  --shapefile ./iowa_border/iowa_border.shp \
  --buffer-km 10 \
  --log-level normal \
  --workers 4
```

**Outputs**: Clipped NetCDF files in `./cmip6_data_clipped`.

- Use `--demo` for a test run with the sample shapefile `iowa_border/iowa_border.shp`.

#### Generate Catalog
Create a JSON catalog of NetCDF files:
```bash
gridflow catalog \
  --input-dir ./cmip6_data \
  --output-dir ./output \
  --log-level normal \
  --workers 4
```

**Outputs**: `catalog.json` in `./output`.

- Use `--demo` for a test run with `demo_catalog.json`.

#### Batch Processing
Run multiple commands from a JSON configuration file:
```bash
gridflow batch --config batch.json
```

**Example `batch.json`**:
```json
[
  {
    "command": "download-cmip5",
    "args": {
      "project": "CMIP5",
      "institute": "MOHC",
      "variable": "tas",
      "model": "HadCM3",
      "experiment": "historical",
      "frequency": "mon",
      "output-dir": "./cmip5_data",
      "log-level": "normal",
      "workers": 4,
      "retries": 5,
      "timeout": 30,
      "max-downloads": 10,
      "openid": "https://esgf-node.llnl.gov/esgf-idp/openid/your_username",
      "username": "your_username",
      "password": "your_password"
    }
  },
  {
    "command": "download",
    "args": {
      "project": "CMIP6",
      "activity": "HighResMIP",
      "variable": "tas",
      "resolution": "50 km",
      "frequency": "day",
      "output-dir": "./cmip6_data",
      "log-level": "normal",
      "workers": 4,
      "retries": 5,
      "timeout": 30,
      "max-downloads": 10
    }
  },
  {
    "command": "crop",
    "args": {
      "input-dir": "./cmip6_data",
      "output-dir": "./cmip6_data_cropped",
      "min-lat": 35,
      "max-lat": 70,
      "min-lon": -10,
      "max-lon": 40,
      "buffer-km": 10,
      "log-level": "normal",
      "workers": 4
    }
  },
  {
    "command": "clip",
    "args": {
      "input-dir": "./cmip6_data",
      "output-dir": "./cmip6_data_clipped",
      "shapefile": "./iowa_border/iowa_border.shp",
      "buffer-km": 10,
      "log-level": "normal",
      "workers": 4
    }
  }
]
```

**Outputs**: Depends on the commands (e.g., NetCDF files in `./cmip5_data`, `./cmip6_data`, cropped/clipped files, logs in `./logs`).

#### Individual Commands
Standalone commands are also available:
- `gridflow-download`
- `gridflow-download-cmip5`
- `gridflow-download-prism`
- `gridflow-crop`
- `gridflow-clip`
- `gridflow-catalog`
- `gridflow-batch`

Example:
```bash
gridflow-download-cmip5 --variable tas --model HadCM3 --experiment historical --institute MOHC --demo
```

## Testing

GridFlow includes a test suite in the `tests/` directory, using pytest to verify functionality for downloading, cropping, clipping, and catalog generation.

### Running Tests
Install testing dependencies:
```bash
pip install pytest pytest-cov
```

Run tests:
```bash
pytest tests/
```

This runs all tests in `tests/` (e.g., `test_downloader.py`, `test_cmip5_downloader.py`, `test_clip_netcdf.py`, `test_crop_netcdf.py`).

### Run Tests with Coverage
To generate a coverage report:
```bash
pytest --cov=gridflow --cov-report=html tests/
```

Outputs a coverage report in `htmlcov/`. View the report by opening `htmlcov/index.html` in a browser.

### Sample Data
The `iowa_border/` directory contains shapefiles (e.g., `iowa_border.shp`) for testing the `clip` subcommand. Ensure test NetCDF files are available in `cmip5_data/` or `cmip6_data/` for full test coverage.

## Building the Package

GridFlow provides pre-built artifacts (wheel and executable) and build configuration files (`setup.py`, `pyproject.toml`, `MANIFEST.in`) for creating distributable packages.

### Using Pre-built Artifacts
Download from [GitHub Releases](https://github.com/shahbhuwan/GridFlow/releases):
- **Wheel**: `gridflow-1.0-py3-none-any.whl` for `pip` installation.
- **Executable**: `gridflow.exe` for Windows (GUI and CLI support).

Install the wheel:
```bash
pip install gridflow-1.0-py3-none-any.whl
```

### Rebuilding the Package
Install build tools:
```bash
pip install --upgrade pip setuptools wheel build twine
```

Build the package:
```bash
python -m build
```

Generates:
- `dist/gridflow-1.0.tar.gz` (source distribution)
- `dist/gridflow-1.0-py3-none-any.whl` (wheel)

Verify the build:
```bash
pip install dist/gridflow-1.0-py3-none-any.whl
gridflow --help
```

### Building the Executable
Install PyInstaller:
```bash
pip install pyinstaller
```

Build the executable:
```bash
pyinstaller gridflow.spec
```

Generates `dist/gridflow/gridflow.exe` (Windows). The executable includes all dependencies and supports both GUI (double-click) and CLI (`gridflow.exe <command>`).

Upload to PyPI (optional):
```bash
twine upload dist/*
```

Requires a PyPI account.

## Project Structure

```
GridFlow/
├── gridflow/
│   ├── __init__.py
│   ├── __main__.py
│   ├── entry.py
│   ├── batch.py
│   ├── clip_netcdf.py
│   ├── crop_netcdf.py
│   ├── catalog_generator.py
│   ├── cmip5_downloader.py
│   ├── cmip6_downloader.py
│   ├── prism_downloader.py
│   │   ├── iowa_border.shp
│   │   ├── iowa_border.dbf
│   │   ├── iowa_border.shx
│   │   ├── ...
├── gui/
├── __init__.py
├── main.py
├── vocab/
│   │   ├── *.json
├── tests/
│   ├── __init__.py
│   ├── test_cmip5_downloader.py
│   ├── test_downloader.py
│   ├── test_clip_netcdf.py
│   ├── test_crop_netcdf.py
│   ├── test_catalog_generator.py
│   ├── test_download_prism.py
│   ├── test_main.py
├── dist/
│   ├── gridflow-1.0.tar.gz
│   ├── gridflow-1.0-py3-none-any.whl
│   ├── gridflow/
│   │   ├── gridflow.exe
├── .github/
│   └── workflows/
│       ├── test.yml
├── setup.py
├── pyproject.toml
├── MANIFEST.in
├── README.md
├── LICENSE.txt
├── requirements.txt
├── gridflow_logo.png
├── gridflow_logo.ico
├── .pytest_cache/
├── htmlcov/
├── .coverage
├── coverage.xml
```

**Notes**:
- `iowa_border/` contains sample shapefiles for testing the `clip` subcommand.
- `gui/vocab/` contains JSON configuration files for the GUI.
- `.pytest_cache/`, `htmlcov/`, `.coverage`, and `coverage.xml` are generated by pytest and coverage tools.
- `.github/workflows/test.yml` defines a GitHub Actions workflow for automated testing.

## Troubleshooting

- **ESGF Node Errors**: If nodes like `esgf-node.llnl.gov` fail, try `--no-verify-ssl` or modify `ESGF_NODES` in `cmip5_downloader.py` or `cmip6_downloader.py` to prioritize `esgf-node.ipsl.upmc.fr`.
- **Authentication Issues**: Ensure valid ESGF credentials for CMIP5/CMIP6 restricted data. Register at `https://esgf-node.llnl.gov`.
- **Dependency Issues**: Ensure system libraries for `geopandas` and `netCDF4` are installed. Reinstall dependencies:
  ```bash
  pip install --force-reinstall -r requirements.txt
  ```
- **No Files Found**: Verify query parameters on the ESGF website (`https://esgf-node.llnl.gov/search/cmip5` or `https://esgf-node.llnl.gov/search/cmip6`). Use `--log-level debug`.
- **Shapefile Errors**: Ensure `iowa_border/iowa_border.shp` is valid. Test with QGIS.
- **GUI Issues**: Ensure `PyQt5` is installed (`pip install PyQt5`). For the executable, verify `gridflow_logo.ico` and `gui/vocab/*.json` are included.
- **Test Failures**: Check `tests/` for required sample data (e.g., NetCDF files in `cmip5_data/` or `cmip6_data/`). Run `pytest --verbose` for details.

Check logs in `./logs` for detailed error messages.

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Run tests (`pytest tests/`).
5. Push to the branch (`git push origin feature/your-feature`).
6. Open a pull request.

## License

GridFlow is licensed under the GNU Affero General Public License v3 or later (AGPLv3+). See `LICENSE.txt` for details.

## Contact

For questions or support, contact Bhuwan Shah at `bshah@iastate.edu` or open an issue on [GitHub](https://github.com/shahbhuwan/GridFlow).
```