# GridFlow

GridFlow is a modular toolset for downloading and processing geospatial and climate data, including PRISM, DEM, CMIP5, CMIP6, and ERA5 datasets. It provides both a **Command-Line Interface (CLI)** for advanced users and a **Graphical User Interface (GUI)** for ease of use. GridFlow supports tasks such as data downloading, cropping, clipping, unit conversion, temporal aggregation, and catalog generation.

**Author**: Bhuwan Shah  
**License**: GNU AGPLv3  
**Version**: 1.0  
**GitHub**: [shahbhuwan/GridFlow](https://github.com/shahbhuwan/GridFlow)  
**Contact**: [bshah@iastate.edu](mailto:bshah@iastate.edu)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Graphical User Interface (GUI)](#graphical-user-interface-gui)
- [Examples](#examples)
- [Dependencies](#dependencies)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Data Download**:
  - PRISM: Daily climate data (e.g., precipitation, temperature) at 4km or 800m resolution.
  - DEM: Digital Elevation Models via OpenTopography (e.g., COP30, SRTM).
  - CMIP5/CMIP6: Climate model data from ESGF nodes.
  - ERA5: High-resolution climate reanalysis data.
- **Data Processing**:
  - Crop NetCDF files to a spatial bounding box.
  - Clip NetCDF files using shapefiles (e.g., Iowa border).
  - Convert units (e.g., Kelvin to Celsius, flux to mm/day).
  - Temporally aggregate data (monthly, seasonal, annual).
  - Generate JSON catalogs summarizing NetCDF metadata.
- **User Interfaces**:
  - CLI for scriptable, automated workflows.
  - GUI with a modern, user-friendly interface built using PyQt5.
- **Demo Mode**: Try sample configurations for quick testing.
- **Cross-Platform**: Compatible with Windows, macOS, and Linux.

---

## Installation

### Prerequisites

- **Python**: Version 3.10 or higher.
- **pip**: Python package manager.
- **Virtual Environment** (recommended): To isolate dependencies.

### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shahbhuwan/GridFlow.git
   cd GridFlow
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv test-env
   source test-env/bin/activate  # On Windows: test-env\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required packages listed in `requirements.txt` or directly via `setup.py`.
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, install the package directly:
   ```bash
   pip install .
   ```

4. **Verify Installation**:
   Check if the CLI is accessible:
   ```bash
   gridflow --version
   ```
   This should output `GridFlow 1.0`.

5. **Optional: Install for GUI**:
   The GUI requires PyQt5, which is included in the dependencies. Ensure you have a compatible display environment (e.g., X11 on Linux, or a desktop environment on Windows/macOS).

6. **Windows Executable (Optional)**:
   A pre-built executable (`GridFlow.exe`) is available in the `dist` directory for Windows users. Run it directly to launch the GUI without installing Python:
   ```bash
   .\dist\GridFlow.exe
   ```

### Notes
- Some datasets (e.g., ERA5, DEM) require API keys. Obtain them from:
  - [CDS API](https://cds.climate.copernicus.eu/) for ERA5.
  - [OpenTopography](https://opentopography.org/) for DEM.
- CMIP5/CMIP6 may require ESGF credentials for restricted data.

---

## Usage

GridFlow offers two interfaces: a CLI for advanced users and a GUI for beginners or those preferring a visual interface.

### Command-Line Interface (CLI)

The CLI is invoked using the `gridflow` command, followed by a subcommand for the desired tool (e.g., `prism`, `crop`). Run `gridflow -h` for help or `gridflow <command> -h` for detailed options.

**Basic Syntax**:
```bash
gridflow <command> [options]
```

**Available Commands**:
- `prism`: Download PRISM climate data.
- `dem`: Download Digital Elevation Models.
- `cmip5`: Download CMIP5 climate model data.
- `cmip6`: Download CMIP6 climate model data.
- `era5`: Download ERA5-Land climate data.
- `crop`: Crop NetCDF files to a bounding box.
- `clip`: Clip NetCDF files using a shapefile.
- `convert`: Convert units in NetCDF files.
- `aggregate`: Temporally aggregate NetCDF files.
- `catalog`: Generate a JSON catalog from NetCDF files.

**Example**:
Download sample PRISM data:
```bash
gridflow prism --demo
```

Run the CLI with verbose output:
```bash
gridflow prism --variable tmean --resolution 4km --start-date 2020-01-01 --end-date 2020-01-05 --output-dir ./downloads_prism --log-level verbose
```

### Graphical User Interface (GUI)

The GUI is launched using the `gridflow-gui` command or the executable on Windows. It provides a user-friendly interface to configure and run the same tasks as the CLI.

**Launch the GUI**:
```bash
gridflow-gui
```
Or, on Windows:
```bash
.\dist\GridFlow.exe
```

**GUI Features**:
- Select data source (PRISM, DEM, CMIP5, CMIP6, ERA5) and process (Download, Crop, Clip, etc.).
- Configure parameters via dropdowns, text fields, and file browsers.
- Choose skill level (Beginner or Advanced) to simplify or expand options.
- View real-time logs and progress bars.
- Copy CLI commands for reproducibility.
- Demo mode for quick testing.

**Steps to Use**:
1. Select a **Data Source** (e.g., PRISM) and **Process** (e.g., Download).
2. Fill in required fields (e.g., API key, output directory).
3. Optionally enable **Demo** mode for sample data.
4. Click **Start** to run the task.
5. Monitor progress and logs in the bottom panel.

---

## Examples

### CLI Examples
1. **Download CMIP6 Data**:
   ```bash
   gridflow cmip6 --variable tas --experiment hist-1950 --model HadGEM3-GC31-LL --frequency day --resolution "250 km" --output-dir ./downloads_cmip6 --metadata-dir ./metadata_cmip6 --log-dir ./gridflow_logs --demo
   ```

2. **Crop NetCDF Files**:
   ```bash
   gridflow crop --input-dir ./downloads_cmip6 --output-dir ./cropped_cmip6 --min-lat 25.0 --max-lat 50.0 --min-lon -125.0 --max-lon -65.0 --log-dir ./gridflow_logs
   ```

3. **Clip with Iowa Shapefile**:
   ```bash
   gridflow clip --input-dir ./downloads_cmip6 --shapefile ./iowa_border/iowa_border.shp --output-dir ./clipped_cmip6 --log-dir ./gridflow_logs
   ```

### GUI Example
1. **Download ERA5 Data**:
   - Select **ERA5** as the Data Source and **Download** as the Process.
   - Enter your CDS API key (UID:KEY format).
   - Set Start Date to `2023-01-01` and End Date to `2023-01-31`.
   - Choose a variable (e.g., `2m_temperature`) and an AOI (e.g., `corn_belt`).
   - Specify output and log directories.
   - Click **Start** to download.

---

## Dependencies

GridFlow relies on the following Python packages (listed in `setup.py`):
- PyQt5 (>=5.15.0): For the GUI.
- requests (>=2.28.0): For HTTP requests.
- numpy (>=1.21.0): For numerical operations.
- netCDF4 (>=1.5.8): For NetCDF file handling.
- geopandas (>=0.10.0): For geospatial data processing.
- shapely (>=1.8.0): For geometric operations.
- python-dateutil (>=2.8.0): For date handling.
- cdsapi (>=0.7.4): For ERA5 data access.

Install them automatically during setup:
```bash
pip install .
```

---

## Directory Structure

```
GridFlow/
├── gridflow/                   # Source code
│   ├── download/              # Download modules (prism, dem, cmip5, cmip6, era5)
│   ├── processing/            # Processing modules (crop, clip, convert, aggregate, catalog)
│   ├── utils/                 # Utility functions (e.g., logging)
│   ├── vocab/                 # JSON files for CMIP5/CMIP6 metadata
│   ├── cli.py                 # CLI implementation
│   ├── gui.py                 # GUI implementation
│   └── __init__.py
├── iowa_border/               # Shapefile for Iowa border
├── gridflow_logs/             # Default log directory
├── dist/                      # Built distributions (wheel, tar.gz, executable)
├── setup.py                   # Package setup
├── requirements.txt           # Dependency list
├── gridflow_logo.png          # Logo for GUI
├── gridflow_logo.svg          # SVG logo for GUI
└── README.md                  # This file
```

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

Please include tests and update documentation as needed. Follow the coding style in existing files (e.g., PEP 8 for Python).

---

## License

GridFlow is licensed under the **GNU Affero General Public License v3.0** (AGPLv3). See the [LICENSE](https://www.gnu.org/licenses/agpl-3.0.en.html) file for details.

---

## Acknowledgments

- **Open-Source Community**: For libraries like PyQt5, netCDF4, and geopandas.
- **Data Providers**: PRISM, OpenTopography, ESGF, and CDS for climate and geospatial data.

---