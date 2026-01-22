# GridFlow: A High-Performance Climate Data Pipeline

**GridFlow** is a modular, high-performance toolset designed to streamline the acquisition and processing of massive geospatial and climate datasets. It provides a unified interface to access petabytes of data from **CMIP6**, **CMIP5**, **ERA5**, **PRISM**, and **global DEM repositories**—without requiring users to navigate complex web portals or write custom scraper scripts.

- **Author:** Bhuwan Shah  
- **License:** GNU AGPLv3  
- **Version:** 1.0  
- **GitHub:** `shahbhuwan/GridFlow`  

---

## 📖 Statement of Need

In the era of big data, acquiring climate data remains a significant bottleneck for researchers, students, and independent analysts.

### The core challenges
- **The "Click Fatigue" Problem:** Archives like ESGF often require users to manually navigate faceted search interfaces, deal with pagination, and download files one-by-one via HTTP or brittle `wget` scripts.
- **API Complexity:** Accessing reanalysis datasets (like ERA5) usually requires accounts, API keys, and queue-based systems.
- **Technical Barrier:** Common post-processing tasks—cropping, clipping, aggregating, and converting NetCDF datasets—often require intermediate Python/R skills.

### How GridFlow solves this
✅ **Streamlined Access:** GUI + CLI interface for researchers, students, and non-programmers.  
✅ **Parallel Downloads:** Multi-threaded downloading for high-speed bulk retrieval.  
✅ **Direct Cloud Access:** ERA5 + DEM can be fetched directly from public AWS Open Data (no API keys required).  
✅ **Built-in Processing:** Crop, clip, aggregate, convert, and catalog NetCDF datasets for immediate analysis.

---

## 🚀 Installation

GridFlow is designed for portability. You do not need to be a Python expert to use it.

### Option 1: Standalone GUI (No Install Required)
✅ **Simplest method**: No Python installation needed.

1. Download the `GridFlow_GUI.exe` from the **Releases** page.
2. Double-click the `.exe` to launch the application.

---

### Option 2: CLI Wrapper (Windows Batch)
If you prefer the command line but want to avoid manual Python setup:

1. Clone or download this repository
2. Double-click `setup_cli.bat`

This will automatically:
- create a virtual environment  
- install dependencies  
- open a ready-to-use terminal for GridFlow commands

---

### Option 3: Developer Setup (Pip)
For developers integrating GridFlow into existing workflows:

```bash
git clone https://github.com/shahbhuwan/GridFlow.git
cd GridFlow
pip install -r requirements.txt
python setup.py install
```

---

## 🧪 Developer & Testing Setup

GridFlow provides **two requirements files**:

### `requirements.txt`
General/runtime dependencies needed to **run GridFlow**.

### `requirements_dev.txt`
Developer/testing dependencies (pytest + tooling), including:
- `pytest`
- `pytest-cov`
- `pytest-qt`
- `pytest-mock`

✅ **Recommended approach:** keep dev requirements as a superset.

That means `requirements_dev.txt` should include:

```txt
-r requirements.txt
pytest
pytest-cov
pytest-qt
pytest-mock
```

Then developers can install everything with:

```bash
pip install -r requirements_dev.txt
```

---

## ✅ CLI Usage

### Help Menu

```bash
gridflow -h
```

Run command-specific help using:

```bash
gridflow <command> -h
```

---

## 🛠️ Data Download Modules

GridFlow abstracts the complexity of different archives into a consistent interface.

---

## 1. CMIP6 Downloader (ESGF)

Download CMIP6 climate model data from ESGF nodes using flexible filters.

### **Command**
```bash
gridflow cmip6 [OPTIONS]
```

### **Argument Table**
| Argument | Flag | Description | Example |
|---------|------|-------------|---------|
| Project | `-p` / `--project` | MIP project name | `CMIP6` |
| Activity | `-a` / `--activity` | Activity ID | `ScenarioMIP` |
| Experiment | `-e` / `--experiment` | Experiment ID | `ssp585` |
| Model | `-m` / `--model` | Source model ID | `HadGEM3-GC31-LL` |
| Variable | `-var` / `--variable` | Variable ID | `tas` |
| Frequency | `-f` / `--frequency` | Time frequency | `day` |
| Resolution | `-r` / `--resolution` | Nominal resolution | `"250 km"` |
| Ensemble | `-en` / `--ensemble` | Variant label | `r1i1p1f1` |
| Grid Label | `-g` / `--grid_label` | Grid label | `gn` |
| Latest only | `--latest` | Only latest version | (flag) |
| Replica | `--replica` | Include replicas | (flag) |
| Data Node | `--data-node` | Restrict ESGF node | `esgf.ceda.ac.uk` |
| Config | `-c` / `--config` | JSON config to prefill args | `cmip6_config.json` |
| Output Dir | `-o` / `--output-dir` | Directory for downloads | `./downloads_cmip6` |
| Metadata Dir | `-md` / `--metadata-dir` | Directory for metadata | `./metadata_cmip6` |
| Log Dir | `-ld` / `--log-dir` | Directory for logs | `./gridflow_logs` |
| Log Level | `-ll` / `--log-level` | minimal / verbose / debug | `debug` |
| Save Mode | `-sm` / `--save-mode` | structured or flat | `structured` |
| Workers | `-w` / `--workers` | Parallel workers | `8` |
| Timeout | `-t` / `--timeout` | Network timeout (sec) | `60` |
| Max Downloads | `--max-downloads` | Limit downloads per run | `10` |
| Disable SSL verify | `--no-verify-ssl` | Turn off SSL verification | (flag) |
| Prefer Nodes | `--prefer-nodes` | Host fragments to prioritize | `esgf.ceda.ac.uk` |
| Resume | `--resume` | Resume partial downloads | (flag) |
| OpenID | `--openid` | ESGF OpenID | `https://.../openid/user` |
| Username | `--id` | ESGF username | `my_user` |
| Password | `--password` | ESGF password | `mypassword` |
| Retry Failed | `--retry-failed` | Retry from failed JSON list | `failed_downloads.json` |
| Dry Run | `--dry-run` | Query only (no download) | (flag) |
| Demo | `--demo` | Run demo query | (flag) |

### Example Command

```bash
gridflow cmip6 -a HighResMIP -var tas -m HadGEM3-GC31-LL -e hist-1950 -f day -o ./my_data
```

---

## 2. CMIP5 Downloader (ESGF)

Download CMIP5 climate model data from ESGF nodes.

### **Command**
```bash
gridflow cmip5 [OPTIONS]
```

### **Argument Table**
| Argument | Flag | Description | Example |
|---------|------|-------------|---------|
| Project | `-p` / `--project` | Project name | `CMIP5` |
| Model | `-m` / `--model` | Model name | `CanESM2` |
| Institute | `-i` / `--institute` | Modeling institute | `CCCMA` |
| Experiment | `-e` / `--experiment` | Experiment name | `historical` |
| Experiment Family | `--experiment_family` | Family label | `RCP` |
| Variable | `-var` / `--variable` | Variable name | `tas` |
| Frequency | `-f` / `--frequency` | Time frequency | `mon` |
| Realm | `-r` / `--realm` | Model realm | `atmos` |
| Ensemble | `-en` / `--ensemble` | Ensemble member | `r1i1p1` |
| Latest only | `-l` / `--latest` | Only get latest versions | (flag) |
| Config | `-c` / `--config` | JSON config to prefill args | `cmip5_config.json` |
| Output Dir | `-o` / `--output-dir` | Download directory | `./downloads_cmip5` |
| Metadata Dir | `-md` / `--metadata-dir` | Metadata directory | `./metadata_cmip5` |
| Log Dir | `-ld` / `--log-dir` | Log directory | `./gridflow_logs` |
| Log Level | `-ll` / `--log-level` | minimal / verbose / debug | `debug` |
| Save Mode | `-sm` / `--save-mode` | structured or flat | `structured` |
| Workers | `-w` / `--workers` | Parallel workers | `4` |
| Timeout | `-t` / `--timeout` | Network timeout (sec) | `30` |
| Max Downloads | `--max-downloads` | Limit downloads per run | `10` |
| Disable SSL verify | `--no-verify-ssl` | Turn off SSL verification | (flag) |
| OpenID | `--openid` | ESGF OpenID | `https://.../openid/user` |
| Username | `--id` | ESGF username | `my_user` |
| Password | `--password` | ESGF password | `mypassword` |
| Retry Failed | `--retry-failed` | Retry from failed JSON list | `failed_downloads.json` |
| Dry Run | `--dry-run` | Query only (no download) | (flag) |
| Demo | `--demo` | Run demo query | (flag) |

### Example Command

```bash
gridflow cmip5 --demo
```

---

## 3. ERA5 Downloader (AWS Open Data)

Fetch ERA5(-Land) climate reanalysis data **directly from AWS S3** (no CDS API key required). fileciteturn3file5

### **Command**
```bash
gridflow era5 [OPTIONS]
```

### **Argument Table**
| Argument | Flag | Description | Example |
|---------|------|-------------|---------|
| Variables | `-var` / `--variables` | Comma-separated variable list | `t2m,precip,u10` |
| Start Date | `-sd` / `--start-date` | Start date (YYYY-MM-DD) | `2021-01-01` |
| End Date | `-ed` / `--end-date` | End date (YYYY-MM-DD) | `2021-12-31` |
| List Variables | `-lv` / `--list-variables` | Print variable table | (flag) |
| Output Dir | `-o` / `--output-dir` | Download directory | `./downloads_era5` |
| Metadata Dir | `-md` / `--metadata-dir` | Metadata directory | `./metadata_era5` |
| Log Dir | `-ld` / `--log-dir` | Log directory | `./gridflow_logs` |
| Log Level | `-ll` / `--log-level` | minimal / verbose / debug | `minimal` |
| Workers | `-w` / `--workers` | Parallel workers | `4` |
| Dry Run | `--dry-run` | Query only (no download) | (flag) |
| Demo | `--demo` | Run demo query | (flag) |
| Config | `-c` / `--config` | JSON config file | `era5_config.json` |
| Retry Failed | `--retry-failed` | Retry from JSON list | `failed_downloads.json` |

### Example Command

```bash
gridflow era5 --variables t2m,precip --start-date 2021-01-01 --end-date 2021-03-01 -o ./era5_data
```

---

## 4. PRISM Downloader (USA, High Resolution)

High-resolution historical climate data for the contiguous United States.

### **Command**
```bash
gridflow prism [OPTIONS]
```

### **Argument Table**
| Argument | Flag | Description | Example |
|---------|------|-------------|---------|
| Variable(s) | `-var` / `--variable` | PRISM climate variable(s) | `tmean` |
| Resolution | `-r` / `--resolution` | Spatial resolution | `4km` |
| Time Step | `-ts` / `--time-step` | daily or monthly | `daily` |
| Start Date | `-sd` / `--start-date` | Start date | `2020-01-01` |
| End Date | `-ed` / `--end-date` | End date | `2020-01-31` |
| Config | `-c` / `--config` | JSON config file | `prism_config.json` |
| Output Dir | `-o` / `--output-dir` | Output directory | `./downloads_prism` |
| Metadata Dir | `-md` / `--metadata-dir` | Metadata directory | `./metadata_prism` |
| Log Dir | `-ld` / `--log-dir` | Log directory | `./gridflow_logs` |
| Log Level | `-ll` / `--log-level` | minimal / verbose / debug | `minimal` |
| Workers | `-w` / `--workers` | Parallel workers | `8` |
| Timeout | `-t` / `--timeout` | Timeout in seconds | `30` |
| Dry Run | `--dry-run` | Check availability only | (flag) |
| Demo | `--demo` | Run demo query | (flag) |

### Example Command

```bash
gridflow prism -var tmean -r 4km -sd 2020-01-01 -ed 2020-01-31 -o ./prism_data
```

---

## 5. DEM Downloader (AWS Open Data)

Downloads elevation tiles based on a bounding box using AWS Open Data sources. fileciteturn3file16

### **Command**
```bash
gridflow dem [OPTIONS]
```

### **Argument Table**
| Argument | Flag | Description | Example |
|---------|------|-------------|---------|
| Bounds | `--bounds` | Bounding box (N S E W) | `43.5 40.0 -90.0 -96.0` |
| DEM Type | `--dem_type` | `COP30` (global) or `USGS10m` (USA) | `COP30` |
| Output Dir | `-o` / `--output-dir` | Output directory | `./downloads_dem` |
| Metadata Dir | `-md` / `--metadata-dir` | Metadata directory | `./metadata_dem` |
| Log Dir | `-ld` / `--log-dir` | Log directory | `./gridflow_logs` |
| Log Level | `-ll` / `--log-level` | minimal / verbose / debug | `minimal` |
| Workers | `-w` / `--workers` | Parallel workers | `4` |
| Dry Run | `--dry-run` | Query only | (flag) |
| Demo | `--demo` | Run Iowa demo | (flag) |
| Config | `-c` / `--config` | JSON config file | `dem_config.json` |
| Retry Failed | `--retry-failed` | Retry from JSON list | `failed_downloads.json` |

### Example Command

```bash
gridflow dem --bounds 43.5 40.3 -90.1 -96.7 --dem_type COP30 -o ./iowa_dem
```

---

# ⚙️ Processing Modules

GridFlow includes tools to prepare raw data for immediate analysis.

---

## 📋 Catalog Generator

Scans a directory of downloaded NetCDF files and generates a `catalog.json` summarizing metadata. fileciteturn2file5

### **Command**
```bash
gridflow catalog [OPTIONS]
```

### **Argument Table**
| Argument | Flag | Description |
|---------|------|-------------|
| Input Dir | `-i` / `--input_dir` | Folder containing NetCDF files (recursive search) |
| Output Dir | `-o` / `--output_dir` | Folder where catalog.json will be saved |
| Log Dir | `-ld` / `--log-dir` | Log directory |
| Log Level | `-ll` / `--log-level` | minimal / verbose / debug |
| Demo | `--demo` | Run demo defaults |

### Example
```bash
gridflow catalog -i ./downloads_cmip6 -o ./catalogs
```

---

## ✂️ Spatial Operations

### 1) Crop (Bounding Box)
Crop NetCDF files to a rectangular Lat/Lon bounding box. fileciteturn2file8

```bash
gridflow crop -i ./downloads -o ./cropped --min_lat 40 --max_lat 45 --min_lon -96 --max_lon -90
```

| Argument | Flag | Description |
|---------|------|-------------|
| Input Dir | `-i` / `--input_dir` | Directory containing NetCDF files |
| Output Dir | `-o` / `--output_dir` | Directory to save cropped files |
| Min Lat | `--min_lat` | Minimum latitude |
| Max Lat | `--max_lat` | Maximum latitude |
| Min Lon | `--min_lon` | Minimum longitude |
| Max Lon | `--max_lon` | Maximum longitude |
| Workers | `-w` / `--workers` | Parallel workers |
| Log Dir | `-ld` / `--log-dir` | Log directory |
| Log Level | `-ll` / `--log-level` | minimal / verbose / debug |
| Demo | `--demo` | Run demo defaults |

---

### 2) Clip (Shapefile)
Clip NetCDF files to an irregular polygon (shapefile), with optional buffering. fileciteturn2file9

```bash
gridflow clip -i ./downloads -o ./clipped -s ./watershed.shp --buffer_km 2
```

| Argument | Flag | Description |
|---------|------|-------------|
| Input Dir | `-i` / `--input_dir` | Directory containing NetCDF files |
| Output Dir | `-o` / `--output_dir` | Directory to save clipped files |
| Shapefile | `-s` / `--shapefile` | Path to `.shp` file |
| Buffer (km) | `--buffer_km` | Optional buffer distance |
| Workers | `-w` / `--workers` | Parallel workers |
| Log Dir | `-ld` / `--log-dir` | Log directory |
| Log Level | `-ll` / `--log-level` | minimal / verbose / debug |
| Demo | `--demo` | Run demo defaults |
| Config | `-c` / `--config` | JSON config file |

---

## ⏳ Temporal & Unit Operations

### 1) Aggregate (Temporal)
Convert daily NetCDF data into monthly/seasonal/annual aggregates. fileciteturn2file7

```bash
gridflow aggregate -i ./raw_data -o ./monthly_data -var tas --output_frequency monthly --method mean
```

| Argument | Flag | Description |
|---------|------|-------------|
| Input Dir | `-i` / `--input_dir` | Directory containing NetCDF files |
| Output Dir | `-o` / `--output_dir` | Output directory |
| Variable | `-var` / `--variable` | Target variable |
| Output Frequency | `--output_frequency` | monthly / seasonal / annual |
| Method | `--method` | mean / sum / min / max |
| Workers | `-w` / `--workers` | Parallel workers |
| Log Dir | `-ld` / `--log-dir` | Log directory |
| Log Level | `-ll` / `--log-level` | minimal / verbose / debug |
| Demo | `--demo` | Run demo defaults |

---

### 2) Convert (Units)
Convert NetCDF units (e.g., Kelvin → Celsius, Flux → mm/day). fileciteturn2file8

```bash
gridflow convert -i ./raw_data -o ./converted --variable tas --target_unit C
```

| Argument | Flag | Description |
|---------|------|-------------|
| Input Dir | `-i` / `--input_dir` | Directory containing NetCDF files |
| Output Dir | `-o` / `--output_dir` | Output directory |
| Variable | `--variable` | Variable ID |
| Target Unit | `--target_unit` | `C`, `mm/day`, `km/h` |
| Workers | `-w` / `--workers` | Parallel workers |
| Log Dir | `-ld` / `--log-dir` | Log directory |
| Log Level | `-ll` / `--log-level` | minimal / verbose / debug |
| Demo | `--demo` | Run demo defaults |
| Config | `-c` / `--config` | JSON config file |

---

## 📂 Directory Structure

```text
GridFlow/
├── gridflow/                   # Source code
│   ├── download/              # Download modules (prism, dem, cmip5, cmip6, era5)
│   ├── processing/            # Processing modules (crop, clip, convert, aggregate, catalog)
│   ├── utils/                 # Utility functions (e.g., logging)
│   ├── vocab/                 # JSON files for CMIP5/CMIP6 metadata
│   ├── cli.py                 # CLI implementation
│   ├── gui.py                 # GUI implementation
│   └── __init__.py
├── conus_border/               # Shapefile for Iowa border
├── gridflow_logs/             # Default log directory
├── dist/                      # Built distributions (wheel, tar.gz, executable)
├── setup.py                   # Package setup
├── requirements.txt           # Dependency list
├── requirements_dev.txt       # Developer + testing dependency list
├── gridflow_logo.png          # Logo for GUI
├── gridflow_logo.svg          # SVG logo for GUI
└── README.md                  # This file
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch:  
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit your changes:  
   ```bash
   git commit -m "Add YourFeature"
   ```
4. Push to the branch:  
   ```bash
   git push origin feature/YourFeature
   ```
5. Open a Pull Request

Please include tests and update documentation as needed. Follow existing coding style (PEP8 for Python).

---

## 📜 License

GridFlow is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

- **Open-Source Community:** for libraries like PyQt5, netCDF4, geopandas, rich, tqdm, and boto3  
- **Data Providers:** PRISM, OpenTopography, ESGF, and CDS for climate and geospatial data
