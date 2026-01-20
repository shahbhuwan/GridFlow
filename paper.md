---
title: 'GridFlow: A modular high-performance toolkit for downloading and processing large-scale climate and geospatial datasets'

tags:
  - climate data
  - CMIP6
  - CMIP5
  - ERA5
  - PRISM
  - NetCDF
  - geospatial
  - hydrology
  - Python

authors:
  - name: Bhuwan Shah
    orcid: "https://orcid.org/0000-0002-9963-6396"   
    affiliation: 1

affiliations:
  - name: Iowa State University, Ames, Iowa, USA
    index: 1
    
date: 2026-01-20
bibliography: paper.bib
---

# Summary

Climate and geospatial research increasingly relies on large observational and model-derived
datasets such as CMIP6, CMIP5, ERA5 reanalysis, PRISM climate records, and global digital
elevation models (DEMs). While these resources are widely available through web portals and
distributed archives, acquiring and preparing them for analysis remains a major bottleneck.
Researchers frequently spend considerable time navigating search interfaces, handling
authentication requirements, writing brittle download scripts, and performing repetitive
post-processing steps such as cropping, clipping, unit conversion, and temporal aggregation.

GridFlow is an open-source Python toolkit that streamlines the complete workflow of climate
and geospatial data preparation. It provides both a command-line interface (CLI) and a
graphical user interface (GUI) to download major climate products and to process NetCDF
datasets into analysis-ready subsets. GridFlow emphasizes modular design, parallel execution,
and usability to support a broad user community, including researchers, students, and
practitioners who need reliable access to large datasets without extensive custom scripting.

# Statement of need

Despite the continued growth of publicly available climate archives, data acquisition and
processing remain disproportionately time-consuming compared to downstream analysis.
For example, Earth System Grid Federation (ESGF) portals supporting CMIP5/CMIP6 provide
powerful search tools, but workflows often involve repeated manual filtering, pagination,
and batch downloads. Similarly, reanalysis data systems can impose account setup, API keys,
or queue-based access patterns that complicate reproducible retrieval.

In many applied workflows (e.g., hydrology, water resources, agriculture, and land-surface
modeling), users require climate variables for specific regions, watersheds, or administrative
boundaries and often need derived temporal summaries (e.g., monthly means, seasonal sums).
These tasks are commonly addressed through ad-hoc scripts using general-purpose scientific
Python tools such as Xarray [@xarray] and GeoPandas [@geopandas]. However, implementing
robust pipelines across multiple datasets and formats can create barriers to entry for new users
and reduce reproducibility across projects.

GridFlow addresses this gap by offering a single, consistent interface for acquiring and preparing
multiple widely used climate and geospatial datasets. The toolkit couples high-level download
modules with post-processing utilities to produce analysis-ready NetCDF outputs and metadata
summaries, enabling rapid adoption in research and educational contexts.

# Functionality and design

GridFlow is designed as a modular toolkit with two primary capability groups: (1) data downloading
modules and (2) processing modules.

## Downloading modules

GridFlow provides dedicated download modules for:

- **CMIP6 climate model data** via ESGF search and retrieval
- **CMIP5 climate model data** via ESGF search and retrieval
- **ERA5-Land reanalysis data** accessed directly from cloud-hosted sources
- **PRISM historical climate datasets** for the contiguous United States
- **Digital Elevation Models (DEM)** including global (Copernicus) and US-focused DEM products

Users interact with these modules using consistent CLI patterns, while GridFlow manages file
organization, logging, metadata generation, and parallel retrieval. Download operations can be
configured via command-line options or JSON configuration files to support reproducible workflows.

## Processing modules

GridFlow also includes post-processing utilities for common climate-data preparation tasks:

- **Spatial subsetting**
  - Cropping NetCDF datasets to a latitude/longitude bounding box
  - Clipping datasets using irregular polygons defined by shapefiles
- **Temporal aggregation**
  - Aggregation from daily to monthly, seasonal, or annual frequency using multiple methods
- **Unit conversion**
  - Converting common climate variable units (e.g., Kelvin to Celsius)
- **Catalog generation**
  - Scanning NetCDF directories and producing a JSON catalog of dataset metadata for inventorying
    large multi-model libraries

These operations are frequently required in climate impact studies, watershed-based modeling,
and regional climate analyses, and GridFlow provides standardized implementations to reduce
boilerplate and ensure consistent outputs.

# Related work

GridFlow complements existing climate data analysis libraries such as xclim [@xclim] and xCDAT [@xcdat],
which focus on robust climate analytics and indicator computation. While these tools emphasize analysis
operations on structured datasets, GridFlow focuses on streamlining the acquisition and preprocessing
stages that are often a prerequisite for analysis at regional scales. GridFlow also aligns with the goals of
packages that simplify weather and climate data acquisition, such as weathercan [@weathercan].

# Availability

GridFlow is released under the GNU Affero General Public License v3.0 (AGPLv3) and is available
on GitHub at:

https://github.com/shahbhuwan/GridFlow

# Acknowledgements

The author thanks the open-source scientific Python community for foundational libraries that enable
GridFlow's functionality, including Xarray, GeoPandas, and the broader NetCDF and geospatial ecosystem.
The author also acknowledges the data providers and maintainers of ESGF, PRISM Climate Group, ECMWF
(ERA5), Copernicus DEM, and USGS datasets for making large-scale climate and geospatial resources publicly accessible.
