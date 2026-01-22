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
  - name: Bhuwan P. Shah
    orcid: 0000-0002-9963-6396
    affiliation: 1
  - name: Ryan P. McGehee
    orcid: 0000-0003-0464-9774
    affiliation: 1

affiliations:
  - name: Iowa State University, Ames, Iowa, USA
    index: 1

date: 2026-01-20
bibliography: paper.bib
---

# Summary

Climate and geospatial research increasingly relies on large observational and model-derived datasets such as CMIP6, CMIP5, ERA5 reanalysis, PRISM climate records, and global digital elevation models (DEMs). While these resources are widely available through web portals and distributed archives, acquiring and preparing them for analysis remains a major bottleneck. Researchers frequently spend considerable time navigating search interfaces, handling authentication requirements, writing brittle download scripts, and performing repetitive post-processing steps such as cropping, clipping, unit conversion, and temporal aggregation.

GridFlow is an open-source Python based toolkit that streamlines the complete workflow of climate and geospatial data preparation. It provides both a command-line interface (CLI) and a graphical user interface (GUI) to download major climate products and to process NetCDF datasets into analysis-ready subsets. GridFlow emphasizes modular design, parallel execution, and usability to support a broad user community, including researchers, students, and practitioners who need reliable access to large datasets without extensive custom scripting.

GridFlow is designed to be accessible to a wide range of users through both a GUI (Figure 1) and a command-line interface (Figure 2), enabling users to reproducibly download and process climate and geospatial datasets without the need for custom scripts.

![GridFlow GUI showing the unified workflow for dataset acquisition and preprocessing.](gridflow_gui.png)

![GridFlow CLI interface and built-in command help (`gridflow -h`).](gridflow_cli_help.png)

# Statement of need

Despite the continued growth of publicly available climate archives, data acquisition and processing remain disproportionately time-consuming compared to downstream analysis. For example, Earth System Grid Federation (ESGF) portals supporting CMIP5/CMIP6 provide powerful search tools, but workflows often involve repeated manual filtering, pagination, and batch downloads. Similarly, reanalysis data systems can impose account setup, API keys, or queue-based access patterns that complicate reproducible retrieval.

In many applied workflows (e.g., hydrology, water resources, agriculture, and land-surface modeling), users require climate variables for specific regions, watersheds, or administrative boundaries and often need derived temporal summaries (e.g., monthly means, seasonal sums). These tasks are commonly addressed through ad-hoc scripts using general-purpose scientific Python tools such as Xarray [@xarray] and GeoPandas [@geopandas]. However, implementing robust pipelines across multiple datasets and formats can create barriers to entry for new users and reduce reproducibility across projects.

GridFlow addresses this gap by offering a single, consistent interface for acquiring and preparing multiple widely used climate and geospatial datasets. The toolkit couples high-level download modules with post-processing utilities to produce analysis-ready NetCDF outputs and metadata summaries, enabling rapid adoption in research and educational contexts.

# Functionality and design

GridFlow is designed as a modular toolkit with two primary capability groups: (1) data downloading modules and (2) processing modules.

## Downloading modules

GridFlow provides dedicated download modules for:

- **CMIP6 climate model data** via ESGF search and retrieval
- **CMIP5 climate model data** via ESGF search and retrieval
- **ERA5-Land reanalysis data** accessed directly from cloud-hosted sources
- **PRISM historical climate datasets** for the contiguous United States
- **Digital Elevation Models (DEM)** including global (Copernicus) and US-focused DEM products

Users interact with these modules using consistent CLI patterns, while GridFlow manages file organization, logging, metadata generation, and parallel retrieval. Download operations can be configured via command-line options or JSON configuration files to support reproducible workflows.

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

Figure 3 demonstrates a typical workflow in which users download global or continental-scale NetCDF datasets and then spatially subset the files to a target watershed or region using cropping or clipping.

![Example of spatial subsetting using GridFlow: clipping to CONUS polygon and cropping to a bounding box.](gridflow_clip_crop_example.png)

These operations are frequently required in climate impact studies, watershed-based modeling, and regional climate analyses, and GridFlow provides standardized implementations to reduce boilerplate and ensure consistent outputs.

# Related work

GridFlow is positioned at the intersection of data acquisition and analysis-ready preprocessing for large climate and geospatial archives. It complements climate analytics libraries such as xclim [@xclim] and xCDAT [@xcdat], which focus on indicator computation, diagnostics, and analysis workflows once data are already locally available and harmonized. In contrast, GridFlow emphasizes reducing friction in the upstream workflow: discovering, downloading, organizing, and regionally subsetting large datasets across heterogeneous archives (e.g., ESGF, AWS-hosted ERA5, PRISM, and DEM tile repositories).

GridFlow also aligns with tools that simplify retrieval of environmental data products, such as weathercan [@weathercan]. However, GridFlow differs by providing a unified interface for multiple high-volume climate and geospatial sources, including climate model ensembles (CMIP5/CMIP6), reanalysis products (ERA5-Land), high-resolution observational gridded products (PRISM), and elevation models (Copernicus DEM, USGS products). Additionally, GridFlow integrates common preprocessing steps—spatial cropping/clipping, temporal aggregation, unit conversion, and catalog generation—into a single toolchain intended for end-to-end reproducibility.

For reanalysis access specifically, the Copernicus Climate Data Store (CDS) API is the standard entry point for downloading ERA5-family products [@cdsapi]. While CDS provides broad access to ECMWF datasets, it typically requires account registration and API credential setup, and downloads involve queue-based processing depending on the request. GridFlow complements this approach by enabling direct retrieval of ERA5 reanalysis data from publicly hosted cloud buckets (e.g., AWS Open Data mirrors such as the NSF/NCAR ERA5 archive), lowering operational overhead for users who need bulk, scriptable access for workflows such as watershed-scale modeling or regional climate analysis.

Finally, GridFlow is intended to serve as an extensible “download hub” for climate and geospatial data, with a modular architecture that supports adding future datasets under a consistent CLI/GUI interface. This design supports the long-term goal of enabling reproducible workflows where both acquisition and processing steps are defined as version-controlled commands rather than manual web interactions.

# Roadmap and future development

GridFlow is actively being expanded toward a general-purpose “downloader hub” for climate and geospatial datasets, where new sources can be added as independent modules while preserving a consistent user experience. Planned future releases will prioritize additional datasets, improved cross-platform packaging, and expanded preprocessing support. Figure 4 outlines a tentative development roadmap; timelines are subject to change depending on community feedback, available compute resources, and potential funding or sponsorship.

![Tentative development roadmap for future GridFlow releases, including planned datasets and processing modules.](gridflow_roadmap.png)

# Availability

GridFlow is released under the GNU Affero General Public License v3.0 (AGPLv3) and is available on GitHub at:

https://github.com/shahbhuwan/GridFlow

# Acknowledgements

The authors thank the open-source scientific Python community for foundational libraries that enable GridFlow's functionality, including Xarray, GeoPandas, and the broader NetCDF and geospatial ecosystem. The authors also acknowledge the data providers and maintainers of ESGF, PRISM Climate Group, ECMWF (ERA5), Copernicus DEM, and USGS datasets for making large-scale climate and geospatial resources publicly accessible.
