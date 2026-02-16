---
title: 'GridFlow: A modular high-performance toolkit for downloading and processing large-scale climate and geospatial datasets'

tags:
  - downloader
  - processor
  - GridFlow
  - CMIP5
  - CMIP6
  - CMIP7
  - ERA5
  - PRISM
  - DEM
  - NetCDF
  - geospatial
  - climate
  - hydrology
  - gridded datasets
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

date: 2026-02-16
bibliography: paper.bib
---

# Summary

Climate and geospatial research increasingly relies on large observational and model-derived datasets such as CMIP6, CMIP5, ERA5 reanalysis, PRISM climate records, and global digital elevation models (DEMs). While these resources are widely available through web portals and distributed archives, acquiring and preparing them for analysis remains a major bottleneck. Researchers frequently spend considerable time navigating search interfaces, handling authentication requirements, writing brittle download scripts, and performing repetitive post-processing steps such as cropping, clipping, unit conversion, and temporal aggregation.

``GridFlow`` is an open-source Python based toolkit that streamlines the complete workflow of climate and geospatial data preparation. It provides both a command-line interface (CLI) and a graphical user interface (GUI) to download major climate products and to process NetCDF datasets into analysis-ready subsets. GridFlow emphasizes modular design, parallel execution, and usability to support a broad user community, including researchers, students, and practitioners who need reliable access to large datasets without extensive custom scripting.

GridFlow is designed to be accessible to a wide range of users through both a GUI (Figure 1) and a command-line interface (Figure 2), enabling users to reproducibly download and process climate and geospatial datasets without the need for custom scripts.

![GridFlow GUI showing the unified workflow for dataset acquisition and preprocessing.](gridflow_gui.png)

![GridFlow CLI interface and built-in command help (`gridflow -h`).](gridflow_cli_help.png)

# Statement of need

Despite the continued growth of publicly available climate archives, data acquisition and processing remain disproportionately time-consuming compared to downstream analysis. For example, Earth System Grid Federation (ESGF) portals supporting CMIP5/CMIP6 provide powerful search tools, but workflows often involve repeated manual filtering, pagination, and batch downloads. Similarly, reanalysis data systems can impose account setup, API keys, or queue-based access patterns that complicate reproducible retrieval.

In many applied workflows (e.g., hydrology, water resources, agriculture, and land-surface modeling), users require climate variables for specific regions, watersheds, or administrative boundaries and often need derived temporal summaries (e.g., monthly means, seasonal sums). These tasks are commonly addressed through ad-hoc scripts using general-purpose scientific Python tools such as Xarray [@xarray] and GeoPandas [@geopandas]. However, implementing robust pipelines across multiple datasets and formats can create barriers to entry for new users and reduce reproducibility across projects.

GridFlow addresses this gap by offering a single, consistent interface for acquiring and preparing multiple widely used climate and geospatial datasets. The toolkit couples high-level download modules with post-processing utilities to produce analysis-ready NetCDF outputs and metadata summaries, enabling rapid adoption in research and educational contexts.

# Software design

GridFlow’s architecture was driven by the need to balance scalability for large, distributed datasets with accessibility for interdisciplinary researchers who may lack advanced software engineering backgrounds. A primary design trade-off was choosing between a purely programmatic API versus a coupled Command-Line Interface (CLI) and Graphical User Interface (GUI). We opted to implement both, driven by a unified JSON configuration backend. The GUI lowers the barrier to entry for students and domain scientists, while the CLI allows for seamless integration into High-Performance Computing (HPC) environments and automated job schedulers.

GridFlow is organized into two primary capability groups: (1) data downloading modules and (2) processing modules.

## Downloading modules

GridFlow provides dedicated download modules for:

- **CMIP6 climate model data** via ESGF search and retrieval
- **CMIP5 climate model data** via ESGF search and retrieval
- **ERA5 reanalysis data** accessed directly from cloud-hosted sources
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

These operations are frequently required in climate impact studies, watershed-based modeling, and regional climate analyses. To handle the inherent instability of remote data portals (e.g., ESGF node timeouts or rate limits), GridFlow implements a robust asynchronous download manager with built-in retry logic and checksum validation. Furthermore, rather than reinventing array operations, the processing modules act as standardized, memory-safe wrappers around `Xarray` [@xarray] and `GeoPandas` [@geopandas]. This design choice ensures that complex operations are executed efficiently without requiring the user to write and maintain boilerplate code. Ultimately, GridFlow's configuration-driven design ensures that every step is explicitly documented and fully reproducible.

# State of the field

GridFlow addresses a recurring challenge in climate and geospatial workflows: while many high-value datasets are publicly available, the end-to-end process of discovering, downloading, organizing, and preparing them for analysis remains fragmented across portals, archive-specific clients, and custom scripts. For example, specialized tools like `weathercan` [@LaZerte2018] provide excellent, streamlined access to specific national repositories (such as Environment and Climate Change Canada), but researchers requiring multi-source inputs must still weave together several disparate APIs and packages. Users often combine these independent utilities for data acquisition with separate tools for preprocessing (e.g., cropping, clipping, aggregation), which ultimately reduces reproducibility and increases maintenance effort.

While the Python ecosystem offers robust downstream climate analytics libraries such as `xclim` [@Bourgault2023] and `xCDAT` [@Vo2024], alongside comprehensive NetCDF post-processing packages like `nctoolkit` [@Wilson2023], these tools inherently assume the data is already locally available or in a standardized format. Specifically, `xclim` facilitates the computation of climate indicators and statistical adjustments, and `xCDAT` streamlines common analysis operations like spatial and temporal averaging on existing xarray objects. Similarly, `nctoolkit` provides an intuitive interface for tasks like regridding and subsetting built on top of the Climate Data Operators (CDO) library. GridFlow instead focuses on the upstream bottleneck: the scalable acquisition and standardized preprocessing of large gridded datasets, seamlessly bridging the gap between raw archives and analysis-ready formats.

Several excellent tools support accessing climate archives through cloud-native workflows and metadata catalogs. For example, `intake-esm` [@intakeesm] enables the discovery and loading of CMIP-style collections through structured catalogs. Frameworks like `SlideRule` [@Shean2023] offer rapid, scalable on-demand processing in the cloud, returning high-level point cloud products directly to the user. However, these approaches typically assume users already operate within specific cloud ecosystems or target specific satellite missions, and do not provide a unified, user-facing pipeline for bulk downloading and local preparation across heterogeneous, non-cloud-optimized repositories.

In the broader geospatial and hydrological modeling domains, tools such as `Geodata-Harvester` [@Haan2023] jumpstart spatial data extraction to generate ready-made datasets for machine learning models, and `HydroMT` [@Eilander2023] automates the reproducible building of model instances from available data. When evaluating the software landscape, a "build versus contribute" decision was made to develop GridFlow independently. Extending existing domain-specific tools like `HydroMT` or `Geodata-Harvester` to act as generalized bulk downloaders for massive, heterogeneous archives (like CMIP5/6 or ERA5) would have unnecessarily bloated their core missions. GridFlow specifically fills this generalized upstream niche.

GridFlow also contrasts with interactive desktop GIS software such as QGIS. While QGIS offers extensive graphical workflows for exploratory analysis, it is not designed for automated, large-scale processing across thousands of NetCDF files, nor does it natively integrate bulk retrieval from multiple climate repositories. GridFlow bridges this gap by offering both a CLI and GUI that expose equivalent configuration options, enabling automated workflows to be repeated and shared as version-controlled commands.

Overall, GridFlow contributes a modular and extensible “downloader hub” that unites acquisition and spatial-temporal preprocessing under a single interface. Its design supports both global-scale datasets and subset-based workflows, establishing reproducible preparation pipelines while significantly reducing the need for archive-specific ad-hoc scripts.

# Research impact statement

The complexity of data acquisition often dictates the scope of climate research. By lowering the technical barrier to accessing high-value datasets like CMIP6, ERA5, and PRISM, GridFlow enables a broader community of researchers—including students, interdisciplinary scientists, and those with limited computational resources—to incorporate state-of-the-art climate data into their work.

Evidence of GridFlow’s near-term significance and utility is demonstrated by its active deployment in applied environmental studies. For instance, GridFlow was utilized to complete an NRCS-CPF project ("Building Understanding and Capacity for the Use of Climate Projections in Conservation Planning and Implementation"), where it facilitated the seamless integration of climate projections into NRCS conservation planning tools to assess climate impacts. Additionally, the software is central to an ongoing Water Resources Center (WRC) project, where it is used to systematically acquire and downscale high-resolution climate projections. This preprocessed data is necessary to drive physical hydrologic models (like WEPP) to simulate runoff, soil erosion, and water quality under historical and future climate scenarios.

Specifically, GridFlow impacts research by:
1. **Enhancing Reproducibility:** By replacing manual downloads and ad-hoc scripts with version-controlled configuration files (`config.json`) and CLI commands, GridFlow allows complex data preparation workflows to be fully audited and reproduced.
2. **Standardizing Preprocessing:** Common operations like regridding, unit conversion, and temporal aggregation are performed using standardized, tested implementations, reducing the risk of errors associated with custom post-processing scripts.
3. **Accelerating Discovery:** Researchers can rapidly prototype analysis on local subsets of data (e.g., a specific watershed or time period) without needing to download and manage entire global archives, significantly reducing the "time-to-analysis."

We anticipate that GridFlow will be particularly valuable in hydrology, ecology, and agricultural modeling, where climate data is a critical input but often not the primary focus of the research. Reflecting its growing utility, we are increasingly being approached by other laboratories at academic institutions to develop and add specific, domain-relevant data modules to the GridFlow ecosystem.

# Roadmap and future development

GridFlow is actively being expanded toward a general-purpose “downloader hub” for climate and geospatial datasets, where new sources can be added as independent modules while preserving a consistent user experience. Planned future releases will prioritize additional datasets, improved cross-platform packaging, and expanded preprocessing support. Figure 4 outlines a tentative development roadmap; timelines are subject to change depending on community feedback, available compute resources, and potential funding or sponsorship.

![Tentative development roadmap for future GridFlow releases, including planned datasets and processing modules.](gridflow_roadmap.png)

# AI usage disclosure

Generative AI tools (specifically, Gemini v3.0 pro) were employed during the development of GridFlow to assist with code refactoring, generating boilerplate for new download modules, and scaffolding unit tests. AI tools were also used for copy-editing and drafting sections of the documentation and this manuscript. All AI-generated code was thoroughly reviewed, manually tested, and validated against standard climate datasets to ensure correctness and adherence to the project's architecture. The authors reviewed, edited, and validated all AI-assisted text outputs, making all core design and architectural decisions. The human authors bear full responsibility for the accuracy, originality, and validity of the software and this publication.

# Availability

GridFlow (version 1.0) is released under the GNU Affero General Public License v3.0 (AGPLv3) and is available on GitHub at:

https://github.com/shahbhuwan/GridFlow

# Acknowledgements

The authors thank the open-source scientific Python community for foundational libraries that enable GridFlow's functionality, including Xarray, GeoPandas, and the broader NetCDF and geospatial ecosystem. The authors also acknowledge the data providers and maintainers of ESGF, PRISM Climate Group, ECMWF (ERA5), Copernicus DEM, and USGS datasets for making large-scale climate and geospatial resources publicly accessible.

This work was supported by the U.S. Department of Agriculture (USDA) Natural Resources Conservation Service (NRCS) under Cooperative Agreement No. NR243A750008C001.