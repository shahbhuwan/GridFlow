from setuptools import setup, find_packages
import os

readme_path = "README.md"
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "A library and GUI to download, crop, clip, and catalog CMIP5, CMIP6, and PRISM climate data with deduplication, error handling, and retry capabilities"

setup(
    name="gridflow",
    version="1.0",
    description="A library and GUI to download, crop, clip, and catalog CMIP5, CMIP6, and PRISM climate data with deduplication, error handling, and retry capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Bhuwan Shah",
    author_email="bshah@iastate.edu",
    url="https://github.com/shahbhuwan/GridFlow",
    license="AGPL-3.0-or-later",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    package_data={
        "gridflow": [
            "iowa_border/*.CPG",
            "iowa_border/*.dbf",
            "iowa_border/*.prj",
            "iowa_border/*.sbn",
            "iowa_border/*.sbx",
            "iowa_border/*.shp",
            "iowa_border/*.shp.xml",
            "iowa_border/*.shx",
            "gridflow_logo.png",
            "gridflow_logo.ico",
        ],
        "gui": [
            "vocab/*.json",
        ],
    },
    install_requires=[
        "requests>=2.28.0,<3.0",
        "netCDF4>=1.6.0,<2.0",
        "numpy>=1.25.0,<2.0",
        "geopandas>=0.10.0,<1.0",
        "fiona>=1.8.0,<2.0",  # Added
        "PyQt5>=5.15.9,<6.0",
        "python-dateutil>=2.8.0,<3.0",
        "shapely>=1.7.1,<2.0",
        "pyproj>=2.6.1.post1,<4.0",
    ],
    extras_require={
        "test": [
            "pytest>=8.3.2",
            "pytest-cov>=5.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "gridflow = gridflow.entry:main",  # Changed from gridflow_entry to gridflow.entry
        ],
    }
)