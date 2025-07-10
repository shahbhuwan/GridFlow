from setuptools import setup, find_packages

setup(
    name="gridflow",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "PyQt5>=5.15.0",
        "requests>=2.28.0",
        "numpy>=1.21.0",
        "netCDF4>=1.5.8",
        "geopandas>=0.10.0",
        "shapely>=1.8.0",
        "python-dateutil>=2.8.0",
        "cdsapi>=0.7.4"
    ],
    package_data={
        "gridflow": [
            "iowa_border/*",
            "gridflow_logo.png",
            "vocab/*.json"
        ]
    },
    entry_points={
        "console_scripts": [
            "gridflow=gridflow.cli:main",
            "gridflow-gui=gridflow.gui:main"
        ]
    },
    author="Bhuwan Shah",
    license="GNU AGPLv3",
    description="A modular toolset for downloading and processing geospatial data."
)