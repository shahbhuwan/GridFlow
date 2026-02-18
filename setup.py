from pathlib import Path
from setuptools import setup, find_packages

VERSION = "1.0"


setup(
    name="gridflow",
    version=VERSION,
    packages=find_packages(),

    package_data={
        "gridflow": [
            "conus_border/*",
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
    description="A modular toolset for downloading and processing geospatial data.",
    python_requires=">=3.11,<3.15",
)
