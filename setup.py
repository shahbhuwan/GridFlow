from setuptools import setup, find_packages

VERSION = "1.0" 

with open('requirements.txt', 'r') as f:
    install_requires = f.read().strip().split('\n')

setup(
    name="gridflow",
    version=VERSION,
    packages=find_packages(),
    
    install_requires=install_requires,
    
    package_data={
        "gridflow": [
            "iowa_border/*",
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
