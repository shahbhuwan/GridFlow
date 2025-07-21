# gridflow/cli.py
# Copyright (c) 2025 Bhuwan Shah
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
import argparse
from gridflow import __version__
from gridflow.download import (
    prism_downloader,
    dem_downloader,
    cmip5_downloader,
    cmip6_downloader,
    era5_downloader
)
from gridflow.processing import (
    crop_netcdf,
    clip_netcdf,
    unit_convert,
    temporal_aggregate,
    catalog_generator
)

def print_intro():
    """Display the GridFlow welcome banner."""
    banner = r"""
==============================================================================================
     ____      _     _ _____ _                
    / ___|_ __(_) __| |  ___| | _____      __ 
   | |  _| '__| |/ _` | |_  | |/ _ \ \ /\ / / 
   | |_| | |  | | (_| |  _| | | (_) \ V  V /  
    \____|_|  |_|\__,_|_|   |_|\___/ \_/\_/   

==============================================================================================
Welcome to GridFlow v{}! Copyright (c) 2025 Bhuwan Shah
Effortlessly download and process PRISM, DEM, CMIP5, and CMIP6 climate data.
Run `gridflow -h` for help or `gridflow prism --demo` to try a sample download.
==============================================================================================
""".format(__version__)
    print(banner)

class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter for consistent CLI help formatting."""
    def _format_args(self, action, default_metavar):
        get_metavar = self._metavar_formatter(action, default_metavar)
        return '%s' % get_metavar(1) if action.nargs is None else super()._format_args(action, default_metavar)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        option_strings = ', '.join(action.option_strings)
        return f'{option_strings} {self._format_args(action, action.dest.upper())}'

def create_parser():
    """Creates the main argument parser with subcommands for each tool."""
    parser = argparse.ArgumentParser(
        description=(
            "GridFlow: A modular toolset for downloading and processing geospatial data.\n"
            "Supports PRISM, DEM, CMIP5, and CMIP6 data downloads, as well as NetCDF processing tasks like cropping, clipping, unit conversion, temporal aggregation, and catalog generation."
        ),
        epilog=(
            "Examples:\n"
            "  gridflow prism --demo                   # Download sample PRISM data\n"
            "  gridflow dem --demo --api_key KEY       # Download sample DEM data\n"
            "  gridflow cmip5 --demo                   # Download sample CMIP5 data\n"
            "  gridflow cmip6 --demo                   # Download sample CMIP6 data\n"
            "  gridflow era5 --demo --api_key UID:KEY  # Download sample ERA5 data\n"
            "  gridflow crop --demo                    # Crop NetCDF files to sample bounds\n"
            "  gridflow clip --demo                    # Clip NetCDF files using Iowa shapefile\n"
            "  gridflow unit-convert --demo            # Convert units in NetCDF files\n"
            "  gridflow temporal-aggregate --demo      # Aggregate NetCDF files temporally\n"
            "  gridflow catalog --demo                 # Generate a sample catalog\n"
            "\nRun 'gridflow <command> -h' for detailed help on each command."
        ),
        formatter_class=CustomHelpFormatter
    )
    parser.add_argument('-v', '--version', action='version', version=f'GridFlow {__version__}')
    subparsers = parser.add_subparsers(title="commands", dest="command", help="Available tools", required=True)

    # PRISM Downloader
    prism_parser = subparsers.add_parser(
        "prism",
        help="Download PRISM climate data",
        description="Download daily PRISM climate data for variables like precipitation or temperature.",
        epilog="Example: gridflow prism --demo\nDownloads sample PRISM tmean data (4km, 2020-01-01 to 2020-01-05).",
        formatter_class=CustomHelpFormatter
    )
    prism_downloader.add_arguments(prism_parser)

    # DEM Downloader
    dem_parser = subparsers.add_parser(
        "dem",
        help="Download Digital Elevation Models via OpenTopography",
        description="Download DEM data (e.g., COP30, SRTM) for a specified bounding box using an OpenTopography API key.",
        epilog="Example: gridflow dem --demo --api_key YOUR_KEY\nDownloads sample COP30 DEM for Iowa.",
        formatter_class=CustomHelpFormatter
    )
    dem_downloader.add_arguments(dem_parser)

    # CMIP5 Downloader
    cmip5_parser = subparsers.add_parser(
        "cmip5",
        help="Download CMIP5 climate model data",
        description="Download CMIP5 climate model data from ESGF nodes based on specified parameters.",
        epilog="Example: gridflow cmip5 --demo\nDownloads sample CMIP5 tas data (CanESM2, historical).",
        formatter_class=CustomHelpFormatter
    )
    cmip5_downloader.add_arguments(cmip5_parser)

    # CMIP6 Downloader
    cmip6_parser = subparsers.add_parser(
        "cmip6",
        help="Download CMIP6 climate model data",
        description="Download CMIP6 climate model data from ESGF nodes based on specified parameters.",
        epilog="Example: gridflow cmip6 --demo\nDownloads sample CMIP6 tas data (HadGEM3-GC31-LL, hist-1950).",
        formatter_class=CustomHelpFormatter
    )
    cmip6_downloader.add_arguments(cmip6_parser)

    # ERA5 Downloader
    era5_parser = subparsers.add_parser(
        "era5",
        help="Download ERA5-Land climate data",
        formatter_class=CustomHelpFormatter
    )
    era5_downloader.add_arguments(era5_parser)

    # Crop NetCDF
    crop_parser = subparsers.add_parser(
        "crop",
        help="Crop NetCDF files to a spatial bounding box",
        description="Crop NetCDF files to a specified spatial bounding box defined by latitude and longitude.",
        epilog="Example: gridflow crop --demo\nCrops sample NetCDF files to a US bounding box.",
        formatter_class=CustomHelpFormatter
    )
    crop_netcdf.add_arguments(crop_parser)

    # Clip NetCDF
    clip_parser = subparsers.add_parser(
        "clip",
        help="Clip NetCDF files using a shapefile",
        description="Clip NetCDF files to a region defined by a shapefile, with optional buffering.",
        epilog="Example: gridflow clip --demo\nClips sample NetCDF files using Iowa shapefile.",
        formatter_class=CustomHelpFormatter
    )
    clip_netcdf.add_arguments(clip_parser)

    # Unit Convert
    unit_parser = subparsers.add_parser(
        "convert",
        help="Convert units in NetCDF files",
        description="Convert units of variables in NetCDF files (e.g., Kelvin to Celsius, flux to mm/day).",
        epilog="Example: gridflow unit-convert --demo\nConverts tas from Kelvin to Celsius in sample files.",
        formatter_class=CustomHelpFormatter
    )
    unit_convert.add_arguments(unit_parser)

    # Temporal Aggregate
    temporal_parser = subparsers.add_parser(
        "aggregate",
        help="Temporally aggregate NetCDF files",
        description="Aggregate NetCDF files to monthly, seasonal, or annual frequency using methods like mean or sum.",
        epilog="Example: gridflow temporal-aggregate --demo\nAggregates tas to monthly mean in sample files.",
        formatter_class=CustomHelpFormatter
    )
    temporal_aggregate.add_arguments(temporal_parser)

    # Catalog Generator
    catalog_parser = subparsers.add_parser(
        "catalog",
        help="Generate a catalog from NetCDF files",
        description="Generate a JSON catalog summarizing metadata from NetCDF files.",
        epilog="Example: gridflow catalog --demo\nGenerates a catalog for sample CMIP6 files.",
        formatter_class=CustomHelpFormatter
    )
    catalog_generator.add_arguments(catalog_parser)

    return parser

def main():
    """Main entry point for the GridFlow CLI."""
    print_intro()
    parser = create_parser()
    args = parser.parse_args()

    command_mapping = {
        "prism": prism_downloader.main,
        "dem": dem_downloader.main,
        "cmip5": cmip5_downloader.main,
        "cmip6": cmip6_downloader.main,
        "era5": era5_downloader.main,
        "crop": crop_netcdf.main,
        "clip": clip_netcdf.main,
        "convert": unit_convert.main,
        "aggregate": temporal_aggregate.main,
        "catalog": catalog_generator.main
    }

    try:
        command_func = command_mapping[args.command]
        command_func(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()