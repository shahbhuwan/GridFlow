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

from .crop_netcdf import add_arguments as crop_add_arguments, main as crop_main
from .clip_netcdf import add_arguments as clip_add_arguments, main as clip_main
from .unit_convert import add_arguments as unit_convert_add_arguments, main as unit_convert_main
from .temporal_aggregate import add_arguments as temporal_aggregate_add_arguments, main as temporal_aggregate_main
from .catalog_generator import add_arguments as catalog_add_arguments, main as catalog_main

crop_netcdf = type('CropNetCDF', (), {'add_arguments': crop_add_arguments, 'main': crop_main})
clip_netcdf = type('ClipNetCDF', (), {'add_arguments': clip_add_arguments, 'main': clip_main})
unit_convert = type('UnitConvert', (), {'add_arguments': unit_convert_add_arguments, 'main': unit_convert_main})
temporal_aggregate = type('TemporalAggregate', (), {'add_arguments': temporal_aggregate_add_arguments, 'main': temporal_aggregate_main})
catalog_generator = type('CatalogGenerator', (), {'add_arguments': catalog_add_arguments, 'main': catalog_main})