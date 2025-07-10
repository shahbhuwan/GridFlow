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

from .prism_downloader import add_arguments as prism_add_arguments, main as prism_main
from .dem_downloader import add_arguments as dem_add_arguments, main as dem_main
from .cmip5_downloader import add_arguments as cmip5_add_arguments, main as cmip5_main
from .cmip6_downloader import add_arguments as cmip6_add_arguments, main as cmip6_main

prism_downloader = type('PrismDownloader', (), {'add_arguments': prism_add_arguments, 'main': prism_main})
dem_downloader = type('DemDownloader', (), {'add_arguments': dem_add_arguments, 'main': dem_main})
cmip5_downloader = type('Cmip5Downloader', (), {'add_arguments': cmip5_add_arguments, 'main': cmip5_main})
cmip6_downloader = type('Cmip6Downloader', (), {'add_arguments': cmip6_add_arguments, 'main': cmip6_main})