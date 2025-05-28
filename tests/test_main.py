import sys
import pytest
from unittest.mock import patch, MagicMock
from gridflow import __main__ as main_module

@pytest.mark.parametrize("args,expected_command", [
    (["download", "--demo"], "download_command"),
    (["download-cmip5", "--demo"], "download_cmip5_command"),
    (["download-prism", "--demo"], "download_prism_command"),
    (["crop", "--demo"], "crop_command"),
    (["clip", "--demo"], "clip_command"),
    (["catalog", "--demo"], "catalog_command"),
])
def test_main_commands_dispatch(args, expected_command):
    sys_argv_patch = ["gridflow"] + args
    with patch.object(sys, 'argv', sys_argv_patch), \
         patch("gridflow.__main__.print_intro"), \
         patch("gridflow.__main__.download_command") as mock_download, \
         patch("gridflow.__main__.download_cmip5_command") as mock_cmip5, \
         patch("gridflow.__main__.download_prism_command") as mock_prism, \
         patch("gridflow.__main__.crop_command") as mock_crop, \
         patch("gridflow.__main__.clip_command") as mock_clip, \
         patch("gridflow.__main__.catalog_command") as mock_catalog:

        main_module.main()

        command_map = {
            "download_command": mock_download,
            "download_cmip5_command": mock_cmip5,
            "download_prism_command": mock_prism,
            "crop_command": mock_crop,
            "clip_command": mock_clip,
            "catalog_command": mock_catalog
        }

        # Check the right command function was called
        for name, mock_func in command_map.items():
            if name == expected_command:
                mock_func.assert_called_once()
            else:
                mock_func.assert_not_called()
