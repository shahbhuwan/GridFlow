import unittest
import json
import logging
import requests
from io import BytesIO
from pathlib import Path
from threading import Lock
from unittest.mock import patch, Mock, MagicMock
from concurrent.futures import ThreadPoolExecutor
from gridflow.cmip5_downloader import FileManager, QueryHandler, Downloader, load_config, parse_file_time_range, run_download
import sys

class TestDownloader(unittest.TestCase):
    def setUp(self):
        self.download_dir = "/tmp/downloads"
        self.metadata_dir = "/tmp/metadata"
        self.file_manager = FileManager(self.download_dir, self.metadata_dir, "flat", prefix="test_", metadata_prefix="meta_")
        self.query_handler = QueryHandler()
        self.downloader = Downloader(
            self.file_manager, max_workers=2, retries=1, timeout=5, max_downloads=10,
            username="test_user", password="test_pass", verify_ssl=True
        )
        logging.getLogger().setLevel(logging.CRITICAL)  # Suppress logging during tests

    def tearDown(self):
        self.downloader.shutdown()

    @patch('pathlib.Path.mkdir')
    def test_file_manager_init(self, mock_mkdir):
        fm = FileManager("/test/downloads", "/test/metadata", "flat")
        mock_mkdir.assert_called()
        self.assertEqual(fm.download_dir, Path("/test/downloads"))
        self.assertEqual(fm.metadata_dir, Path("/test/metadata"))
        self.assertEqual(fm.save_mode, "flat")

    @patch('pathlib.Path.mkdir')
    def test_file_manager_init_failure(self, mock_mkdir):
        mock_mkdir.side_effect = PermissionError("Access denied")
        with self.assertRaises(SystemExit):
            FileManager("/test/downloads", "/test/metadata", "flat")

    @patch('pathlib.Path.mkdir')
    def test_file_manager_init_general_exception(self, mock_mkdir):
        mock_mkdir.side_effect = Exception("Unexpected failure")
        with self.assertRaises(SystemExit):
            FileManager("/invalid_path", "/invalid_meta", "flat")

    def test_file_manager_get_output_path_flat(self):
        file_info = {
            'title': 'test_file.nc',
            'institute': ['CCCMA'],
            'variable': ['tas'],
            'model': ['CanESM2']
        }
        path = self.file_manager.get_output_path(file_info)
        expected = Path(self.download_dir) / "test_CCCMA_250km_test_file.nc"
        self.assertEqual(path, expected)

    @patch('pathlib.Path.mkdir')
    def test_file_manager_get_output_path_hierarchical(self, mock_mkdir):
        fm = FileManager(self.download_dir, self.metadata_dir, "hierarchical")
        file_info = {
            'title': 'test_file.nc',
            'institute': ['CCCMA'],
            'variable': ['tas'],
            'model': ['CanESM2']
        }
        path = fm.get_output_path(file_info)
        expected = Path(self.download_dir) / "tas/250km/CCCMA/test_file.nc"
        self.assertEqual(path, expected)
        mock_mkdir.assert_called()

    def test_file_manager_output_path_unknown_resolution(self):
        file_info = {
            'title': 'test_file.nc',
            'institute': ['CCCMA'],
            'variable': ['tas'],
            'model': ['UnknownModel']
        }
        path = self.file_manager.get_output_path(file_info)
        self.assertIn("unknown", str(path))

    def test_file_manager_hierarchical_no_resolution(self):
        fm = FileManager(self.download_dir, self.metadata_dir, "hierarchical")
        file_info = {
            'title': 'test_file.nc',
            'institute': ['CCCMA'],
            'variable': ['tas'],
            'model': ['UnknownModel']
        }
        path = fm.get_output_path(file_info)
        self.assertTrue(path.name == "test_file.nc")

    @patch('builtins.open', new_callable=MagicMock)
    def test_file_manager_save_metadata(self, mock_open):
        files = [{'title': 'test_file.nc'}]
        self.file_manager.save_metadata(files, "test_metadata.json")
        mock_open.assert_called_with(Path(self.metadata_dir) / "meta_test_metadata.json", 'w', encoding='utf-8')
        mock_open().__enter__().write.assert_called()

    @patch('builtins.open', new_callable=MagicMock)
    def test_file_manager_save_metadata_failure(self, mock_open):
        mock_open.side_effect = IOError("Write error")
        files = [{'title': 'test_file.nc'}]
        with self.assertLogs(level='ERROR'):
            self.file_manager.save_metadata(files, "test_metadata.json")

    def test_file_manager_save_metadata_error(self):
        with patch("builtins.open", side_effect=IOError("fail")):
            self.file_manager.save_metadata([{"title": "test_file.nc"}], "meta.json")

    def test_query_handler_build_query(self):
        params = {'variable': 'tas', 'model': 'CanESM2'}
        url = self.query_handler.build_query("https://test-node/esg-search/search", params)
        expected = ("https://test-node/esg-search/search?type=File&project=CMIP5&format=application/solr%2Bjson"
                    "&limit=1000&distrib=true&variable=tas&model=CanESM2")
        self.assertEqual(url, expected)

    @patch('requests.Session.get')
    def test_query_handler_fetch_datasets(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': {'docs': [{'id': 'file1', 'title': 'test_file.nc'}], 'numFound': 1}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        params = {'variable': 'tas'}
        files = self.query_handler.fetch_datasets(params, timeout=5)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]['title'], 'test_file.nc')

    @patch('requests.Session.get')
    def test_query_handler_fetch_datasets_node_failure(self, mock_get):
        mock_get.side_effect = requests.RequestException("Connection error")
        with self.assertLogs(level='ERROR'):
            files = self.query_handler.fetch_datasets({'variable': 'tas'}, timeout=5)
        self.assertEqual(files, [])

    @patch("requests.Session.get")
    def test_query_handler_fetch_from_node_http_error(self, mock_get):
        mock_get.side_effect = requests.RequestException("Boom")
        with self.assertRaises(requests.RequestException):
            self.query_handler._fetch_from_node("https://fake-node", {}, 5)

    def test_query_handler_no_nodes(self):
        qh = QueryHandler(nodes=[])
        results = qh.fetch_datasets({}, timeout=2)
        self.assertEqual(results, [])

    @patch('requests.Session.get')
    def test_query_handler_fetch_specific_file(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': {'docs': [{'title': 'test_file.nc'}], 'numFound': 1}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        file_info = {'title': 'test_file.nc', 'variable': ['tas'], 'model': ['CanESM2']}
        result = self.query_handler.fetch_specific_file(file_info, timeout=5)
        self.assertEqual(result['title'], 'test_file.nc')

    def test_downloader_init_with_auth(self):
        downloader = Downloader(self.file_manager, 2, 1, 5, 10, "user", "pass", True)
        self.assertEqual(downloader.session.auth, ("user", "pass"))
        self.assertEqual(downloader.max_workers, 2)
        self.assertEqual(downloader.retries, 1)

    def test_openid_warning(self):
        with self.assertLogs(level="WARNING") as cm:
            Downloader(self.file_manager, 2, 1, 5, 5, None, None, True, openid="test_openid")
        self.assertTrue(any("OpenID provided" in msg for msg in cm.output))

    @patch('builtins.open', new_callable=MagicMock)
    @patch('requests.Session.get')
    @patch('pathlib.Path.open')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.rename')
    def test_downloader_download_file_success(self,
                                             mock_rename,
                                             mock_unlink,
                                             mock_exists,
                                             mock_path_open,
                                             mock_get,
                                             mock_builtin_open):
        mock_response = Mock()
        mock_response.iter_content.return_value = [b'data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        mock_exists.return_value = False
        write_handle = MagicMock()
        write_handle.write = Mock()
        mock_path_open.return_value = write_handle
        read_handle = MagicMock()
        read_handle.read.return_value = b"data"
        mock_builtin_open.return_value.__enter__.return_value = read_handle
        file_info = {
            'title': 'test_file.nc',
            'url': ['https://test.com/file|HTTPServer'],
            'checksum': ['a1b2c3'],
            'checksum_type': ['sha256'],
            'institute': ['CCCMA'],
            'variable': ['tas'],
            'model': ['CanESM2']
        }
        with patch('hashlib.sha256') as mock_sha256:
            sha_obj = Mock()
            sha_obj.update = Mock()
            sha_obj.hexdigest.return_value = 'a1b2c3'
            mock_sha256.return_value = sha_obj
            path, failed = self.downloader.download_file(file_info)
        expected = str(Path(self.download_dir) / "test_CCCMA_250km_test_file.nc")
        self.assertEqual(path, expected)
        self.assertIsNone(failed)
        mock_rename.assert_called()

    @patch('requests.Session.get')
    @patch('pathlib.Path.open')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.unlink')
    def test_downloader_download_file_checksum_mismatch(self, mock_unlink, mock_exists, mock_open, mock_get):
        mock_response = Mock()
        mock_response.iter_content.return_value = [b'data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        mock_exists.return_value = False
        mock_file = MagicMock()
        mock_file.write = Mock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_open.return_value = mock_file
        file_info = {
            'title': 'test_file.nc',
            'url': ['https://test.com/file|HTTPServer'],
            'checksum': ['a1b2c3'],
            'checksum_type': ['sha256'],
            'institute': ['CCCMA'],
            'variable': ['tas'],
            'model': ['CanESM2']
        }
        with patch('hashlib.sha256') as mock_sha256:
            mock_sha256_obj = Mock()
            mock_sha256_obj.update = Mock()
            mock_sha256_obj.hexdigest.return_value = 'wrong'
            mock_sha256.return_value = mock_sha256_obj
            path, failed = self.downloader.download_file(file_info)
        self.assertIsNone(path)
        self.assertEqual(failed, file_info)
        mock_unlink.assert_called()

    @patch('pathlib.Path.unlink', side_effect=OSError("unlink failed"))
    def test_download_file_unlink_exception(self, mock_unlink):
        file_info = {
            'title': 'test_file.nc',
            'url': ['https://test.com/test_file.nc|HTTPServer'],
            'checksum': ['bad'],
            'checksum_type': ['sha256'],
            'institute': ['CCCMA'],
            'variable': ['tas'],
            'model': ['CanESM2']
        }
        with patch.object(self.downloader, "verify_checksum", return_value=False):
            with patch("pathlib.Path.exists", return_value=True):
                with self.assertLogs(level="ERROR") as cm:
                    self.downloader.download_file(file_info)
        self.assertTrue(any("Failed to remove existing file" in msg for msg in cm.output))

    @patch('builtins.open', new_callable=MagicMock)
    def test_download_file_write_chunk_none(self, mock_open):
        file_info = {
            'title': 'test_file.nc',
            'url': ['https://test.com/test_file.nc|HTTPServer'],
            'checksum': ['abc'],
            'checksum_type': ['sha256'],
            'institute': ['CCCMA'],
            'variable': ['tas'],
            'model': ['CanESM2']
        }
        response = Mock()
        response.iter_content.return_value = [None]
        response.raise_for_status.return_value = None
        with patch("requests.Session.get", return_value=response):
            path, failed = self.downloader.download_file(file_info)
        self.assertIsNone(path)
        self.assertEqual(failed, file_info)
        mock_open.assert_not_called()  # No file should be written

    def test_downloader_download_file_invalid_url_format(self):
        file_info = {
            'title': 'test_file.nc',
            'url': ['ftp://invalid'],
            'checksum': ['x'],
            'checksum_type': ['sha256'],
            'institute': ['CCCMA'],
            'variable': ['tas'],
            'model': ['CanESM2']
        }
        path, failed = self.downloader.download_file(file_info)
        self.assertIsNone(path)
        self.assertEqual(failed, file_info)

    def test_downloader_download_file_invalid_metadata(self):
        file_info = {"url": []}
        path, failed = self.downloader.download_file(file_info)
        self.assertIsNone(path)
        self.assertIsNotNone(failed)

    def test_downloader_download_file_checksum_fail_all_retries(self):
        file_info = {
            'title': 'test_file.nc',
            'url': ['https://test.com/test_file.nc|HTTPServer'],
            'checksum': ['x'],
            'checksum_type': ['sha256'],
            'institute': ['CCCMA'],
            'variable': ['tas'],
            'model': ['CanESM2']
        }
        with patch.object(self.downloader, "verify_checksum", return_value=False):
            with patch("requests.Session.get") as mock_get:
                mock_response = Mock()
                mock_response.iter_content.return_value = [b"x"]
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response
                path, failed = self.downloader.download_file(file_info)
        self.assertIsNone(path)
        self.assertEqual(failed, file_info)

    @patch('concurrent.futures.as_completed')
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_downloader_download_all(self, mock_executor, mock_as_completed):
        mock_future = Mock()
        mock_future.result.return_value = ("path/to/file.nc", None)
        mock_executor.return_value.__enter__.return_value.submit.side_effect = [mock_future]
        mock_as_completed.return_value = [mock_future]
        files = [{
            'title': 'test_file.nc',
            'url': ['https://test.com/file|HTTPServer'],
            'institute': ['CCCMA'],
            'variable': ['tas'],
            'model': ['CanESM2']
        }]
        with patch.object(self.downloader, 'download_file', return_value=("path/to/file.nc", None)) as mock_download_file:
            downloaded, failed = self.downloader.download_all(files)
        self.assertEqual(len(downloaded), 1)
        self.assertEqual(len(failed), 0)
        self.assertEqual(self.downloader.successful_downloads, 1)
        mock_download_file.assert_called_once_with(files[0])

    def test_downloader_download_all_zero_files(self):
        downloaded, failed = self.downloader.download_all([])
        self.assertEqual(downloaded, [])
        self.assertEqual(failed, [])

    @patch.object(Downloader, 'download_file', side_effect=Exception("boom"))
    def test_downloader_download_all_exception_during_future(self, mock_download_file):
        files = [{'title': 'test_file.nc', 'url': ['https://test.com/test_file.nc|HTTPServer'], 'institute': ['CCCMA'], 'variable': ['tas'], 'model': ['CanESM2']}]
        downloaded, failed = self.downloader.download_all(files)
        self.assertEqual(len(failed), 1)

    @patch('concurrent.futures.as_completed')
    @patch('concurrent.futures.ThreadPoolExecutor')
    @patch.object(QueryHandler, 'fetch_specific_file')
    def test_downloader_retry_failed(self, mock_fetch, mock_executor, mock_as_completed):
        mock_future = Mock()
        mock_future.result.return_value = ("path/to/file.nc", None)
        mock_executor.return_value.__enter__.return_value.submit.side_effect = [mock_future]
        mock_as_completed.return_value = [mock_future]
        mock_fetch.return_value = {
            'title': 'test_file.nc',
            'url': ['https://test.com/file|HTTPServer'],
            'institute': ['CCCMA'],
            'variable': ['tas'],
            'model': ['CanESM2']
        }
        failed_files = [{'title': 'test_file.nc'}]
        with patch.object(self.downloader, 'download_file', return_value=("path/to/file.nc", None)) as mock_download_file:
            downloaded, failed = self.downloader.retry_failed(failed_files)
        self.assertEqual(len(downloaded), 1)
        self.assertEqual(len(failed), 0)
        mock_download_file.assert_called()

    @patch.object(QueryHandler, 'fetch_specific_file', return_value=None)
    def test_downloader_retry_failed_missing_metadata(self, mock_fetch):
        failed_files = [{'title': 'test_file.nc'}]
        downloaded, still_failed = self.downloader.retry_failed(failed_files)
        self.assertEqual(downloaded, [])
        self.assertEqual(still_failed, failed_files)

    def test_retry_failed_no_files(self):
        downloaded, failed = self.downloader.retry_failed([])
        self.assertEqual(downloaded, [])
        self.assertEqual(failed, [])

    def test_retry_failed_exception_during_future(self):
        file_info = {'title': 'test_file.nc', 'url': ['https://test.com/test_file.nc|HTTPServer'], 'institute': ['CCCMA'], 'variable': ['tas'], 'model': ['CanESM2']}
        self.downloader.executor = Mock()
        future = Mock()
        future.result.side_effect = Exception("boom")
        self.downloader.pending_futures = [future]
        self.downloader.query_handler.fetch_specific_file = lambda x, y: x
        result = self.downloader.retry_failed([file_info])
        self.assertEqual(len(result[1]), 1)

    @patch('builtins.open', new_callable=MagicMock)
    def test_load_config(self, mock_open):
        mock_open().__enter__.return_value.read.return_value = '{"project": "CMIP5"}'
        config = load_config("config.json")
        self.assertEqual(config, {"project": "CMIP5"})

    @patch('builtins.open', new_callable=MagicMock)
    def test_load_config_valid(self, mock_open):
        mock_open().__enter__.return_value.read.return_value = '{"project": "CMIP5"}'
        config = load_config("config.json")
        self.assertEqual(config, {"project": "CMIP5"})

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_config_failure(self, mock_open):
        with self.assertRaises(SystemExit):
            load_config("nonexistent.json")

    @patch("builtins.open", side_effect=FileNotFoundError("Not found"))
    def test_load_config_missing_file(self, mock_open):
        with self.assertRaises(SystemExit):
            load_config("missing.json")

    def test_parse_file_time_range(self):
        filename = "tas_mon_CanESM2_historical_r1i1p1_18500101-20051231.nc"
        start, end = parse_file_time_range(filename)
        self.assertEqual(start, "1850-01-01")
        self.assertEqual(end, "2005-12-31")

    def test_parse_file_time_range_month_format(self):
        filename = "tas_mon_CanESM2_historical_r1i1p1_185001-200512.nc"
        start, end = parse_file_time_range(filename)
        self.assertEqual(start, "1850-01-01")
        self.assertEqual(end, "2005-12-01")

    def test_parse_file_time_range_invalid(self):
        filename = "tas_mon_CanESM2_historical_r1i1p1_invalid.nc"
        start, end = parse_file_time_range(filename)
        self.assertIsNone(start)
        self.assertIsNone(end)

    def test_parse_file_time_range_bad_format(self):
        filename = "invalid_file.nc"
        start, end = parse_file_time_range(filename)
        self.assertIsNone(start)
        self.assertIsNone(end)

    @patch('builtins.open', new_callable=MagicMock)
    def test_verify_checksum_md5_success(self, mock_open):
        mock_open.return_value.__enter__.return_value.read.return_value = b"data"
        file_info = {'checksum': ['5d41402abc4b2a76b9719d911017c592'], 'checksum_type': ['md5']}
        with patch('hashlib.md5') as mock_md5:
            mock_md5_obj = mock_md5.return_value
            mock_md5_obj.update = Mock()
            mock_md5_obj.hexdigest.return_value = '5d41402abc4b2a76b9719d911017c592'
            result = self.downloader.verify_checksum(Path("test_file.nc"), file_info)
            self.assertTrue(result)

    @patch('builtins.open', new_callable=MagicMock)
    def test_verify_checksum_mismatch(self, mock_open):
        mock_open.return_value.__enter__.return_value.read.return_value = b"corrupt"
        file_info = {'checksum': ['bad'], 'checksum_type': ['sha256']}
        with patch('hashlib.sha256') as mock_sha256:
            mock_sha256_obj = mock_sha256.return_value
            mock_sha256_obj.hexdigest.return_value = 'wrong'
            mock_sha256_obj.update = Mock()
            result = self.downloader.verify_checksum(Path("test_file.nc"), file_info)
            self.assertFalse(result)

    @patch('builtins.open', side_effect=IOError("File read error"))
    def test_downloader_verify_checksum_file_open_error(self, mock_open):
        file_info = {'checksum': ['123'], 'checksum_type': ['sha256']}
        result = self.downloader.verify_checksum(Path("test_file.nc"), file_info)
        self.assertFalse(result)

    @patch('builtins.open', new_callable=MagicMock)
    def test_downloader_verify_checksum_unsupported_type(self, mock_open):
        mock_open.return_value.__enter__.return_value.read.return_value = b"data"
        file_info = {'checksum': ['123'], 'checksum_type': ['sha512']}
        result = self.downloader.verify_checksum(Path("test_file.nc"), file_info)
        self.assertTrue(result)  # Unsupported type logs warning and returns True

    def test_shutdown_executor_cleanup(self):
        mock_executor = Mock()
        self.downloader.executor = mock_executor
        future = Mock()
        self.downloader.pending_futures = [future]
        self.downloader.shutdown()
        future.cancel.assert_called()
        mock_executor.shutdown.assert_called_with(wait=False)

    # Fixed Tests Below

    @patch('gridflow.cmip5_downloader.load_config')
    @patch('gridflow.cmip5_downloader.QueryHandler')
    @patch('gridflow.cmip5_downloader.FileManager')
    @patch('gridflow.cmip5_downloader.Downloader')
    def test_run_download(self, mock_downloader, mock_file_manager, mock_query_handler, mock_load_config):
        # Mock args with stop_event to avoid early exits
        mock_args = Mock(
            retry_failed=None,
            config=None,
            project="CMIP5",
            variable="tas",
            model=None,
            experiment=None,
            frequency=None,
            ensemble=None,
            institute=None,
            latest=False,
            extra_params=None,
            demo=False,
            test=False,
            output_dir="/tmp/downloads",
            metadata_dir="/tmp/metadata",
            save_mode="flat",
            workers=2,
            retries=1,
            timeout=5,
            max_downloads=10,
            id=None,
            password=None,
            no_verify_ssl=False,
            openid=None,
            dry_run=False,
            stop_event=Mock(is_set=Mock(return_value=False))  # Mock stop_event
        )
        mock_query_handler.return_value.fetch_datasets.return_value = [{'title': 'test_file.nc'}]
        mock_downloader.return_value.download_all.return_value = (["path/to/file.nc"], [])

        run_download(mock_args)

        mock_query_handler.return_value.fetch_datasets.assert_called_once()  # Verify fetch_datasets was called
        mock_file_manager.return_value.save_metadata.assert_called_with([{'title': 'test_file.nc'}], "query_results.json")
        mock_downloader.return_value.download_all.assert_called_with([{'title': 'test_file.nc'}], phase="initial")
        mock_downloader.return_value.shutdown.assert_called()

    @patch('gridflow.cmip5_downloader.load_config')
    @patch('gridflow.cmip5_downloader.QueryHandler')
    @patch('gridflow.cmip5_downloader.FileManager')
    @patch('gridflow.cmip5_downloader.Downloader')
    def test_run_download_no_params(self, mock_downloader, mock_file_manager, mock_query_handler, mock_load_config):
        mock_args = Mock(
            retry_failed=None,
            config=None,
            project=None,
            variable=None,
            model=None,
            experiment=None,
            frequency=None,
            ensemble=None,
            institute=None,
            latest=False,
            extra_params=None,
            demo=False,
            test=False,
            output_dir="/tmp/downloads",
            metadata_dir="/tmp/metadata",
            save_mode="flat",
            workers=2,
            retries=1,
            timeout=5,
            max_downloads=10,
            id=None,
            password=None,
            no_verify_ssl=False,
            openid=None,
            dry_run=False,
            stop_event=Mock(is_set=Mock(return_value=False))
        )
        with self.assertLogs(level='ERROR') as cm:
            with self.assertRaises(SystemExit) as cm_exit:
                run_download(mock_args)
        self.assertEqual(cm_exit.exception.code, 1)
        self.assertTrue(any("No valid search parameters provided" in msg for msg in cm.output))

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('gridflow.cmip5_downloader.FileManager')
    @patch('gridflow.cmip5_downloader.Downloader')
    def test_run_download_retry_failed(self, mock_downloader, mock_file_manager, mock_open, mock_is_file, mock_exists):
        mock_args = Mock(
            retry_failed="failed.json",
            output_dir="/tmp/downloads",
            metadata_dir="/tmp/metadata",
            save_mode="flat",
            workers=2,
            retries=1,
            timeout=5,
            max_downloads=10,
            id=None,
            password=None,
            no_verify_ssl=False,
            openid=None,
            project="CMIP5",
            dry_run=False,
            stop_event=Mock(is_set=Mock(return_value=False))
        )
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps([{'title': 'test_file.nc'}])
        mock_downloader.return_value.download_all.return_value = ([], [{'title': 'test_file.nc'}])
        mock_downloader.return_value.retry_failed.return_value = ([], [{'title': 'test_file.nc'}])

        run_download(mock_args)

        mock_open.assert_called_once_with(Path("failed.json"), 'r', encoding='utf-8')  # Fixed mock_open call
        mock_file_manager.return_value.save_metadata.assert_any_call([{'title': 'test_file.nc'}], "failed_downloads.json")
        mock_file_manager.return_value.save_metadata.assert_any_call([{'title': 'test_file.nc'}], "failed_downloads_final.json")
        mock_downloader.return_value.retry_failed.assert_called_with([{'title': 'test_file.nc'}])
        mock_downloader.return_value.shutdown.assert_called()

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('gridflow.cmip5_downloader.FileManager')
    @patch('gridflow.cmip5_downloader.Downloader')
    def test_run_download_retry_failed_with_failures(self, mock_downloader, mock_file_manager, mock_open, mock_is_file, mock_exists):
        mock_args = Mock(
            retry_failed="failed.json",
            output_dir="/tmp/downloads",
            metadata_dir="/tmp/metadata",
            save_mode="flat",
            workers=2,
            retries=1,
            timeout=5,
            max_downloads=10,
            id=None,
            password=None,
            no_verify_ssl=False,
            openid=None,
            project="CMIP5",
            dry_run=False,
            stop_event=Mock(is_set=Mock(return_value=False))
        )
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps([{'title': 'test_file.nc'}])
        mock_downloader.return_value.download_all.return_value = ([], [{'title': 'test_file.nc'}])
        mock_downloader.return_value.retry_failed.return_value = ([], [{'title': 'test_file.nc'}])

        run_download(mock_args)

        mock_open.assert_called_once_with(Path("failed.json"), 'r', encoding='utf-8')  # Fixed mock_open call
        mock_file_manager.return_value.save_metadata.assert_any_call([{'title': 'test_file.nc'}], "failed_downloads.json")
        mock_file_manager.return_value.save_metadata.assert_any_call([{'title': 'test_file.nc'}], "failed_downloads_final.json")
        mock_downloader.return_value.retry_failed.assert_called_with([{'title': 'test_file.nc'}])
        mock_downloader.return_value.shutdown.assert_called()

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    def test_run_download_missing_retry_file(self, mock_is_file, mock_exists):
        mock_args = Mock(
            retry_failed="nonexistent.json",
            config=None,
            project="CMIP5",
            output_dir="/tmp/downloads",
            metadata_dir="/tmp/metadata",
            save_mode="flat",
            workers=1,
            retries=1,
            timeout=5,
            max_downloads=1,
            id=None,
            password=None,
            no_verify_ssl=False,
            openid=None,
            extra_params=None,
            latest=False,
            demo=False,
            test=False,
            variable="tas",
            model=None,
            experiment=None,
            frequency=None,
            ensemble=None,
            institute=None,
            dry_run=False,
            stop_event=Mock(is_set=Mock(return_value=False))
        )
        mock_exists.return_value = False  # Simulate missing file
        mock_is_file.return_value = False

        with self.assertLogs(level='ERROR') as cm:
            with self.assertRaises(SystemExit) as cm_exit:
                run_download(mock_args)
        self.assertEqual(cm_exit.exception.code, 1)
        self.assertTrue(any("Retry file nonexistent.json does not exist" in msg for msg in cm.output))

    @patch('gridflow.cmip5_downloader.load_config')
    @patch('gridflow.cmip5_downloader.QueryHandler')
    @patch('gridflow.cmip5_downloader.FileManager')
    @patch('gridflow.cmip5_downloader.Downloader')
    def test_run_download_dry_run(self, mock_downloader, mock_file_manager, mock_query_handler, mock_load_config):
        mock_args = Mock(
            retry_failed=None,
            config=None,
            project="CMIP5",
            output_dir="/tmp/downloads",
            metadata_dir="/tmp/metadata",
            save_mode="flat",
            workers=1,
            retries=1,
            timeout=5,
            max_downloads=1,
            id=None,
            password=None,
            no_verify_ssl=False,
            openid=None,
            latest=False,
            extra_params=None,
            demo=False,
            test=False,
            variable="tas",
            model=None,
            experiment=None,
            frequency=None,
            ensemble=None,
            institute=None,
            dry_run=True,
            stop_event=Mock(is_set=Mock(return_value=False))
        )
        mock_query_handler.return_value.fetch_datasets.return_value = [{"title": "test_file.nc"}]

        with self.assertLogs(level='INFO') as cm:
            with self.assertRaises(SystemExit) as cm_exit:
                run_download(mock_args)
        self.assertEqual(cm_exit.exception.code, 0)
        self.assertTrue(any("Dry run: Would download 1 files" in msg for msg in cm.output))

    @patch('gridflow.cmip5_downloader.load_config')
    def test_run_download_extra_params_invalid_json(self, mock_load_config):
        mock_args = Mock(
            retry_failed=None,
            config=None,
            project="CMIP5",
            output_dir="/tmp/downloads",
            metadata_dir="/tmp/metadata",
            save_mode="flat",
            workers=1,
            retries=1,
            timeout=5,
            max_downloads=1,
            id=None,
            password=None,
            no_verify_ssl=False,
            openid=None,
            latest=False,
            extra_params="{bad_json: true",
            demo=False,
            test=False,
            variable="tas",
            model=None,
            experiment=None,
            frequency=None,
            ensemble=None,
            institute=None,
            dry_run=False,
            stop_event=Mock(is_set=Mock(return_value=False))
        )
        with self.assertLogs(level='ERROR') as cm:
            with self.assertRaises(SystemExit) as cm_exit:
                run_download(mock_args)
        self.assertEqual(cm_exit.exception.code, 1)
        self.assertTrue(any("Invalid extra-params JSON" in msg for msg in cm.output))

if __name__ == '__main__':
    unittest.main()