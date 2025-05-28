import unittest
import json
import logging
import requests
from pathlib import Path
from threading import Lock
from unittest.mock import patch, Mock, MagicMock
from concurrent.futures import ThreadPoolExecutor
from gridflow.cmip6_downloader import FileManager, QueryHandler, Downloader, load_config, parse_file_time_range, run_download
import sys

class TestDownloader(unittest.TestCase):
    def setUp(self):
        self.download_dir = "/tmp/downloads"
        self.metadata_dir = "/tmp/metadata"
        self.file_manager = FileManager(self.download_dir, self.metadata_dir, "flat", prefix="test_", metadata_prefix="meta_")
        self.query_handler = QueryHandler()
        self.downloader = Downloader(
            self.file_manager, max_workers=2, retries=2, timeout=5, max_downloads=10,
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

    def test_file_manager_get_output_path_flat(self):
        file_info = {
            'title': 'test_file.nc',
            'activity_id': ['ScenarioMIP'],
            'variable_id': ['tas'],
            'source_id': ['CanESM5'],
            'nominal_resolution': ['250 km']
        }
        path = self.file_manager.get_output_path(file_info)
        expected = Path(self.download_dir) / "test_ScenarioMIP_250km_test_file.nc"
        self.assertEqual(path, expected)

    @patch('pathlib.Path.mkdir')
    def test_file_manager_get_output_path_hierarchical(self, mock_mkdir):
        fm = FileManager(self.download_dir, self.metadata_dir, "hierarchical")
        file_info = {
            'title': 'test_file.nc',
            'activity_id': ['ScenarioMIP'],
            'variable_id': ['tas'],
            'source_id': ['CanESM5'],
            'nominal_resolution': ['250 km']
        }
        path = fm.get_output_path(file_info)
        expected = Path(self.download_dir) / "tas/250km/ScenarioMIP/test_file.nc"
        self.assertEqual(path, expected)
        mock_mkdir.assert_called()

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

    def test_query_handler_build_query(self):
        params = {'variable_id': 'tas', 'source_id': 'CanESM5'}
        url = self.query_handler.build_query("https://test-node/esg-search/search", params)
        expected = ("https://test-node/esg-search/search?type=File&project=CMIP6&format=application/solr%2Bjson"
                    "&limit=1000&distrib=true&variable_id=tas&source_id=CanESM5")
        self.assertEqual(url, expected)

    @patch('requests.Session.get')
    def test_query_handler_fetch_datasets(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': {'docs': [{'id': 'file1', 'title': 'test_file.nc'}], 'numFound': 1}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        params = {'variable_id': 'tas'}
        files = self.query_handler.fetch_datasets(params, timeout=5)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]['title'], 'test_file.nc')

    @patch('requests.Session.get')
    def test_query_handler_fetch_datasets_node_failure(self, mock_get):
        mock_get.side_effect = requests.RequestException("Connection error")
        with self.assertLogs(level='ERROR'):
            files = self.query_handler.fetch_datasets({'variable_id': 'tas'}, timeout=5)
        self.assertEqual(files, [])

    @patch('requests.Session.get')
    def test_query_handler_fetch_specific_file(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': {'docs': [{'title': 'test_file.nc'}], 'numFound': 1}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        file_info = {'title': 'test_file.nc', 'variable_id': ['tas'], 'source_id': ['CanESM5']}
        result = self.query_handler.fetch_specific_file(file_info, timeout=5)
        self.assertEqual(result['title'], 'test_file.nc')

    def test_downloader_init_with_auth(self):
        downloader = Downloader(self.file_manager, 2, 2, 5, 10, "user", "pass", True)
        self.assertEqual(downloader.session.auth, ("user", "pass"))
        self.assertEqual(downloader.max_workers, 2)
        self.assertEqual(downloader.retries, 2)

    @patch('requests.Session.get')
    @patch('pathlib.Path.open')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.rename')
    def test_downloader_download_file_success(self, mock_rename, mock_unlink, mock_exists, mock_open, mock_get):
        mock_response = Mock()
        mock_response.iter_content.return_value = [b'data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.write.side_effect = lambda chunk: None
        mock_file.read.return_value = b'data'
        mock_open.side_effect = [mock_file, mock_file]
        file_info = {
            'title': 'test_file.nc',
            'url': ['https://test.com/file|HTTPServer'],
            'checksum': ['a1b2c3'],
            'checksum_type': ['sha256'],
            'activity_id': ['ScenarioMIP'],
            'variable_id': ['tas'],
            'source_id': ['CanESM5'],
            'nominal_resolution': ['250 km']
        }
        with patch('hashlib.sha256') as mock_sha256:
            mock_sha256_obj = Mock()
            mock_sha256_obj.update = Mock()
            mock_sha256_obj.hexdigest.return_value = 'a1b2c3'
            mock_sha256.return_value = mock_sha256_obj
            path, failed = self.downloader.download_file(file_info)

        expected_path = str(Path(self.download_dir) / "test_ScenarioMIP_250km_test_file.nc")
        self.assertEqual(path, expected_path)
        self.assertIsNone(failed)
        mock_get.assert_called_with('https://test.com/file', stream=True, verify=True, timeout=(5, 1))
        mock_sha256_obj.update.assert_called_with(b'data')
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
        mock_file = Mock()
        mock_file.write.side_effect = lambda chunk: None
        mock_file.read.return_value = b'data'
        mock_open.side_effect = [mock_file, mock_file]
        file_info = {
            'title': 'test_file.nc',
            'url': ['https://test.com/file|HTTPServer'],
            'checksum': ['a1b2c3'],
            'checksum_type': ['sha256'],
            'activity_id': ['ScenarioMIP'],
            'variable_id': ['tas'],
            'source_id': ['CanESM5'],
            'nominal_resolution': ['250 km']
        }
        with patch('hashlib.sha256') as mock_sha256:
            mock_sha256_obj = Mock()
            mock_sha256_obj.update = Mock()
            mock_sha256_obj.hexdigest.return_value = 'wrong'
            mock_sha256.return_value = mock_sha256_obj
            path, failed = self.downloader.download_file(file_info)

        self.assertIsNone(path)
        self.assertEqual(failed, file_info)
        mock_get.assert_called_with('https://test.com/file', stream=True, verify=True, timeout=(5, 1))
        mock_sha256_obj.update.assert_called_with(b'data')
        mock_unlink.assert_called()


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
            'activity_id': ['ScenarioMIP'],
            'variable_id': ['tas'],
            'source_id': ['CanESM5'],
            'nominal_resolution': ['250 km']
        }]
        with patch.object(self.downloader, 'download_file', return_value=("path/to/file.nc", None)) as mock_download_file:
            downloaded, failed = self.downloader.download_all(files)
        self.assertEqual(len(downloaded), 1)
        self.assertEqual(len(failed), 0)
        self.assertEqual(self.downloader.successful_downloads, 1)
        mock_download_file.assert_called_once_with(files[0])

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
            'activity_id': ['ScenarioMIP'],
            'variable_id': ['tas'],
            'source_id': ['CanESM5'],
            'nominal_resolution': ['250 km']
        }
        failed_files = [{'title': 'test_file.nc'}]
        with patch.object(self.downloader, 'download_file', return_value=("path/to/file.nc", None)) as mock_download_file:
            downloaded, failed = self.downloader.retry_failed(failed_files)
        self.assertEqual(len(downloaded), 1)
        self.assertEqual(len(failed), 0)
        mock_download_file.assert_called()

    @patch('builtins.open', new_callable=MagicMock)
    def test_load_config(self, mock_open):
        mock_open().__enter__.return_value.read.return_value = '{"project": "CMIP6"}'
        config = load_config("config.json")
        self.assertEqual(config, {"project": "CMIP6"})

    @patch('builtins.open', new_callable=MagicMock)
    def test_load_config_failure(self, mock_open):
        mock_open.side_effect = FileNotFoundError
        with self.assertRaises(SystemExit):
            load_config("nonexistent.json")

    def test_parse_file_time_range(self):
        filename = "tas_day_CanESM5_ssp585_r1i1p1f1_20200101-20201231.nc"
        start, end = parse_file_time_range(filename)
        self.assertEqual(start, "2020-01-01")
        self.assertEqual(end, "2020-12-31")

    def test_parse_file_time_range_invalid(self):
        filename = "tas_day_CanESM5_ssp585_r1i1p1f1_invalid.nc"
        start, end = parse_file_time_range(filename)
        self.assertIsNone(start)
        self.assertIsNone(end)

    @patch('gridflow.cmip6_downloader.load_config')
    @patch('gridflow.cmip6_downloader.QueryHandler')
    @patch('gridflow.cmip6_downloader.FileManager')
    @patch('gridflow.cmip6_downloader.Downloader')
    def test_run_download(self, mock_downloader, mock_file_manager, mock_query_handler, mock_load_config):
        mock_args = Mock(
            retry_failed=None, config=None, project="CMIP6", activity=None, experiment=None,
            frequency=None, variable="tas", model=None, ensemble=None, institution=None,
            source_type=None, grid_label=None, resolution=None, latest=False,
            extra_params=None, demo=False, test=False, output_dir="/tmp/downloads",
            metadata_dir="/tmp/metadata", save_mode="flat", workers=2, retries=2,
            timeout=5, max_downloads=10, id=None, password=None, no_verify_ssl=False,
            openid=None, dry_run=False
        )
        mock_query_handler.return_value.fetch_datasets.return_value = [{'title': 'test_file.nc'}]
        mock_downloader.return_value.download_all.return_value = (["path/to/file.nc"], [])
        
        run_download(mock_args)
        
        mock_query_handler.return_value.fetch_datasets.assert_called()
        mock_file_manager.return_value.save_metadata.assert_called_with([{'title': 'test_file.nc'}], "query_results.json")
        mock_downloader.return_value.download_all.assert_called_with([{'title': 'test_file.nc'}], phase="initial")
        mock_downloader.return_value.shutdown.assert_called()

    @patch('gridflow.cmip6_downloader.load_config')
    @patch('gridflow.cmip6_downloader.QueryHandler')
    @patch('gridflow.cmip6_downloader.FileManager')
    @patch('gridflow.cmip6_downloader.Downloader')
    def test_run_download_no_params(self, mock_downloader, mock_file_manager, mock_query_handler, mock_load_config):
        mock_args = Mock(
            retry_failed=None, config=None, project=None, activity=None, experiment=None,
            frequency=None, variable=None, model=None, ensemble=None, institution=None,
            source_type=None, grid_label=None, resolution=None, latest=False,
            extra_params=None, demo=False, test=False, output_dir="/tmp/downloads",
            metadata_dir="/tmp/metadata", save_mode="flat", workers=2, retries=2,
            timeout=5, max_downloads=10, id=None, password=None, no_verify_ssl=False,
            openid=None, dry_run=False
        )
        with self.assertRaises(SystemExit) as cm:
            run_download(mock_args)
        self.assertEqual(cm.exception.code, 1)

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('gridflow.cmip6_downloader.FileManager')
    @patch('gridflow.cmip6_downloader.Downloader')
    def test_run_download_retry_failed(self, mock_downloader, mock_file_manager, mock_open, mock_is_file, mock_exists):
        mock_args = Mock(
            retry_failed="failed.json",
            output_dir="/tmp/downloads",
            metadata_dir="/tmp/metadata",
            save_mode="flat",
            workers=2,
            retries=2,
            timeout=5,
            max_downloads=10,
            id=None,
            password=None,
            no_verify_ssl=False,
            openid=None,
            project="CMIP6",
            dry_run=False
        )
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_open().__enter__.return_value.read.return_value = json.dumps([{'title': 'test_file.nc'}])
        mock_downloader.return_value.download_all.return_value = ([], [{'title': 'test_file.nc'}])
        mock_downloader.return_value.retry_failed.return_value = ([], [{'title': 'test_file.nc'}])

        run_download(mock_args)

        mock_open.assert_called_with(Path("failed.json"), 'r', encoding='utf-8')
        mock_file_manager.return_value.save_metadata.assert_any_call([{'title': 'test_file.nc'}], "failed_downloads.json")
        mock_file_manager.return_value.save_metadata.assert_any_call([{'title': 'test_file.nc'}], "failed_downloads_final.json")
        mock_downloader.return_value.retry_failed.assert_called_with([{'title': 'test_file.nc'}])
        mock_downloader.return_value.shutdown.assert_called()

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('gridflow.cmip6_downloader.FileManager')
    @patch('gridflow.cmip6_downloader.Downloader')
    def test_run_download_retry_failed_with_failures(self, mock_downloader, mock_file_manager, mock_open, mock_is_file, mock_exists):
        mock_args = Mock(
            retry_failed="failed.json",
            output_dir="/tmp/downloads",
            metadata_dir="/tmp/metadata",
            save_mode="flat",
            workers=2,
            retries=2,
            timeout=5,
            max_downloads=10,
            id=None,
            password=None,
            no_verify_ssl=False,
            openid=None,
            project="CMIP6",
            dry_run=False
        )
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_open().__enter__.return_value.read.return_value = json.dumps([{'title': 'test_file.nc'}])
        mock_downloader.return_value.download_all.return_value = ([], [{'title': 'test_file.nc'}])
        mock_downloader.return_value.retry_failed.return_value = ([], [{'title': 'test_file.nc'}])

        run_download(mock_args)

        mock_open.assert_called_with(Path("failed.json"), 'r', encoding='utf-8')
        mock_file_manager.return_value.save_metadata.assert_any_call([{'title': 'test_file.nc'}], "failed_downloads.json")
        mock_file_manager.return_value.save_metadata.assert_any_call([{'title': 'test_file.nc'}], "failed_downloads_final.json")
        mock_downloader.return_value.retry_failed.assert_called_with([{'title': 'test_file.nc'}])
        mock_downloader.return_value.shutdown.assert_called()

    @patch('pathlib.Path.mkdir')
    def test_file_manager_init_general_exception(self, mock_mkdir):
        mock_mkdir.side_effect = Exception("Unexpected failure")
        with self.assertRaises(SystemExit):
            FileManager("/invalid_path", "/invalid_meta", "flat")

    @patch("requests.Session.get")
    def test_query_handler_fetch_from_node_http_error(self, mock_get):
        mock_get.side_effect = requests.RequestException("Boom")
        handler = QueryHandler()
        with self.assertRaises(requests.RequestException):
            handler._fetch_from_node("https://fake-node", {}, 5)

    @patch("builtins.open", side_effect=IOError("File read error"))
    def test_downloader_verify_checksum_file_open_error(self, mock_open):
        file_info = {'checksum': ['123'], 'checksum_type': ['sha256']}
        result = self.downloader.verify_checksum(Path("somefile.nc"), file_info)
        self.assertFalse(result)

    def test_downloader_verify_checksum_unsupported_type(self):
        with patch("builtins.open", mock_open := MagicMock()):
            mock_open.return_value.__enter__.return_value.read.return_value = b"data"
            file_info = {'checksum': ['123'], 'checksum_type': ['sha512']}
            result = self.downloader.verify_checksum(Path("file.nc"), file_info)
            self.assertTrue(result)  # Unsupported type gets skipped

    def test_downloader_download_file_invalid_url_format(self):
        file_info = {'title': 'file.nc', 'url': ['ftp://invalid'], 'checksum': ['x'], 'checksum_type': ['sha256']}
        path, failed = self.downloader.download_file(file_info)
        self.assertIsNone(path)
        self.assertEqual(failed, file_info)

    def test_downloader_download_all_zero_files(self):
        downloaded, failed = self.downloader.download_all([])
        self.assertEqual(downloaded, [])
        self.assertEqual(failed, [])

    @patch.object(QueryHandler, 'fetch_specific_file', return_value=None)
    def test_downloader_retry_failed_missing_metadata(self, mock_fetch):
        failed = [{'title': 'test_file.nc'}]
        downloaded, still_failed = self.downloader.retry_failed(failed)
        self.assertEqual(downloaded, [])
        self.assertEqual(still_failed, failed)

    @patch.object(Downloader, 'download_file', side_effect=Exception("boom"))
    def test_downloader_download_all_exception_during_future(self, mock_dl):
        files = [{'title': 'x.nc', 'url': ['https://x.nc|HTTPServer']}]
        downloaded, failed = self.downloader.download_all(files)
        self.assertEqual(len(failed), 1)

    @patch.object(Downloader, 'verify_checksum', return_value=False)
    @patch('requests.Session.get')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('pathlib.Path.unlink')
    @patch('time.sleep', return_value=None)
    def test_downloader_download_file_checksum_retry_exceeded(self, mock_sleep, mock_unlink, mock_open, mock_get, mock_verify):
        mock_response = Mock()
        mock_response.iter_content.return_value = [b'data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        file_info = {
            'title': 'file.nc',
            'url': ['https://test.com/file.nc|HTTPServer'],
            'checksum': ['bad'], 'checksum_type': ['sha256'],
            'activity_id': ['ScenarioMIP'],
            'variable_id': ['tas'],
            'source_id': ['CanESM5'],
            'nominal_resolution': ['250 km']
        }
        path, failed = self.downloader.download_file(file_info)
        self.assertIsNone(path)
        self.assertEqual(failed, file_info)

    def setUp(self):
        self.download_dir = "/tmp/downloads"
        self.metadata_dir = "/tmp/metadata"
        self.file_manager = FileManager(self.download_dir, self.metadata_dir, "flat", prefix="test_", metadata_prefix="meta_")
        self.query_handler = QueryHandler()
        self.downloader = Downloader(
            self.file_manager, max_workers=2, retries=2, timeout=5, max_downloads=10,
            username="test_user", password="test_pass", verify_ssl=True
        )

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data='{"project": "CMIP6"}')
    def test_load_config_valid(self, mock_open):
        cfg = load_config("config.json")
        self.assertEqual(cfg['project'], "CMIP6")

    @patch("builtins.open", side_effect=FileNotFoundError("Not found"))
    def test_load_config_missing_file(self, mock_open):
        with self.assertRaises(SystemExit):
            load_config("missing.json")

    def test_parse_file_time_range_month_format(self):
        s, e = parse_file_time_range("tas_mon_model_experiment_202201-202212.nc")
        self.assertEqual(s, "2022-01-01")
        self.assertEqual(e, "2022-12-01")

    def test_parse_file_time_range_bad_format(self):
        s, e = parse_file_time_range("invalid_file.nc")
        self.assertIsNone(s)
        self.assertIsNone(e)

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data=b"dummy")
    def test_verify_checksum_md5_success(self, mock_open):
        file_info = {'checksum': ['5d41402abc4b2a76b9719d911017c592'], 'checksum_type': ['md5']}
        path = Path("dummy.txt")
        with patch("hashlib.md5") as mock_md5:
            md5obj = mock_md5.return_value
            md5obj.hexdigest.return_value = '5d41402abc4b2a76b9719d911017c592'
            md5obj.update = Mock()
            result = self.downloader.verify_checksum(path, file_info)
            self.assertTrue(result)

    def test_download_file_invalid_metadata(self):
        file_info = {"url": []}
        path, failed = self.downloader.download_file(file_info)
        self.assertIsNone(path)
        self.assertIsNotNone(failed)

    def test_retry_failed_no_files(self):
        downloaded, failed = self.downloader.retry_failed([])
        self.assertEqual(downloaded, [])
        self.assertEqual(failed, [])

    def test_query_handler_no_nodes(self):
        qh = QueryHandler(nodes=[])
        results = qh.fetch_datasets({}, timeout=2)
        self.assertEqual(results, [])

    def test_file_manager_output_path_unknown_resolution(self):
        info = {'title': 'f.nc', 'activity_id': ['A'], 'variable_id': ['tas'], 'source_id': ['X']}
        path = self.file_manager.get_output_path(info)
        self.assertIn("unknown", str(path))

    def test_file_manager_save_metadata_error(self):
        with patch("builtins.open", side_effect=IOError("fail")):
            self.file_manager.save_metadata([{"title": "f"}], "meta.json")

    def test_file_manager_hierarchical_no_resolution(self):
        fm = FileManager(self.download_dir, self.metadata_dir, "hierarchical")
        info = {'title': 'f.nc', 'activity_id': ['A'], 'variable_id': ['tas'], 'source_id': ['X']}
        out = fm.get_output_path(info)
        self.assertTrue(out.name == "f.nc")

    def setUp(self):
        self.download_dir = "/tmp/downloads"
        self.metadata_dir = "/tmp/metadata"
        self.file_manager = FileManager(self.download_dir, self.metadata_dir, "flat", prefix="test_", metadata_prefix="meta_")
        self.query_handler = QueryHandler()
        self.downloader = Downloader(
            self.file_manager, max_workers=2, retries=1, timeout=5, max_downloads=10,
            username="test_user", password="test_pass", verify_ssl=True
        )

    def test_openid_warning(self):
        with self.assertLogs(level="WARNING") as cm:
            Downloader(self.file_manager, 2, 2, 5, 5, None, None, True, openid="test_openid")
        self.assertTrue(any("OpenID provided" in msg for msg in cm.output))

    def test_shutdown_executor_cleanup(self):
        mock_executor = Mock()
        self.downloader.executor = mock_executor
        future = Mock()
        self.downloader.pending_futures = [future]
        self.downloader.shutdown()
        # pending futures should be cancelled
        future.cancel.assert_called()
        # and the executor.shutdown(wait=False) should have been called
        mock_executor.shutdown.assert_called_with(wait=False)

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data=b"corrupt")
    def test_verify_checksum_mismatch(self, mock_open):
        file_info = {"checksum": ["bad"], "checksum_type": ["sha256"]}
        with patch("hashlib.sha256") as mock_sha:
            mock_sha_obj = mock_sha.return_value
            mock_sha_obj.hexdigest.return_value = "wrong"
            mock_sha_obj.update = Mock()
            result = self.downloader.verify_checksum(Path("dummy"), file_info)
            self.assertFalse(result)

    @patch("pathlib.Path.unlink", side_effect=OSError("unlink failed"))
    def test_download_file_unlink_exception(self, mock_unlink):
        file_info = {
            'title': 'f.nc',
            'url': ['https://test.com/f.nc|HTTPServer'],
            'checksum': ['bad'], 'checksum_type': ['sha256']
        }
        with patch.object(self.downloader, "verify_checksum", return_value=False):
            with patch("pathlib.Path.exists", return_value=True):
                with self.assertLogs(level="ERROR") as cm:
                    self.downloader.download_file(file_info)
        self.assertTrue(any("Failed to remove existing file" in msg for msg in cm.output))

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_download_file_write_chunk_none(self, mock_open):
        file_info = {
            'title': 'f.nc',
            'url': ['https://test.com/f.nc|HTTPServer'],
            'checksum': ['abc'], 'checksum_type': ['sha256']
        }
        response = Mock()
        response.iter_content.return_value = [None]
        response.raise_for_status.return_value = None
        with patch("requests.Session.get", return_value=response):
            with patch.object(self.downloader, "verify_checksum", return_value=True):
                self.downloader.download_file(file_info)

    def test_download_file_checksum_fail_all_retries(self):
        file_info = {
            'title': 'f.nc',
            'url': ['https://test.com/f.nc|HTTPServer'],
            'checksum': ['x'], 'checksum_type': ['sha256']
        }
        with patch.object(self.downloader, "verify_checksum", return_value=False):
            with patch("requests.Session.get") as mget:
                mresp = Mock()
                mresp.iter_content.return_value = [b"x"]
                mresp.raise_for_status.return_value = None
                mget.return_value = mresp
                path, failed = self.downloader.download_file(file_info)
        self.assertIsNone(path)
        self.assertEqual(failed, file_info)

    def test_retry_failed_exception_during_future(self):
        file_info = {'title': 'f.nc', 'url': ['https://test.com/f.nc|HTTPServer']}
        self.downloader.executor = Mock()
        future = Mock()
        future.result.side_effect = Exception("boom")
        self.downloader.pending_futures = [future]
        self.downloader.query_handler.fetch_specific_file = lambda x, y: x
        result = self.downloader.retry_failed([file_info])
        self.assertEqual(len(result[1]), 1)

    def test_run_download_missing_retry_file(self):
        args = Mock()
        args.retry_failed = "nonexistent.json"
        args.config = None
        args.project = "CMIP6"
        args.output_dir = "/tmp/downloads"
        args.metadata_dir = "/tmp/metadata"
        args.save_mode = "flat"
        args.workers = 1
        args.retries = 1
        args.timeout = 5
        args.max_downloads = 1
        args.id = None
        args.password = None
        args.no_verify_ssl = False
        args.openid = None
        args.extra_params = None
        args.latest = False
        args.demo = False
        args.test = False
        args.variable = "tas"
        args.activity = args.experiment = args.frequency = None
        args.ensemble = args.institution = args.model = args.source_type = None
        args.grid_label = args.resolution = None
        args.dry_run = False

        path = Path(args.retry_failed)
        if path.exists():
            path.unlink()

        with self.assertRaises(SystemExit) as cm:
            run_download(args)
        self.assertEqual(cm.exception.code, 1)

    def test_run_download_dry_run(self):
        args = Mock()
        args.retry_failed = None
        args.config = None
        args.project = "CMIP6"
        args.output_dir = "/tmp/downloads"
        args.metadata_dir = "/tmp/metadata"
        args.save_mode = "flat"
        args.workers = 1
        args.retries = 1
        args.timeout = 5
        args.max_downloads = 1
        args.id = None
        args.password = None
        args.no_verify_ssl = False
        args.openid = None
        args.latest = False
        args.extra_params = None
        args.demo = False
        args.test = False
        args.variable = "tas"
        args.activity = args.experiment = args.frequency = None
        args.ensemble = args.institution = args.model = args.source_type = None
        args.grid_label = args.resolution = None
        args.dry_run = True

        with patch.object(QueryHandler, "fetch_datasets", return_value=[{"title": "f.nc"}]):
            with self.assertRaises(SystemExit) as cm:
                run_download(args)
        self.assertEqual(cm.exception.code, 0)

    def test_run_download_extra_params_invalid_json(self):
        args = Mock()
        args.retry_failed = None
        args.config = None
        args.project = "CMIP6"
        args.output_dir = "/tmp/downloads"
        args.metadata_dir = "/tmp/metadata"
        args.save_mode = "flat"
        args.workers = 1
        args.retries = 1
        args.timeout = 5
        args.max_downloads = 1
        args.id = None
        args.password = None
        args.no_verify_ssl = False
        args.openid = None
        args.latest = False
        args.extra_params = "{bad_json: true"  # malformed
        args.demo = False
        args.test = False
        args.variable = "tas"
        args.activity = args.experiment = args.frequency = None
        args.ensemble = args.institution = args.model = args.source_type = None
        args.grid_label = args.resolution = None
        args.dry_run = False

        with self.assertRaises(SystemExit) as cm:
            run_download(args)
        self.assertEqual(cm.exception.code, 1)

if __name__ == '__main__':
    unittest.main()
