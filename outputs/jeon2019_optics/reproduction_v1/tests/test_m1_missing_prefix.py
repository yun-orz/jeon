"""M1缺失下载前缀的回归检查；仅使用本地小型样例，全部fixture保留。"""
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import m1_environment
import m1_download
from m1_fixture_server import start_fixture
from m1_resume import ResumeError,sha256_file


class TestMissingPrefix(unittest.TestCase):
    def setUp(self):
        self.folder=HERE.parents[2]/'work/datasets/reproduction_v1/fixtures'/('m1_missing_'+uuid.uuid4().hex)
        self.folder.mkdir(parents=True)

    def test_check_installed_environment_without_download_prefix(self):
        gpu={'exists':True,'cuda_available':True,'torch_file':str(m1_environment.ENVIRONMENT/'lib/site-packages/torch/__init__.py')}
        cpu={'exists':True,'base_prefix':str(m1_environment.BASE_PYTHON.parent)}
        with patch.object(m1_environment,'PREFIX',self.folder/'missing.whl'),patch.object(m1_environment,'disk_free',return_value=100*1024**3),patch.object(m1_environment,'interpreter_report',side_effect=[cpu,gpu]),patch.object(m1_environment,'gpu_hardware',return_value={'available':True}):
            report=m1_environment.check(persist=False)
        self.assertEqual(report['prefix']['remaining_bytes'],m1_environment.TORCH_BYTES)
        self.assertEqual(report['status'],'gpu_ready')
        self.assertTrue(report['gpu_environment_ready'])

    def download_fixture(self,prefix_exists=False,wrong_sha=False):
        payload=bytes(range(256))*256
        fixture,server,url=start_fixture(payload)
        prefix=self.folder/'old.whl'
        if prefix_exists:prefix.write_bytes(payload[:8192])
        try:
            with patch.object(m1_download,'PREFIX',prefix),patch.object(m1_download,'WORKDIR',self.folder/'new_download'),patch.object(m1_download,'TORCH_URL',url),patch.object(m1_download,'TORCH_BYTES',len(payload)),patch.object(m1_download,'TORCH_SHA256','0'*64 if wrong_sha else fixture.sha256()):
                engine=m1_download.resumer(block_bytes=8192,attempts=1)
                if wrong_sha:
                    with self.assertRaises(ResumeError):engine.run()
                    return
                manifest=engine.run()
                self.assertEqual(sha256_file(engine.destination),fixture.sha256())
                self.assertEqual(manifest['prefix_bytes'],8192 if prefix_exists else 0)
                if prefix_exists:self.assertEqual(prefix.read_bytes(),payload[:8192])
                else:self.assertFalse(prefix.exists())
                get_requests=[r for r in fixture.requests if r['method']=='GET']
                self.assertEqual(get_requests[0]['range'],'bytes=%d-%d'%((8192 if prefix_exists else 0),(16383 if prefix_exists else 8191)))
        finally:
            server.shutdown();server.server_close()

    def test_installed_cuda_without_device_does_not_recommend_reinstall(self):
        gpu={'exists':True,'cuda_available':False,'cuda_runtime':'12.1','torch_file':str(m1_environment.ENVIRONMENT/'lib/site-packages/torch/__init__.py')}
        cpu={'exists':True,'base_prefix':str(m1_environment.BASE_PYTHON.parent)}
        with patch.object(m1_environment,'PREFIX',self.folder/'missing.whl'),patch.object(m1_environment,'disk_free',return_value=100*1024**3),patch.object(m1_environment,'interpreter_report',side_effect=[cpu,gpu]),patch.object(m1_environment,'gpu_hardware',return_value={'available':False}):
            report=m1_environment.check(persist=False)
        self.assertEqual(report['status'],'gpu_device_unavailable')
        self.assertFalse(report['gpu_environment_ready'])
        self.assertNotIn('m1_download.py all',report['next_action'])

    def test_missing_prefix_and_directory_download_from_zero(self):
        self.download_fixture()

    def test_existing_prefix_still_resumes_without_changing_original(self):
        self.download_fixture(prefix_exists=True)

    def test_from_zero_does_not_bypass_whole_sha(self):
        self.download_fixture(wrong_sha=True)


if __name__=='__main__':unittest.main()
