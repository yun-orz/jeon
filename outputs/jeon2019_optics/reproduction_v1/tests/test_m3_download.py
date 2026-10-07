"""离线下载断点核对；所有合成文件保留，不启动外网或删除文件。"""
import hashlib
import sys
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import ROOT
from m3_common import save_json
from m3_download import download


class TestDownload(unittest.TestCase):
    def fixture(self, tail=b'', offset=0):
        folder=ROOT/'work/datasets/reproduction_v1/fixtures'/('download_'+uuid.uuid4().hex)
        folder.mkdir(parents=True)
        content=b'only synthetic fixture'
        (folder/'payload.bin').write_bytes(content+tail)
        row={'url':'https://synthetic.invalid/file','file':'payload.bin','expected_bytes':len(content),'source_identity':{'etag':'fixture','last_modified':None},'segments':[{'offset':offset,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}],'completed_bytes':len(content),'status':'partial'}
        save_json(folder/'progress_00000000.json',row)
        return folder,content,row

    def test_final_segment_resume_without_network(self):
        folder,content,row=self.fixture(tail=b'uncommitted tail')
        original=(folder/'payload.bin').read_bytes()
        with patch('urllib.request.urlopen',side_effect=AssertionError('不应访问网络')):
            result=download(row['url'],folder,len(content),time.monotonic()+10,lambda:None,hashlib.sha256(content).hexdigest())
        self.assertEqual(result['status'],'complete')
        self.assertEqual(Path(result['file']).read_bytes(),content)
        self.assertEqual((folder/'payload.bin').read_bytes(),original)

    def test_wrong_source_sha_rejected(self):
        folder,content,row=self.fixture()
        with self.assertRaisesRegex(ValueError,'SHA'):
            download(row['url'],folder,len(content),time.monotonic()+10,lambda:None,'0'*64)

    def test_noncontiguous_journal_rejected(self):
        folder,content,row=self.fixture(offset=1)
        with self.assertRaisesRegex(ValueError,'不连续'):
            download(row['url'],folder,len(content),time.monotonic()+10,lambda:None)


if __name__=='__main__':unittest.main()
