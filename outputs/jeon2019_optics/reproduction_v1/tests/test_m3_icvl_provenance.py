"""固定ICVL交接目录的摘要和版本核对，不访问云盘或账号。"""
import copy
import sys
import unittest
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import ROOT,read_json
from m3_common import save_json
from m3_icvl_provenance import load_pinned_inventory


class TestPinnedICVL(unittest.TestCase):
    def provenance(self):
        base=ROOT/'work/datasets/reproduction_v1/icvl_acquisition'
        return {'source':base/'icvl_source.json','manifest':base/'icvl_file_manifest.json','tree':base/'remote_tree_pinned.json'}

    def modified(self,kind,body):
        folder=ROOT/'work/datasets/reproduction_v1/fixtures'/('icvl_provenance_'+uuid.uuid4().hex)
        folder.mkdir(parents=True)
        save_json(folder/'kind.json',{'kind':'synthetic_negative_fixture_only','formal_scenes':0})
        save_json(folder/(kind+'.json'),body)
        return folder/(kind+'.json')

    def test_pinned_official_inventory(self):
        records=load_pinned_inventory(self.provenance())
        self.assertEqual(len(records),202)
        self.assertEqual(sum(r['bytes'] for r in records.values()),28199177384)

    def test_redacted_oid_is_rejected(self):
        p=self.provenance();tree=copy.deepcopy(read_json(p['tree']))
        row=next(r for r in tree if r['path'].startswith('mat/') and r['type']=='file')
        row['lfs']['oid']='*'*64
        p['tree']=self.modified('redacted_tree',tree)
        with self.assertRaisesRegex(ValueError,'遮蔽'):load_pinned_inventory(p)

    def test_revision_mismatch_is_rejected(self):
        p=self.provenance();manifest=read_json(p['manifest']);manifest['revision']='0'*40
        p['manifest']=self.modified('wrong_revision',manifest)
        with self.assertRaisesRegex(ValueError,'commit'):load_pinned_inventory(p)


if __name__=='__main__':unittest.main()
