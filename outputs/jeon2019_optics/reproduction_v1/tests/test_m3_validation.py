"""独立审核的合成合同测试，不登记正式数据通过状态。"""
import sys
import shutil
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import ROOT, read_json, sha256
from m3_common import save_json, fingerprint
from m3_index import normalization_scale, build_index, BoundedReader
from m3_pipeline import write_index, final_packages
from m3_validate import validate_artifacts
import test_m3


class TestIndependentAudit(unittest.TestCase):
    def test_environment_error_is_explicit_without_aborting_data_check(self):
        import main
        broken=SimpleNamespace(check=Mock(side_effect=KeyError('remaining_bytes')))
        with patch.dict(sys.modules,{'m1_environment':broken}):result=main.environment()
        self.assertEqual(result['status'],'failed')
        self.assertIn('remaining_bytes',result['error'])
        self.assertFalse(result['gpu_execution_verified'])

    def artifacts(self):
        dest=ROOT/'work/datasets/reproduction_v1/fixtures'/('audit_'+uuid.uuid4().hex)
        dest.mkdir(parents=True)
        save_json(dest/'kind.json',{'kind':'synthetic_fixture_only','formal_scenes':0})
        records=test_m3.TestM3().fixtures()
        scales,scale_info=normalization_scale(records)
        save_json(dest/'normalization_audit.json',{'scales':scales,'datasets':scale_info})
        index=build_index(records,scales,20261006,30000)
        index_sha=write_index(dest/'patch_index.jsonl',index)
        write_index(dest/'patch_index_replay.jsonl',index)
        picked={}
        for row in index:picked.setdefault(row['scene_id'],row)
        gain={'value':.25,'physical_rms':1.,'sum_squares':100.,'count':100,'training_only':True,'selected_indices':picked}
        save_json(dest/'gain_audit.json',gain)
        scenes=[];prepared={}
        for row in records:
            digests={'base.npy':sha256(row['data_file']),'mask.npy':sha256(row['mask_file'])}
            prepared[row['scene_id']]=digests
            scenes.append({'scene_id':row['scene_id'],'scene_group_id':row['scene_group_id'],'split':'train','dataset':row['dataset'],
                           'source_sha256':{},'base_resolution':{'data_file':row['data_file'],'sha256':digests,'details':{'valid_pixels':512*512,'shape_chw':[25,512,512]}},'valid_mask':{'file':row['mask_file']}})
        for i in range(10):
            scenes.append({'scene_id':'kaist:synthetic_test_'+str(i),'scene_group_id':'synthetic_test_group_'+str(i),'split':'test','dataset':'kaist','source_sha256':{},'base_resolution':{'data_file':None},'valid_mask':{'file':None}})
        for row in scenes:row['fingerprint']=fingerprint(row)
        save_json(dest/'scene_manifest.json',scenes)
        core={'kind':'synthetic_fixture_only','preprocessed_sha256':prepared,'patch_index_sha256':index_sha}
        core['fingerprint']=fingerprint(core);save_json(dest/'data_version.json',core)
        m2=ROOT/'outputs/jeon2019_optics/results/reproduction_v1/m2/run_20261007_104321_03d47940'
        parents={method:read_json(m2/(method+'_package.json')) for method in ['jeon','fresnel']}
        operators=final_packages(parents,m2,dest,core['fingerprint'],gain['value'],dest/'gain_audit.json')
        tests=[r for r in scenes if r['split']=='test']
        seal={'scene_ids':[r['scene_id'] for r in tests],'scene_groups':[r['scene_group_id'] for r in tests],
              'source_sha256':{r['scene_id']:r['source_sha256'] for r in tests},'operator_fingerprints':{m:r['fingerprint'] for m,r in operators.items()},
              'open_milestone':'M7','preprocessed_before_M7':False,'data_version_fingerprint':core['fingerprint']}
        seal['fingerprint']=fingerprint(seal);save_json(dest/'test_seal.json',seal)
        reader=BoundedReader(records,scales)
        row=index[0];reader.read(row['scene_id'],row['scale'],row['y'],row['x'])
        save_json(dest/'cache_audit.json',reader.metrics())
        return dest,parents,operators

    def test_valid_artifacts_and_changed_physical_attachment(self):
        dest,parents,operators=self.artifacts()
        result=validate_artifacts(dest,parents,operators)
        self.assertTrue(result['passed']);self.assertGreater(result['check_count'],90000)
        # 创建错误的新附件副本以验证拒绝，不修改或删除原父包/已有附件。
        faulty=dest/'faulty_operator';faulty.mkdir()
        package=read_json(operators['jeon']['file'])
        bad=package['spectral_response']
        for path in Path(operators['jeon']['file']).parent.iterdir():
            if path.name not in ['package.json',bad]:shutil.copyfile(path,faulty/path.name)
        (faulty/bad).write_bytes(b'synthetic invalid attachment')
        package['fingerprint']=fingerprint({k:v for k,v in package.items() if k!='fingerprint'})
        save_json(faulty/'package.json',package)
        rejected=dict(operators,jeon={'file':str(faulty/'package.json'),'fingerprint':package['fingerprint']})
        with self.assertRaisesRegex(ValueError,'物理附件'):
            validate_artifacts(dest,parents,rejected)


if __name__=='__main__':unittest.main()
