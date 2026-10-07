# -*- coding: utf-8 -*-
"""G4独立审核：原始MAT重算校正/尺度、观测及验证模型。"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.io import loadmat
import torch
ROOT=Path(__file__).resolve().parents[2];ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ENGINE));sys.path.insert(0,str(ROOT))
from main_g0_verify import verify_manifest
from optics.g2_rgb import RGBForward
from optics.g3_hqs import TorchRGB,HQSDecoder

def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def main(run):
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    validation=read('metrics/validation.json');assert validation['status']=='completed' and validation['small_real_hsi_training_passed']
    assert not validation['paper_alignment_passed'] and not validation['real_rgb_response_verified']
    c=read('config_effective.json');meta=read('dataset_manifest.json');metrics=read('metrics/training.json')
    assert all(sha(ENGINE/name)==value for name,value in read('source_manifest.json').items())
    evidence=read('source_evidence.json')
    for record in evidence.values():
        source=Path(record['path'])
        assert {p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}==record['sha256']
    folder=(ENGINE/c['dataset']['directory']).resolve();sensitivity=np.loadtxt(folder/'calib.txt').reshape(-1)
    with np.load(run/'arrays/g4_subset.npz',allow_pickle=False) as z:saved={name:z[name] for name in z.files}
    size=meta['context_size'];waves=np.arange(420,661,10);original={}
    for split in ['train','validation']:
        record=meta[split];path=folder/record['scene'];assert sha(path)==record['sha256']
        raw=loadmat(path,variable_names=['ref','lbl']);patches=[]
        for y,x in record['origins_yx']:
            assert np.all(raw['lbl'][y:y+size,x:x+size]!=0)
            piece=raw['ref'][y:y+size,x:x+size,:25]
            patches.append(piece/sensitivity[:25]*waves/540.)
        original[split]=np.stack(patches);del raw
    expected_scale=float(np.percentile(original['train'],c['dataset']['train_percentile']))
    assert expected_scale==meta['normalization_scale']
    for split in original:assert np.array_equal((original[split]/expected_scale).astype(np.float32),saved[split+'_targets'])
    assert meta['train']['scene']!=meta['validation']['scene'] and meta['train']['sha256']!=meta['validation']['sha256']
    with np.load(ENGINE/c['source_g2']/'arrays/g2_measurement.npz',allow_pickle=False) as data:k=data['kernels'];response=data['response']
    op=RGBForward(k,response);errors=[]
    for split in ['train','validation']:
        for target,measurement in zip(saved[split+'_targets'],saved[split+'_measurements']):
            expected=op.forward(target).astype(np.float32)
            assert np.array_equal(expected,measurement)
            errors.append(float(np.max(np.abs(expected-measurement))))
    torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['seed'])
    model=HQSDecoder(TorchRGB(k,response),c['network']).float();model.eval()
    tensor=lambda a:torch.tensor(a.transpose(2,0,1)[None].copy())
    with torch.no_grad():baseline=np.stack([model(tensor(y))[0].permute(1,2,0).numpy() for y in saved['validation_measurements']])
    assert np.array_equal(baseline,saved['baseline_validation_predictions'])
    checkpoint=torch.load(run/'arrays/best_checkpoint.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():prediction=np.stack([model(tensor(y))[0].permute(1,2,0).numpy() for y in saved['validation_measurements']])
    assert np.array_equal(prediction,saved['best_validation_predictions'])
    h=meta['halo'];core=meta['core_size'];losses=[]
    for p,t in zip(prediction,saved['validation_targets']):losses.append(float(np.mean(np.abs(p[h:h+core,h:h+core]-t[h:h+core,h:h+core]))))
    loss=float(np.mean(losses));assert abs(loss-metrics['best_validation_l1'])<2e-8
    selected=min(metrics['history'],key=lambda row:row['validation_l1'])
    assert checkpoint['best_epoch']==selected['epoch']==metrics['best_epoch']
    assert torch.equal(model.operator.kernels,torch.tensor(k,dtype=torch.float32))
    assert torch.equal(model.operator.response,torch.tensor(response,dtype=torch.float32))
    frozen=verify_manifest(json.loads((ROOT/'baseline/g0_20261005/manifest.json').read_text(encoding='utf-8')));assert frozen['passed']
    result=dict(passed=True,run=str(run),original_mat_preprocessing_matches=True,train_only_normalization_matches=True,
        disjoint_scene_files=True,all_context_masks_valid=True,reencoded_measurements_max_error=max(errors),checkpoint_predictions_exact=True,
        reconstructed_validation_l1=loss,best_epoch=metrics['best_epoch'],optical_kernels_unchanged=True,source_and_dataset_unchanged=True,g0=frozen,
        limitations=['仅一幅训练场景和一幅验证场景','无独立测试集','相对单位且RGB响应为合成设置'])
    with (Path(__file__).parent/'independent_audit.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
