# -*- coding: utf-8 -*-
"""独立G3审核：用NumPy/G2算子重算HQS各阶段，再核对模型状态。"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ENGINE));sys.path.insert(0,str(ROOT))
from main_g0_verify import verify_manifest
from optics.g2_rgb import RGBForward
from optics.g3_hqs import TorchRGB,HQSDecoder

def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()

def main(run):
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    validation=read('metrics/validation.json');assert validation['status']=='completed' and validation['structural_validation_passed']
    assert not validation['genuine_hsi_training'] and not validation['paper_alignment_passed']
    assert all(digest(ENGINE/name)==sha for name,sha in read('source_manifest.json').items())
    evidence=read('source_evidence.json');source=Path(evidence['path'])
    assert {p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file()}==evidence['sha256']
    c=read('config_effective.json');g2=json.loads((source/'config_effective.json').read_text(encoding='utf-8'))
    with np.load(source/'arrays/g2_measurement.npz',allow_pickle=False) as data:k=data['kernels'];r=data['response']
    op=RGBForward(k,r,g2['input_unit'],g2['bin_width_nm'])
    with np.load(run/'arrays/g3_diagnostics.npz',allow_pickle=False) as z:data={name:z[name] for name in z.files}
    y=data['rgb']/float(data['normalization_photons']);initial=op.adjoint(y)
    initial_error=float(np.linalg.norm(initial-data['initial_full'])/np.linalg.norm(initial));assert initial_error<2e-6
    rows=[]
    for i in range(c['network']['stages']):
        prev=data['stage_previous'][i];prior=data['stage_prior'][i];update=data['stage_updated'][i]
        assert np.array_equal(prev,data['initial_full'] if i==0 else data['stage_updated'][i-1])
        gradient=op.adjoint(op.forward(prev))-initial
        expected=prev-float(data['epsilon'][i])*(gradient+float(data['rho'][i])*(prev-prior))
        error=float(np.linalg.norm(expected-update)/max(np.linalg.norm(update),1e-30));assert error<2e-6
        rows.append(dict(stage=i,numpy_equation21_relative=error))
    assert np.array_equal(data['stage_updated'][-1],data['smoke_full'])
    np.testing.assert_allclose(op.forward(data['smoke_target']),data['smoke_rgb'],rtol=1e-13,atol=1e-13)
    torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['seed'])
    model=HQSDecoder(TorchRGB(k,r,op.spectral_factor),c['network']).float();model.eval()
    tensor=lambda a:torch.tensor(a.transpose(2,0,1)[None].copy(),dtype=torch.float32)
    with torch.no_grad():original=model(tensor(y))[0].permute(1,2,0).numpy()
    assert np.array_equal(original,data['untrained_full'])
    pristine={name:t.clone() for name,t in model.state_dict().items()}
    payload=torch.load(run/'arrays/smoke_checkpoint.pt',map_location='cpu',weights_only=True);model.load_state_dict(payload['state_dict'])
    changed=[]
    for i in range(len(model.stages)):
        prefix=f'stages.{i}.'
        names=[n for n in pristine if n.startswith(prefix)]
        changes=sum(not torch.equal(pristine[n],payload['state_dict'][n]) for n in names)
        assert changes>3;changed.append(dict(stage=i,changed_state_tensors=changes))
    assert torch.equal(model.operator.kernels,torch.tensor(k,dtype=torch.float32))
    assert torch.equal(model.operator.response,torch.tensor(r,dtype=torch.float32))
    with torch.no_grad():restored=model(tensor(y))[0].permute(1,2,0).numpy()
    assert np.array_equal(restored,data['smoke_full'])
    frozen=verify_manifest(json.loads((ROOT/'baseline/g0_20261005/manifest.json').read_text(encoding='utf-8')));assert frozen['passed']
    result=dict(passed=True,run=str(run),source_unchanged=True,source_manifest_passed=True,initial_numpy_relative=initial_error,
        independent_stage_checks=rows,checkpoint_matches_full_output=True,recreated_initialization_matches=True,
        changed_stage_parameters=changed,g0=frozen,limitations=['仅两步单合成patch梯度验算','非官方精确架构','非真实HSI训练'])
    with (Path(__file__).parent/'independent_audit.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
