# -*- coding: utf-8 -*-
"""阶段02A独立审核：只新增证据，不删除文件、不改生产源码。"""
from pathlib import Path
import copy
import hashlib
import io
import json
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'outputs' / 'jeon2019_optics'
OUT = Path(__file__).resolve().parent
RUN = PROJECT / 'results/stage02a/run_20261001_203729_02'
sys.path.insert(0, str(PROJECT))
sys.stdout.reconfigure(encoding='utf-8')
import main_stage02 as entry


def independent_n(lam):
    """独立写出文献三项色散公式，不调用待审材料模块。"""
    l2 = (np.asarray(lam) * 1e6) ** 2
    return np.sqrt(1 + sum(b*l2/(l2-c*c) for b, c in
        [(0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)]))


def main():
    evidence = {'reviewed_run': str(RUN), 'production_code_edited': False}
    cfg = json.loads((PROJECT / 'config_stage02a.json').read_text(encoding='utf-8'))
    h = np.load(RUN / 'arrays/doe_continuous.npz', allow_pickle=False)
    xp, yp = h['x_in_m'], h['y_in_m']
    xx, yy = np.meshgrid(xp, yp)
    radius = np.hypot(xx, yy)
    delta = radius**2 / (np.sqrt(radius**2 + 0.05**2)+0.05)
    theta = np.mod(np.arctan2(yy, xx), 2*np.pi)
    ld = 420e-9 + 240e-9 * np.mod(theta, 2*np.pi/3)/(2*np.pi/3)
    ld[radius == 0] = 420e-9
    mask = (radius <= 0.0005).astype(float)
    height = -np.remainder(delta, ld)/(independent_n(ld)-1)*mask
    evidence['height'] = {
        'max_abs_difference_m': float(np.max(np.abs(height-h['delta_h_m']))),
        'mask_equal': bool(np.array_equal(mask, h['mask'])),
        'lambda_design_max_error_m': float(np.max(np.abs(ld-h['lambda_design_m']))),
        'min_m': float(np.min(height)), 'max_m': float(np.max(height)),
        'fingerprint_stored': str(h['fingerprint'])}
    dx, dy = xp[1]-xp[0], yp[1]-yp[0]
    points = [(150,150), (170,161), (105,95), (250,230), (50,10), (290,290)]
    previous_axes = None
    psfs = []
    for nm in [420, 540, 660]:
        a = np.load(RUN / f'arrays/psf_{nm}nm.npz', allow_pickle=False)
        lam = float(a['wavelength_m'])
        u1 = mask*np.exp(2j*np.pi*(independent_n(lam)-1)*height/lam)
        field_points = []
        for j, i in points:
            xo, yo = a['x_out_m'][i], a['y_out_m'][j]
            # 对全部孔径点直接求和，不调用待审传播器或可分离矩阵函数。
            value = (np.exp(2j*np.pi*0.05/lam)/(1j*lam*0.05)
                * np.sum(u1*np.exp(1j*np.pi*((xo-xx)**2+(yo-yy)**2)/(lam*0.05))) * dx*dy)
            field_points.append((complex(value), complex(a['u2_complex'][j,i])))
        num = np.asarray([v[0] for v in field_points])
        saved = np.asarray([v[1] for v in field_points])
        intensity = a['intensity_raw']
        xo, yo = a['x_out_m'], a['y_out_m']
        pin = np.sum(np.abs(u1)**2)*dx*dy
        pwin = np.sum(intensity)*(xo[1]-xo[0])*(yo[1]-yo[0])
        psfs.append({'wavelength_nm': nm,
            'direct_sum_points_relative_l2': float(np.linalg.norm(num-saved)/np.linalg.norm(saved)),
            'intensity_identity_max_error': float(np.max(np.abs(intensity-np.abs(a['u2_complex'])**2))),
            'finite_nonzero': bool(np.all(np.isfinite(intensity)) and np.max(intensity)>0),
            'common_axes_equal': previous_axes is None or bool(np.array_equal(xo,previous_axes[0]) and np.array_equal(yo,previous_axes[1])),
            'fingerprint_equal': str(a['doe_fingerprint']) == str(h['fingerprint']),
            'Pin': float(pin), 'Pwindow': float(pwin), 'eta_window': float(pwin/pin),
            'n_independent': float(independent_n(lam))})
        previous_axes = (xo, yo)
    evidence['psf_independent_checks'] = psfs
    evidence['material_550nm_independent'] = float(independent_n(550e-9))

    # 复跑所有不含删除操作的测试，明确排除一个会unlink的新增测试。
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for name in ['tests.test_propagation', 'tests.test_negative_paths',
                 'tests.test_validation_completion', 'tests.test_doe', 'tests.test_materials']:
        suite.addTests(loader.loadTestsFromName(name))
    for name in ['test_entry_no_save_mode', 'test_invocation_from_outside_project']:
        suite.addTests(loader.loadTestsFromName('tests.test_stage02_entry.TestStage02Entry.'+name))
    with (OUT/'安全测试复跑.log').open('x',encoding='utf-8') as log:
        t = time.perf_counter()
        result = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    evidence['tests'] = {'run_count':result.testsRun,'success':result.wasSuccessful(),
        'elapsed_s':time.perf_counter()-t,
        'not_run':['tests.test_stage02_entry.TestStage02Entry.test_invalid_config_fails_cleanly（含自动unlink，未获用户删除授权）']}

    # 真实无保存全流程，避免现入口覆盖docs中的既有报告。
    t = time.perf_counter()
    cmd = [sys.executable,'-B',str(PROJECT/'main_stage02.py'),'--no-save']
    done = subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    with (OUT/'项目外完整无保存运行.log').open('x',encoding='utf-8') as log:
        log.write(done.stdout+'\n'+done.stderr)
    evidence['default_physics_nosave_rerun'] = {'command':cmd,'cwd':str(ROOT),
        'exit_code':done.returncode,'elapsed_s':time.perf_counter()-t}

    probes=[]
    variations=[('半宽字段与实际输入网格不符', lambda c:c['grid']['input'].update(n=11)),
        ('输出半宽字段与实际网格不符',lambda c:c['grid']['output'].update(half_width_m=0.009)),
        ('非整数翼数被截断',lambda c:c['optical'].update(wings_N=3.5)),
        ('未知材料模型',lambda c:c['optical'].update(material_model='not_a_material')),
        ('未知传播模型',lambda c:c['propagation'].update(method='not_a_method')),
        ('NaN传播距离',lambda c:c['optical'].update(distance_m=float('nan')))]
    for index,(name,change) in enumerate(variations):
        c=copy.deepcopy(cfg);change(c)
        p=OUT/f'反例配置_{index+1}.json'
        with p.open('x',encoding='utf-8') as f:json.dump(c,f,ensure_ascii=False,indent=2)
        try:entry.load_and_validate_config(p);accepted=True
        except Exception:accepted=False
        probes.append({'case':name,'invalid_config_accepted':accepted})
    evidence['config_probes']=probes
    # 故意返回全零传播场，检查验收能否发现无光的失败路径，不写生产结果。
    with patch.object(entry,'fresnel_kernel_separable',side_effect=lambda **k:np.zeros((len(k['y_out']),len(k['x_out'])),dtype=complex)):
        zero=entry.run_stage02a(copy.deepcopy(cfg),run_dir=None,only_mode='preview')
    evidence['zero_field_probe']={'completion':zero['completion_state'],'self_check':zero['self_check']}
    modified=[]
    baseline=json.loads((ROOT/'outputs/Antigravity交接_阶段02_20261001/03_交接时文件指纹与环境.json').read_text(encoding='utf-8'))
    for a in baseline['current_source_fingerprints']:
        p=Path(a['absolute_path'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=a['sha256']:modified.append(a['project_relative_path'])
    evidence['changed_since_handoff']=modified
    with (OUT/'独立审核数值.json').open('x',encoding='utf-8') as f:json.dump(evidence,f,ensure_ascii=False,indent=2)
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
