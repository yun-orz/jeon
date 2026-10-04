# -*- coding: utf-8 -*-
"""02F-1合成电子数测量及统计验证，CPU/PyCharm直接运行。"""
import argparse,csv,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from optics.electron_noise import electron_response,sanitize_poisson_mean,sample_electrons,noise_statistics
from optics.stage02e4_source import load_noise_source
from optics.imaging import SpectralImager,direct_shift_sum
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,configure_plotting,effective_runtime
from optics.runutil import environment_info
from main_stage02b import allocate_unique_run_dir
ROOT=Path(__file__).resolve().parent

def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def validate_config(c):
    for key,allowed in [('devices',['continuous','nearest_depth']),('scenes',['coincident_points','separated_points','lines_and_square'])]:
        if not c[key] or len(set(c[key]))!=len(c[key]) or any(v not in allowed for v in c[key]):raise ValueError('实验身份不合法')
    d=c['detector'];g=d['gains_e_per_relative_power']
    if not g or len(set(g))!=len(g) or any(isinstance(v,bool) or not np.isfinite(v) or v<=0 for v in g):raise ValueError('增益须互异正有限数')
    for key in ['read_sigma_e','background_e']:
        if isinstance(d[key],bool) or not np.isfinite(d[key]) or d[key]<0:raise ValueError(key+'须非负有限')
    for key in ['budget_seconds','roundoff_relative','direct_relative_tolerance']:
        if isinstance(c[key],bool) or not np.isfinite(c[key]) or c[key]<=0:raise ValueError(key+'须正有限')
    s=c['statistics']
    for key,minimum in [('repeats',2),('block_size',1)]:
        if isinstance(s[key],bool) or not isinstance(s[key],int) or s[key]<minimum:raise ValueError('统计次数非法')
    if isinstance(s['z_tolerance'],bool) or not np.isfinite(s['z_tolerance']) or s['z_tolerance']<=0:raise ValueError('Z门槛须正有限')
    if isinstance(c['seed'],bool) or not isinstance(c['seed'],int) or c['seed']<0:raise ValueError('种子须非负整数')
    if isinstance(c['plots']['dpi'],bool) or not isinstance(c['plots']['dpi'],int) or c['plots']['dpi']<=0:raise ValueError('dpi须正整数')

def run(c,no_save=False,show_plots=False):
    validate_config(c);runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots'])
    dest=allocate_unique_run_dir('stage02f1') if runtime['save_results'] else None
    start=time.perf_counter();records=[];saved={};figs=[];budget=False
    try:
        items,evidence=load_noise_source(resolve_project_path(c['source_run'],ROOT),ROOT)
        d=c['detector'];s=c['statistics'];planned=len(c['devices'])*len(c['scenes'])*len(d['gains_e_per_relative_power'])
        controls={}
        for name,mu,sigma in [('zero',np.zeros((32,32)),0.),('read_only',np.zeros((32,32)),d['read_sigma_e']),('shot_only',np.full((32,32),17.),0.)]:
            controls[name]=noise_statistics(mu,sigma,c['seed'],'control_'+name,**dict(repeats=s['repeats'],block_size=s['block_size'],z_tolerance=s['z_tolerance']))
        if dest:write_json(dest/'config_effective.json',dict(c,runtime=runtime));write_json(dest/'source_evidence.json',evidence);write_json(dest/'metrics/controls.json',controls)
        for item in items:
            if item['device'] not in c['devices'] or item['scene'] not in c['scenes']:continue
            v=item['values'];op=SpectralImager(v['kernels'],list(v['truth'].shape[1:]),float(v['pitch_m']),v['response'],v['crop_indices'].tolist())
            bands=op.per_band_full(v['truth'])[:,op.slices[0],op.slices[1]]
            reference=[]
            for b in range(len(v['response'])):
                single=np.zeros_like(v['truth']);single[b]=v['truth'][b]
                reference.append(direct_shift_sum(single,v['kernels'],v['response'])[op.slices])
            reference=np.asarray(reference);error=float(np.linalg.norm(bands-reference)/max(np.linalg.norm(reference),1e-300))
            if error>c['direct_relative_tolerance'] or not np.allclose(bands.sum(axis=0),v['measurement'],rtol=1e-13,atol=1e-15):raise ValueError('波段FFT与逐点参考/来源测量不一致')
            response,photon_energy=electron_response(v['wavelengths_m'],d['quantum_efficiency'],d['reference_wavelength_m'])
            for gain in d['gains_e_per_relative_power']:
                if time.perf_counter()-start>c['budget_seconds']:budget=True;break
                identity=item['device']+'_'+item['scene']+'_G_'+format(gain,'.17g')
                signal=gain*np.einsum('b,byx->yx',response,bands);raw=signal+d['background_e']
                mean,repair=sanitize_poisson_mean(raw,c['roundoff_relative']);sample=sample_electrons(mean,d['read_sigma_e'],c['seed'],identity)
                stats=noise_statistics(mean,d['read_sigma_e'],c['seed'],identity,s['repeats'],s['block_size'],s['z_tolerance'])
                values={k:v[k] for k in ['truth','kernels','response','wavelengths_m','pitch_m','fixed_height_fingerprint','crop_indices','output_x_m','output_y_m','scene_x_m','scene_y_m']}
                values.update(per_band_relative_power=bands,mean_signal_raw_e=signal,mean_raw_e=raw,mean_sampling_e=mean,theoretical_variance_e2=mean+d['read_sigma_e']**2,
                    electron_response=response,quantum_efficiency=np.asarray(d['quantum_efficiency']),photon_energy_J=photon_energy,gain_e_per_relative_power=np.asarray(gain),
                    read_sigma_e=np.asarray(d['read_sigma_e']),background_e=np.asarray(d['background_e']),reference_wavelength_m=np.asarray(d['reference_wavelength_m']),seed=np.asarray(c['seed']),identity=np.asarray(identity),measurement_unit=np.asarray('electron'),**sample)
                rec=dict(identity=identity,device=item['device'],scene=item['scene'],gain_e_per_relative_power=gain,direct_relative_error=error,roundoff_repair=repair,statistics=stats,
                    mean_total_e=float(mean.sum()),negative_readout_pixels=int(np.count_nonzero(sample['noisy_measurement_e']<0)))
                records.append(rec);saved[identity]=values
                if dest:
                    np.savez_compressed(dest/'arrays'/(identity+'.npz'),**values);write_json(dest/'metrics'/(identity+'_statistics.json'),stats)
                    write_json(dest/'progress.json',dict(status='running',completed_jobs=len(records),planned_jobs=planned))
                print(identity+': 统计通过='+str(stats['passed'])+', max|Z|=%.4f'%max(abs(a['z']) for a in stats['checks'].values()),flush=True)
                fig,axes=plt.subplots(1,3,figsize=(12,3.7));figs.append(fig)
                extent=[v['output_x_m'][0]*1e6,v['output_x_m'][-1]*1e6,v['output_y_m'][0]*1e6,v['output_y_m'][-1]*1e6]
                vmax=max(float(mean.max()),float(sample['noisy_measurement_e'].max()));vmin=min(0.,float(sample['noisy_measurement_e'].min()))
                for ax,a,title in zip(axes[:2],[mean,sample['noisy_measurement_e']],['期望电子数','一次含噪测量']):
                    im=ax.imshow(a,origin='lower',extent=extent,vmin=vmin,vmax=vmax);ax.set_title(title);fig.colorbar(im,ax=ax,label='电子数')
                residual=sample['noisy_measurement_e']-mean;limit=max(float(abs(residual).max()),1e-30)
                im=axes[2].imshow(residual,origin='lower',extent=extent,cmap='coolwarm',vmin=-limit,vmax=limit);axes[2].set_title('测量－期望');fig.colorbar(im,ax=axes[2],label='电子数')
                for ax in axes:ax.set_xlabel('x / μm');ax.set_ylabel('y / μm')
                fig.suptitle(identity);fig.tight_layout()
                if dest:fig.savefig(dest/'figures'/(identity+'.png'),dpi=c['plots']['dpi'])
            if budget:break
        for source in evidence['sources']:
            if tree_sha(Path(source['path']))!=source['sha256']:raise RuntimeError('旧来源目录被修改')
        if dest:
            for identity,values in saved.items():
                with np.load(dest/'arrays'/(identity+'.npz'),allow_pickle=False) as z:
                    if set(z.files)!=set(values) or any(not np.array_equal(z[k],a) for k,a in values.items()):raise RuntimeError('落盘数组不一致')
                if not (dest/'figures'/(identity+'.png')).is_file():raise RuntimeError('结果图缺失')
                rec=next(r for r in records if r['identity']==identity)
                if json.loads((dest/'metrics'/(identity+'_statistics.json')).read_text(encoding='utf-8'))!=rec['statistics']:raise RuntimeError('落盘统计不一致')
            with (dest/'metrics/statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.writer(f);w.writerow(['identity','gain','check','z','passed'])
                for r in records:
                    for key,value in r['statistics']['checks'].items():w.writerow([r['identity'],r['gain_e_per_relative_power'],key,value['z'],value['passed']])
            paths=['main_stage02f1.py','config_stage02f1.json','optics/electron_noise.py','optics/stage02e4_source.py','optics/imaging.py','optics/stage02_runtime.py']
            write_json(dest/'source_manifest.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
        if runtime['show_plots']:plt.show()
        passed=len(records)==planned and all(r['statistics']['passed'] for r in records) and all(r['passed'] for r in controls.values())
        report=dict(status='completed' if passed else 'incomplete',validation_passed=bool(passed),budget_exhausted=budget,planned_jobs=planned,completed_jobs=len(records),records=records,controls=controls,
            environment=environment_info(),elapsed_s=time.perf_counter()-start,config=c,runtime=runtime,backend=backend,source_unchanged=True,
            scope='Jeon光学编码复现＋合成电子数测量验证；没有噪声重建',
            assumptions=['G、QE、读出标准差、背景均为预先设定的合成假设，不是论文相机标定。','固定DOE与有限窗口核保持通光效率；不重新归一化核，不再乘像元面积。','相对波段积分功率经QE与lambda因子换算电子数；无ADC/满阱/暗电流漂移。','读出后负值保留；仅泊松均值的FFT微负舍入记录并修正。','论文旋转方向对应、真实GUI/PyCharm手动点击尚未验证。'])
        if dest:write_json(dest/'metrics/validation.json',report)
        print('结果目录：'+str(dest) if dest else '无保存模式：未写结果文件',flush=True)
        return report
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for fig in figs:plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='config_stage02f1.json');parser.add_argument('--no-save',action='store_true');parser.add_argument('--show-plots',action='store_true');args=parser.parse_args()
    c=json.loads(resolve_project_path(args.config,ROOT).read_text(encoding='utf-8'));return 0 if run(c,args.no_save,args.show_plots)['validation_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
