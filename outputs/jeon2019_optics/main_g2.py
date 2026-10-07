# -*- coding: utf-8 -*-
"""PyCharm无参数入口：25波段到RGB的基础强度成像验证。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from main_g1 import write_json
from main_stage02b import allocate_unique_run_dir
from optics.g2_rgb import RGBForward,synthetic_response,demo_scene,operator_diagnostics
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,effective_runtime,configure_plotting
from optics.runutil import environment_info

ROOT=Path(__file__).resolve().parent


def load_source(c):
    """对照上轮审核保存的原始SHA清单，拒绝损坏或未经批准的来源。"""
    if c['source_kind']!='g1_ccw_frozen':raise ValueError('本批只支持已审核原G1；不自动改用CW诊断变体')
    source=resolve_project_path(c['source_run'],ROOT)
    evidence=json.loads(resolve_project_path(c['source_evidence'],ROOT).read_text(encoding='utf-8'))
    before=tree_sha(source)
    # 来源记录中的绝对路径只作历史出处；用内容清单核验，允许整项目搬迁。
    if before!=evidence['sha256']:raise ValueError('G1来源与上轮审核清单不一致')
    status=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    if status['status']!='completed' or status['numerical_validation_passed'] is not True or (source/'failed.json').exists():raise ValueError('源G1未完成数值审核')
    codes=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    for name,sha in codes.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('G1源代码指纹改变')
    with np.load(source/'arrays/psf_bank.npz',allow_pickle=False) as data:
        bank=data['kernels'];waves=data['wavelengths_m'];pitch=float(data['pitch_m']);fingerprint=str(data['fingerprint'])
        x=data['pixel_x_m'];y=data['pixel_y_m']
    if bank.shape!=(25,97,97) or not np.array_equal(waves,np.array([n/1e9 for n in range(420,661,10)])):raise ValueError('本批核库或波长不匹配')
    expected=(np.arange(97)-48)*6.22e-6
    if pitch!=6.22e-6 or not np.array_equal(x,expected) or not np.array_equal(y,expected):raise ValueError('源物理像元坐标不匹配')
    with np.load(source/'arrays/height_fixed.npz',allow_pickle=False) as data:
        if str(data['fingerprint'])!=fingerprint:raise ValueError('高度与核库器件指纹不同')
    return source,before,codes,bank,waves,pitch,fingerprint


def validate_config(c):
    for k in ['height','width']:
        if type(c['scene'][k]) is not int or not 8<=c['scene'][k]<=512:raise ValueError('本批场景高宽须为8到512的整数')
    for value in [c['scene']['amplitude_photons_per_bin'],c['bin_width_nm'],*c['thresholds'].values()]:
        if isinstance(value,bool) or not np.isfinite(value) or value<=0:raise ValueError('幅值、带宽与阈值须有限正值')
    if c['bin_width_nm']!=10.:raise ValueError('本批固定10nm带宽')
    if type(c['seed']) is not int or c['seed']<0:raise ValueError('随机种子须为非负整数')
    if type(c['plots']['dpi']) is not int or c['plots']['dpi']<50:raise ValueError('图片dpi须为至少50的整数')
    if any(type(c['runtime'][k]) is not bool for k in ['save_results','show_plots']):raise ValueError('运行开关须为布尔值')
    if c['input_unit'] not in ['photons_per_bin','photons_per_nm']:raise ValueError('输入单位不支持')


def run(c,no_save=False,show_plots=False):
    validate_config(c);runtime=effective_runtime(c,no_save,show_plots)
    source,before,codes,bank,waves,pitch,fingerprint=load_source(c)
    response=synthetic_response(waves,c['response'])
    operator=RGBForward(bank,response,c['input_unit'],c['bin_width_nm'])
    plt,backend=configure_plotting(runtime['show_plots']);figures=[]
    dest=allocate_unique_run_dir('g2') if runtime['save_results'] else None
    start=time.perf_counter()
    try:
        cube=demo_scene(c['scene']['height'],c['scene']['width'],waves,c['scene']['amplitude_photons_per_bin'])
        if c['input_unit']=='photons_per_nm':cube/=c['bin_width_nm']
        operator.check_cube(cube,physical=True);blurred=operator.convolve_bands(cube);rgb=operator.forward(cube)
        backprojection=operator.adjoint(rgb)
        checks=operator_diagnostics(operator,c['seed']);ledger=operator.flux_ledger(cube,rgb)
        # 非负输入的FFT舍入可能产生极小负数；保留线性原始数组，不裁零。
        minimum=float(rgb.min());scale=max(float(rgb.max()),1.)
        passed=(max(v['relative_error'] for v in checks['adjoint'])<=c['thresholds']['adjoint_relative']
                and checks['center_point_relative_error']<=c['thresholds']['conservation_relative']
                and checks['unit_conversion_max_error']<1e-10 and checks['zero_input_max']==0.
                and np.all(np.isfinite(rgb)) and minimum>=-1e-12*scale
                and min(ledger['crop_loss_electrons'])>=-1e-10*max(ledger['full_convolution_electrons'])
                and min(checks['edge_point']['crop_loss_electrons'])>0)
        fig,ax=plt.subplots(figsize=(7,4))
        for k,label in enumerate(['R','G','B']):ax.plot(waves*1e9,response[k],'.-',label=label)
        ax.set_xlabel('波长（nm）');ax.set_ylabel('合成相对QE（电子/光子）');ax.set_title('测试用响应；非Canon或作者标定');ax.legend();ax.grid(alpha=.3);fig.tight_layout()
        figures.append(('synthetic_response.png',fig))
        fig,axes=plt.subplots(2,3,figsize=(12,8))
        bound=c['scene']['amplitude_photons_per_bin']
        for b,ax in zip([4,12,19],axes[0]):
            ax.imshow(cube[:,:,b]*operator.spectral_factor,origin='lower',cmap='inferno',vmin=0,vmax=bound)
            ax.set_title(f'{round(waves[b]*1e9)}nm 输入（显示上限固定）')
        upper=float(rgb.max())
        for k,ax in enumerate(axes[1]):
            ax.imshow(rgb[:,:,k],origin='lower',cmap='inferno',vmin=0,vmax=upper)
            ax.set_title(['R','G','B'][k]+' 线性观测：电子数（共同显示尺度）')
        for ax in axes.flat:ax.set_xlabel('列（像元）');ax.set_ylabel('行（像元）')
        fig.tight_layout();figures.append(('scene_rgb_planes.png',fig))
        fig,ax=plt.subplots(figsize=(6,6));ax.imshow(np.clip(rgb/upper,0,1),origin='lower')
        ax.set_title('RGB示意：共同线性缩放；非sRGB标定');fig.tight_layout();figures.append(('rgb_linear_preview.png',fig))
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime))
            np.savez_compressed(dest/'arrays/g2_measurement.npz',cube=cube,blurred_bands=blurred,rgb=rgb,
                backprojection=backprojection,kernels=bank,response=response,wavelengths_m=waves,pitch_m=pitch,
                fingerprint=fingerprint,input_unit=c['input_unit'],bin_width_nm=c['bin_width_nm'],
                response_status='synthetic_not_calibrated',output_unit='expected_electrons')
            with (dest/'metrics/response.csv').open('x',encoding='utf-8-sig',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['wavelength_nm','R_synthetic_QE','G_synthetic_QE','B_synthetic_QE'])
                writer.writerows([round(lam*1e9),*response[:,b]] for b,lam in enumerate(waves))
            write_json(dest/'metrics/operator_checks.json',checks);write_json(dest/'metrics/flux_ledger.json',ledger)
            for name,fig in figures:fig.savefig(dest/'figures'/name,dpi=c['plots']['dpi'])
            with np.load(dest/'arrays/g2_measurement.npz',allow_pickle=False) as data:
                expected=dict(cube=cube,blurred_bands=blurred,rgb=rgb,backprojection=backprojection,kernels=bank,
                    response=response,wavelengths_m=waves,pitch_m=pitch,fingerprint=fingerprint,input_unit=c['input_unit'],
                    bin_width_nm=c['bin_width_nm'],response_status='synthetic_not_calibrated',output_unit='expected_electrons')
                if any(not np.array_equal(data[k],v) for k,v in expected.items()):raise RuntimeError('G2保存数据与内存不同')
            files=sorted(set(['main_g2.py','config_g2.json','optics/g2_rgb.py',*codes]))
            write_json(dest/'source_manifest.json',{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files})
            write_json(dest/'source_evidence.json',dict(path=str(source),sha256=before))
            with (dest/'report_g2.md').open('x',encoding='utf-8') as stream:
                stream.write('# G2 RGB基础成像验证\n\nJeon公式实现假设下的光学编码＋RGB基础验证。响应为自定义合成QE，不是作者或Canon标定。未重建、未训练网络。\n\n输入为波段光子数或每nm光子谱密度，输出期望电子数；无噪声、CFA、去马赛克、gamma、白平衡或曝光增益。零延拓线性卷积中心裁剪，核保留有限窗效率。Φᵀ为伴随，不是重建结果。\n\n完整说明见G2_README.md。\n')
            required=['arrays/g2_measurement.npz','metrics/response.csv','metrics/operator_checks.json','metrics/flux_ledger.json',
                      'source_manifest.json','source_evidence.json','report_g2.md','config_effective.json',*['figures/'+n for n,_ in figures]]
            if any(not (dest/n).is_file() or (dest/n).stat().st_size==0 for n in required):raise RuntimeError('G2必需结果缺失')
        if tree_sha(source)!=before:raise RuntimeError('旧G1来源在运行期间变化')
        if runtime['show_plots']:plt.show()
        report=dict(status='completed' if passed else 'diagnostic_checks_failed',numerical_validation_passed=bool(passed),
                    paper_alignment_passed=False,real_response_verified=False,response_status='synthetic_not_calibrated',
                    input_shape=list(cube.shape),output_shape=list(rgb.shape),source_kind=c['source_kind'],source_unchanged=True,
                    input_unit=c['input_unit'],output_unit='expected_electrons',height_fingerprint=fingerprint,
                    rgb_min_raw=minimum,environment=environment_info(),runtime=runtime,backend=backend,elapsed_s=time.perf_counter()-start)
        if dest:write_json(dest/'metrics/validation.json',report)
        print('G2结果：'+str(dest) if dest else 'G2无保存：未写结果文件')
        return report
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for _,fig in figures:plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='config_g2.json')
    parser.add_argument('--no-save',action='store_true');parser.add_argument('--show-plots',action='store_true');a=parser.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    return 0 if run(c,a.no_save,a.show_plots)['numerical_validation_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
