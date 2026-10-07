# -*- coding: utf-8 -*-
"""阶段02本地运行与验收工具；不改变材料、高度和传播公式。"""
from pathlib import Path
import importlib
import math
import numpy as np


def resolve_project_path(value, project_root):
    """相对路径统一按项目根解释。"""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(project_root) / path).resolve()


def prepare_run_dir(path):
    """先拒绝非空目标，再创建目录；绝不清理或复用旧run。"""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"结果目录不是空目录，拒绝覆盖：{path}")
    path.mkdir(parents=True, exist_ok=True)
    for name in ('arrays', 'figures', 'metrics'):
        (path / name).mkdir(exist_ok=False)
    return path


def assert_fresh_run_dir(path):
    """直接计算函数只接受初始化后的空run，允许本轮已建的空日志。"""
    path = Path(path)
    for item in path.iterdir():
        if item.name in ('arrays', 'figures', 'metrics') and item.is_dir() and not any(item.iterdir()):
            continue
        if item.name == 'run.log' and item.is_file() and item.stat().st_size == 0:
            continue
        raise ValueError(f"计算目录含既有产物，拒绝覆盖：{item}")


def effective_runtime(config, no_save=False, show_plots=False):
    """CLI只覆盖对应开关，显示和保存相互独立。"""
    runtime = dict(config['runtime'])
    runtime['save_results'] = runtime['save_results'] and not no_save
    runtime['show_plots'] = runtime['show_plots'] or show_plots
    return runtime


def _probe_tk():
    """验证Tk依赖与窗口系统；探测窗口保持隐藏。"""
    import tkinter
    root = tkinter.Tk()
    try:
        root.withdraw()
        root.update_idletasks()
    finally:
        root.destroy()
    importlib.import_module('matplotlib.backends.backend_tkagg')


def configure_plotting(show):
    """在pyplot导入前选TkAgg或Agg，并如实返回显示能力。"""
    import matplotlib
    backend, available, reason = 'Agg', False, '未请求显示'
    if show:
        try:
            _probe_tk()
            backend, available, reason = 'TkAgg', True, 'Tk依赖及窗口系统探测成功；实际show尚未执行'
        except Exception as exc:
            reason = f'TkAgg不可用，回退Agg，未弹窗：{type(exc).__name__}: {exc}'
    matplotlib.use(backend, force=True)
    pyplot = importlib.import_module('matplotlib.pyplot')
    pyplot.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
    pyplot.rcParams['axes.unicode_minus'] = False
    return pyplot, {'backend': backend, 'available': available, 'reason': reason}


def field_health(u1, u2, intensity, grid_in, grid_out, energy_max):
    """控制和预览共享绝对幅度/功率检查；不把形状归一化用于通量。"""
    expected = grid_out.shape
    if u1.shape != grid_in.shape or u2.shape != expected or intensity.shape != expected:
        return {'status': 'fail', 'reason': '复场或强度shape与物理网格不符'}
    if not all(np.all(np.isfinite(a)) for a in (u1, u2, intensity)):
        return {'status': 'fail', 'reason': '复场/强度包含NaN或Inf'}
    with np.errstate(over='ignore', invalid='ignore'):
        pin = float(np.sum(np.abs(u1)**2) * grid_in.cell_area)
        pwin = float(np.sum(intensity) * grid_out.cell_area)
        peak = float(np.max(intensity))
        eta = pwin / pin if pin > 0 else float('nan')
        identity = np.allclose(intensity, np.abs(u2)**2, rtol=1e-12, atol=0)
    passed = all(math.isfinite(v) for v in (pin, pwin, peak, eta)) and pin > 0 and pwin > 0 and peak > 0 and 0 < eta <= energy_max and identity
    return {'status': 'pass' if passed else 'fail', 'Pin': pin, 'Pwindow': pwin,
            'peak_intensity': peak, 'eta_window': eta, 'reason': '' if passed else '光场非零/有限/模平方/绝对功率检查失败'}


def required_checks(only_mode):
    """必需项目显式列出，数量相同不能替代项目身份。"""
    height = {'aperture_area', 'height_boundaries', 'local_focusing_phase_identity', 'device_identity'}
    mapping = {'height': height, 'control': {'control_550nm'},
               'preview': height | {'psf_preview'}, 'all': height | {'control_550nm', 'psf_preview'}}
    if only_mode not in mapping:
        raise ValueError(f'未知执行模式：{only_mode}')
    return mapping[only_mode]


def checks_pass(checks, only_mode):
    return all(name in checks and checks[name].get('status') == 'pass' for name in required_checks(only_mode))


def verify_saved_artifacts(run_dir, only_mode, config, fingerprint, grid_in, grid_out, profile=None, control_profile=None):
    """完成前重读必需数组，核对字段、精确波长、坐标和器件定义。"""
    root = Path(run_dir)
    wanted = []
    if only_mode in ('height', 'preview', 'all'):
        wanted.append(('doe_continuous.npz', None, None))
    if only_mode in ('control', 'all'):
        wanted.append(('fresnel_control.npz', float(config['optical']['control_wavelength_m']), None))
    if only_mode in ('preview', 'all'):
        for lam in config['optical']['preview_wavelengths_m']:
            wanted.append((f'psf_{round(lam*1e9)}nm.npz', float(lam), fingerprint))
    for name, lam, fp in wanted:
        with np.load(root / 'arrays' / name, allow_pickle=False) as data:
            if lam is None:
                for key, value in [('x_in_m', grid_in.x.coords), ('y_in_m', grid_in.y.coords)]:
                    if not np.array_equal(data[key], value):
                        raise ValueError(f'{name}的输入坐标不符')
                if str(data['fingerprint']) != fingerprint or data['delta_h_m'].shape != grid_in.shape or data['mask'].shape != grid_in.shape or not np.all(np.isfinite(data['delta_h_m'])):
                    raise ValueError(f'{name}的器件字段不符')
                if profile is not None and not all(np.array_equal(data[key], value) for key, value in
                        [('delta_h_m', profile.delta_h), ('mask', profile.mask), ('lambda_design_m', profile.lambda_design)]):
                    raise ValueError(f'{name}保存的器件内容与实际使用高度不符')
            else:
                if float(data['wavelength_m']) != lam:
                    raise ValueError(f'{name}的波长标签与请求值不符')
                for key, value in [('x_out_m', grid_out.x.coords), ('y_out_m', grid_out.y.coords)]:
                    if not np.array_equal(data[key], value):
                        raise ValueError(f'{name}的探测器坐标不符')
                field, intensity = data['u2_complex'], data['intensity_raw']
                if field.shape != grid_out.shape or intensity.shape != grid_out.shape or not np.all(np.isfinite(field)) or not np.all(np.isfinite(intensity)) or not np.array_equal(intensity, np.abs(field)**2):
                    raise ValueError(f'{name}的原始复场/强度不符')
                if fp is not None and str(data['doe_fingerprint']) != fp:
                    raise ValueError(f'{name}的器件指纹不符')
                if name == 'fresnel_control.npz':
                    for key, value in [('x_in_m', grid_in.x.coords), ('y_in_m', grid_in.y.coords)]:
                        if not np.array_equal(data[key], value):
                            raise ValueError('控制输入坐标不符')
                    if data['delta_h_m'].shape != grid_in.shape or data['mask'].shape != grid_in.shape:
                        raise ValueError('控制器件定义缺失或shape不符')
                    if control_profile is not None and (str(data['control_fingerprint']) != control_profile.compute_fingerprint() or
                            not np.array_equal(data['delta_h_m'], control_profile.delta_h) or not np.array_equal(data['mask'], control_profile.mask)):
                        raise ValueError('控制器件内容指纹或高度不符')
                if not (math.isfinite(float(data['Pin'])) and float(data['Pin']) > 0 and math.isfinite(float(data['Pwindow'])) and
                        0 < float(data['Pwindow']) / float(data['Pin']) <= config['acceptance']['energy_efficiency_max']):
                    raise ValueError(f'{name}保存的功率不符合验收范围')
    return {'status': 'pass', 'files_verified': [name for name, _, _ in wanted]}
