"""独立复核已保存M1数组与指纹；仅存档审核，不将CPU核对冒充CUDA重跑。"""
import json,sys,hashlib,subprocess
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
ROOT=Path(__file__).resolve().parents[6]
ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ENGINE/'reproduction_v1'))
from audit import verify_manifest,read_json,safe_path,g0_check
from m1_resume import sha256_file
RUN=ENGINE/'results/g3_gpu/run_20261007_103806'
def tree(path):
    return {p.relative_to(path).as_posix():sha256_file(p) for p in sorted(path.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
def relative(a,b):
    return float(np.linalg.norm(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))/max(float(np.linalg.norm(np.asarray(b,dtype=np.float64))),1e-30))
report=read_json(RUN/'metrics/validation.json');summary=report['summary'];old=read_json(ROOT/'outputs/执行_M1_GPU环境_20261007/最终审核.json')
run_actual=tree(RUN)
run_difference=sorted(k for k in set(run_actual)|set(old['run_sha256']) if run_actual.get(k)!=old['run_sha256'].get(k))
codes=read_json(RUN/'source_manifest.json');code_difference=[name for name,digest in codes.items() if sha256_file(ENGINE/name)!=digest]
evidence=read_json(RUN/'source_evidence.json');source_comparison={name:tree(Path(row['path']))==row['sha256'] for name,row in evidence.items()}
with np.load(RUN/'arrays/gpu_checks.npz',allow_pickle=False) as archive:data={name:archive[name].copy() for name in archive.files}
kernels=data['kernels'];response=data['response']
def forward(x):
    blurred=np.stack([fftconvolve(x[:,:,b],kernels[b],mode='same') for b in range(25)],axis=-1)
    return np.einsum('hwb,cb->hwc',blurred,response)
def adjoint(y):
    mixed=np.einsum('hwc,cb->hwb',y,response)
    return np.stack([fftconvolve(mixed[:,:,b],kernels[b,::-1,::-1],mode='same') for b in range(25)],axis=-1)
f=forward(data['optical_x'])
optical={'forward':relative(data['optical_forward'],f),'adjoint':relative(data['optical_adjoint'],adjoint(data['optical_y'])),'gradient':relative(data['optical_gradient'],adjoint(f-data['optical_y']))}
prediction={'gpu_vs_new_cpu':relative(data['gpu_predictions'],data['new_cpu_predictions']),'new_cpu_vs_saved_cpu':relative(data['new_cpu_predictions'],data['saved_cpu_predictions']),'gpu_vs_saved_cpu':relative(data['gpu_predictions'],data['saved_cpu_predictions'])}
stages={name:relative(data['gpu_trace_'+name],data['cpu_trace_'+name]) for name in ['initial','previous','prior','data_gradient','updated','epsilon','rho','threshold']}
eq=[];back=adjoint(data['validation_measurements'][0].astype(np.float64))
for i in range(3):
    previous=data['gpu_trace_previous'][i].astype(np.float64);prior=data['gpu_trace_prior'][i].astype(np.float64)
    gradient=adjoint(forward(previous))-back
    expected=previous-data['gpu_trace_epsilon'][i]*(gradient+data['gpu_trace_rho'][i]*(previous-prior))
    eq.append(relative(data['gpu_trace_updated'][i],expected))
config=read_json(ENGINE/'config_g3_gpu.json');effective=read_json(RUN/'config_effective.json')
config_exact=all(effective[k]==v for k,v in config.items() if k!='runtime')
thresholds=config['thresholds'];loss=abs(summary['smoke']['gpu']['loss']-summary['smoke']['cpu']['loss'])/abs(summary['smoke']['cpu']['loss'])
metric_error=max(abs(prediction[k]-summary['prediction_checks'][k]) for k in prediction)
metric_error=max(metric_error,max(abs(stages[k]-summary['stage_checks'][k]) for k in stages))
index=read_json(ENGINE/'reproduction_v1/m0_index.json');protection=verify_manifest(read_json(safe_path(index['protection'])));g0=g0_check()
cpu=subprocess.run(['D:/dev/python/python3.10.4/python.exe','-B',str(ROOT/'outputs/执行_GPU验证_20261006/cpu_environment_snapshot.py')],capture_output=True,text=True,encoding='utf8')
cpu_before=json.loads((ROOT/'outputs/执行_GPU验证_20261006/cpu_environment_before.json').read_text(encoding='utf-8-sig'))
cpu_unchanged=cpu.returncode==0 and json.loads(cpu.stdout)==cpu_before
passed=not run_difference and not code_difference and all(source_comparison.values()) and config_exact and max(optical.values())<=1e-10 and max(prediction.values())<=1e-5 and max(stages.values())<=1e-5 and max(eq)<=1e-5 and metric_error<=1e-12 and loss<=1e-5 and summary['smoke']['gradient_relative']<=1e-4 and summary['smoke']['updated_parameter_relative']<=1e-5 and protection['passed'] and g0['passed'] and cpu_unchanged
result={'status':'passed' if passed else 'failed','scope':'仅已保存GPU验收结果的独立存档核对；不代表本次CUDA执行通过','official_run':str(RUN),'run_files_checked':len(run_actual),'run_sha_difference':run_difference,'code_files_checked':len(codes),'code_sha_difference':code_difference,'sources_unchanged':source_comparison,'configuration_exact':config_exact,'independent_scipy_optical':optical,'independent_predictions':prediction,'independent_stages':stages,'independent_equation21':eq,'summary_metric_max_error':metric_error,'loss_relative_from_saved_scalars':loss,'gradient_adam_scope':'梯度及Adam参数完整数组未保存，当前CUDA不可用，只核对原记录门槛及历史独立审核，不宣称本次重算','cpu_dependencies_unchanged':cpu_unchanged,'protection':protection,'g0':g0,'cuda_rerun_passed':False,'test_predictions_read':False}
with Path(__file__).with_name('offline_archive_review.json').open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
if not passed:sys.exit(2)
