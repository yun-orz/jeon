# -*- coding: utf-8 -*-
"""按真实结果生成对照不确定性与中文完成记录，保留旧文件。"""
from pathlib import Path
import hashlib,json,re,sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics');EV=Path(__file__).resolve().parent
log=(EV/'正式运行.log').read_text(encoding='utf-8')
summary=json.loads([line for line in log.splitlines() if line.startswith('{')][-1]);RUN=Path(summary['run_dir'])
v=json.loads((RUN/'metrics/validation.json').read_text(encoding='utf-8'))
a=json.loads((EV/('独立器件对照审核_'+RUN.name+'.json')).read_text(encoding='utf-8'))
i=json.loads((EV/'入口交付验收.json').read_text(encoding='utf-8'))
assert a['implementation_audit_passed'] and i['no_save_files_unchanged'] and v['completed_jobs']==6
match=re.search(r'Ran (\d+) tests in ([0-9.]+)s',(EV/'回归.log').read_text(encoding='utf-8'));count=int(match[1]);test_seconds=float(match[2]);assert count==206
manifest=json.loads((RUN/'source_manifest.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==s for k,s in manifest.items())
uncertainty=[]
for pair in v['pairs']:
    cr=next(r for r in v['records'] if r['scene']==pair['scene'] and r['device']=='continuous')
    nr=next(r for r in v['records'] if r['scene']==pair['scene'] and r['device']=='nearest_depth')
    with np.load(RUN/'arrays'/(cr['identity']+'.npz')) as z:tnorm=float(np.linalg.norm(z['truth']))
    radius=(cr['certificate']['distance_upper_bound']+nr['certificate']['distance_upper_bound'])/tnorm
    delta=nr['evaluation']['cube_relative_l2']-cr['evaluation']['cube_relative_l2']
    uncertainty.append(dict(scene=pair['scene'],nearest_minus_continuous_cube_error=delta,optimization_uncertainty_radius=radius,
        lower=delta-radius,upper=delta+radius,ordering_supported_by_theoretical_bounds=bool(delta-radius>0 or delta+radius<0),
        caveat='不同实际α的模型对照；理论界数值计算未作区间舍入包络，非纯量化因果结论'))
(EV/'器件差异优化不确定性.json').write_text(json.dumps(uncertainty,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 02E-3完成记录：连续与最近深度级器件弱α对照','','由Codex在Windows CPU实际实现、运行及审核；范围为Jeon光学编码复现＋基础重建验证，非原论文网络或双孔径。',
'', '## 固定物理与比较策略','',
'比较continuous和nearest_depth两种完整单孔径高度器件。沿用各自同一张高度图的420/540/660 nm响应、q8像元积分功率核、6.22 μm统一探测器网格、三个32×32理想像平面场景和96×96 crop。非相干各波段功率相加，单位辐射响应、无噪声；不重新旋转/设计PSF，不将有限窗口捕获核归一到1。最近深度级采用之前02C已验证的制造量化规则，本阶段不改材料或光学传播。',
'',
'实际α按各器件02D-3基线×0.01：continuous约1.0589016188494556e−6，nearest_depth约1.0032299973098796e−6。属于相同相对正则化尺度比较，实际α不同，因此不能把恢复差异全部归因于高度量化。共同实际α对照将在下一批独立验证；本批未运行。',
'',
'只读来源D3 run_20261002_222854及PG/成像/光学链，并读取E2首轮run_20261002_230253和追加run_20261002_230506作连续器件代表对照，递归E1，共7个唯一源run。逐组验证α/数据/坐标/拟合/轨迹/状态/证书与SHA；允许E2首轮证书未全通过，保留其false标志。旧阶段源码/配置/结果未修改。',
'', '## 运行与停止','',
'六组零初始化、非负岭目标、加速投影梯度/回溯/单调重启；门槛1e−8/1e−10同时满足，max_iterations=15000。预算600秒在完整组间检查，单组受最大轮数限制，不强制中断当前组；逐组保存进度。真值不参与迭代、停止或选择α。',
'',
'分别验收优化停止门槛与相对于自身固定目标最优解的1%距离理论界；stability_validation_passed要求全部计划组的来源/算子/优化/界通过。理论界来自α强凸性，s在X>0处为梯度、X=0处为min(梯度,0)，d=||s||/α；不把小正值视为零，不把投影梯度映射直接当s。界不是恢复真值误差，也未作区间机器舍入包络。详见02E-1推导。',
'', '## 正式六组结果','',
'|器件/场景|状态|轮数|cube相对L2|数据残差|SAM/deg|总功率比|最优解距离相对界|',
'|---|---|---:|---:|---:|---:|---:|---:|']
for r in v['records']:lines.append('|%s/%s|%s|%d|%.8g|%.8g|%.8g|%.8g|%.8g|'%(r['device'],r['scene'],r['solver_status'],r['iterations'],r['evaluation']['cube_relative_l2'],r['final_optimization']['residual_relative'],r['evaluation']['sam_mean_deg'],r['recovered_to_truth_total_power'],r['certificate']['distance_upper_bound_relative']))
lines += ['', '总实现验收=%s，优化验收=%s，1%%界验收=%s，数值稳定性验收=%s。首轮完整保存，没有改变标签或阈值。'%(v['validation_passed'],v['optimization_validation_passed'],v['certificate_validation_passed'],v['stability_validation_passed']),
'', '## 小差异如何判断','',
'设真值T、当前cube误差e=||X−T||/||T||，距离理论上界d=||X−X*||的上界。三角不等式得到充分优化解的cube误差区间[max(0,e−d/||T||),e+d/||T||]。两器件误差差δ的区间为δ±(d_cont+d_near)/||T||。跨0时不能据点估计排序；1%距离门槛并不自动支持小于1%的器件差异结论。',
'', '|场景|量化−连续cube误差|优化不确定性半径|理论差值区间|界支持误差排序|','|---|---:|---:|---|---|']
for u in uncertainty:lines.append('|%s|%.8g|%.8g|[%.8g, %.8g]|%s|'%(u['scene'],u['nearest_minus_continuous_cube_error'],u['optimization_uncertainty_radius'],u['lower'],u['upper'],u['ordering_supported_by_theoretical_bounds']))
lines += ['',
'本表仅针对两个不同实际α的固定模型；即使界支持排序，也不是纯量化效果或真实相机性能结论。未用真值挑参数，也不据合成无噪声结果指定真实相机最佳α。总功率接近1或较小拟合残差不能替代光谱/空间恢复误差。',
'', '## E2参考对照','']
for r in v['records']:
    for ref in r['reference_comparisons']:lines.append('- %s/%s 对 %s：恢复相对变化 %.8g，cube误差变化 %.8g；原参考状态%s，1%%界通过%s。'%(r['device'],r['scene'],Path(ref['run']).name,ref['reconstruction_relative_change'],ref['cube_error_change'],ref['solver_status'],ref['certificate_passed']))
lines += ['', '## 实际验证与文件','',
'- 正式run：%s，CPU %.3f秒（与回归并行时测得），completed_jobs=%d，CLI退出%d。各门槛与状态保存在metrics/validation.json，不能只读completed。'%(RUN.relative_to(ROOT).as_posix(),v['elapsed_s'],v['completed_jobs'],0 if v['stability_validation_passed'] else 2),
'- 新增6项有效检查含器件/场景配对、E2未过证书源允许、伪造参考标志与α篡改拒绝、非法配置、重复参考与六组实际一轮报告。完整%d项unittest全部通过，%.3f秒；此前NNLS/强凸界与光学物理检查仍通过。'%(count,test_seconds),
'- 独立审核六份NPZ：逐点核平移重算前向、直接转置重算梯度/证书/目标/停止量，与D3固定真值/核/测量/坐标逐字段比对，七份源run SHA和源码指纹一致。',
'- 六份原始NPZ、六组逐轮JSON/CSV、评价CSV、六张历史/新结果图、三张器件成对图和一张停止曲线、progress、中文运行报告与来源证据完整；数组完整重读，JSON轨迹及CSV轮数核对。成对图统一原始功率色标，实际查看线条与共点图。',
'- 从项目外真实无参数执行；缺源失败，一轮未收敛和预算截断退出2；报告写入和模拟show故障留failed.json而无完成标记，图窗关闭。默认六组--no-save及mock显示分支执行，项目%s个文件SHA/集合未变化；真实PyCharm点击及GUI仍未人工验证。'%i['files_checked'],
'',
'实测Python3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9；解释器D:/dev/python/python3.10.4/python.exe，Windows CPU，无新增依赖/GPU。运行时长随负载改变。外层执行目录保留完整测试、正式运行、独立审核、入口验收及理论不确定性JSON证据。',
'', '## PyCharm/Windows手动运行','',
'打开outputs/jeon2019_optics项目，选本地解释器；新环境python -m pip install -r requirements.txt；右键main_stage02e3.py即可CPU无参数运行。配置集中在config_stage02e3.json，所有相对路径基于入口__file__；迁移需保留七份源run完整文件与相容源码。无需聊天状态、Harness、Notebook、Docker或云端。',
'', '```powershell','python -B main_stage02e3.py','python -B main_stage02e3.py --no-save','python -B main_stage02e3.py --show-plots','python -B -m unittest discover -s tests','```','',
'配置source_run指定D3；reference_runs指定两份E2；devices/scenes可取本阶段子集；alpha_factor固定0.01；solver为门槛/轮数/回溯；certificate_relative_threshold=0.01；budget_seconds为组间预算；其余检查种子/容差/SAM阈值/dpi/保存/显示集中配置。修改α策略需独立下一阶段，不绕过范围检查。',
'',
'每次唯一results/stage02e3/run_...，arrays保留原始物理数组与恢复，metrics保留轨迹/评价/验收，figures保留对照图；progress逐组更新，最终validation最后写。退出0要求全计划数值稳定性验收通过；退出2需查优化/界/预算各标志；异常留failed.json。不覆盖/删除旧结果。',
'', '## 未验证与下一步','',
'本批未研究共同实际α、噪声/标定误差、真实相机、宽波段、原网络或双孔径；原论文旋转方向对应仍未解决。下一阶段02E-4固定共同实际α，仅新增量化器件三场景并严格验证连续结果复用，规划已保存，本批未执行。']
(ROOT/'docs/stage02e3_completed.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (ROOT/'README.md').open('a',encoding='utf-8') as stream:
    stream.write("""
## 2026-10-02：02E-3连续与量化器件弱α对照

在PyCharm右键main_stage02e3.py无参数运行，配置config_stage02e3.json。continuous/nearest_depth各三个场景，使用各自D3实际α×0.01，因此是相同相对正则化尺度对照，实际α略不同，不能全部归因于量化。三单色、单完整孔径、无噪声、单位辐射响应，未扩展双孔径或网络。

参数集中设置devices/scenes、source_run/reference_runs、solver门槛1e-8/1e-10、最多15000轮、1%距离理论界、组间600秒预算及保存/显示/dpi。路径基于入口文件；迁移保留七份源run及相容源码。本地CPU，Python3.10.4/NumPy2.2.6/SciPy1.15.3/Matplotlib3.10.9，无新增依赖，安装沿用requirements.txt。

```powershell
python -B main_stage02e3.py
python -B main_stage02e3.py --no-save
python -B main_stage02e3.py --show-plots
```

新结果唯一results/stage02e3/run_...：6份原始NPZ/逐轮JSON和CSV、10张统一功率或停止量PNG、评价CSV、来源SHA、progress和中文报告。完成标记最后写；退出0要求全部优化与1%距离界通过，退出2查各状态，异常留failed.json。真实CLI/无保存SHA/mock显示已验证，真实PyCharm点击/GUI仍未人工验证。

完整数值、206项回归和独立审核、器件差异的优化不确定性解释见docs/stage02e3_completed.md。不要仅按小的点估计差异排序，也不要将最优解距离界当真值误差界。仍为Jeon光学编码复现＋基础重建验证。
""")
result=dict(stage='02E-3',formal_run=str(RUN),implementation_audit_passed=True,optimization_validation_passed=v['optimization_validation_passed'],
    certificate_validation_passed=v['certificate_validation_passed'],stability_validation_passed=v['stability_validation_passed'],
    completed_jobs=6,regression_tests_passed=count,regression_elapsed_s=test_seconds,formal_elapsed_s=v['elapsed_s'],
    independent_audit=a,entry_audit=i,comparison_strategy=v['comparison_strategy'],comparison_uncertainty=uncertainty,
    source_manifest_matches=True,real_PyCharm_and_GUI_verified=False,next_stage_executed=False,next_plan=str(EV/'下一阶段02E4详细规划.md'))
(EV/'阶段最终验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('中文阶段记录/README与差异不确定性记录已保存，正式源码指纹匹配。')
