# -*- coding: utf-8 -*-
"""根据实际结果新增中文记录，README只追加，历史保留。"""
from pathlib import Path
import json,hashlib,sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics');EV=Path(__file__).resolve().parent
FIRST=ROOT/'results/stage02e2/run_20261002_230253';EXTRA=ROOT/'results/stage02e2/run_20261002_230506'
f=json.loads((FIRST/'metrics/validation.json').read_text(encoding='utf-8'));e=json.loads((EXTRA/'metrics/validation.json').read_text(encoding='utf-8'))
i=json.loads((EV/'入口交付验收.json').read_text(encoding='utf-8'))
a=[json.loads((EV/('独立稳定性审核_'+p.name+'.json')).read_text(encoding='utf-8')) for p in [FIRST,EXTRA]]
assert all(v['implementation_audit_passed'] for v in a) and i['no_save_files_unchanged']
assert f['optimization_validation_passed'] and not f['certificate_validation_passed'] and e['stability_validation_passed']
point=f['records'][0];line=e['records'][0]
assert point['certificate_passed'] and line['certificate_passed']
with np.load(FIRST/'arrays/continuous_lines_and_square_crop_alpha_0p01_refined.npz') as z:xf=z['reconstruction'].copy()
with np.load(EXTRA/'arrays/continuous_lines_and_square_crop_alpha_0p01_refined.npz') as z:xe=z['reconstruction'].copy()
line_change=float(np.linalg.norm(xe-xf)/np.linalg.norm(xe))
for run in [FIRST,EXTRA]:
    manifest=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==s for k,s in manifest.items())
extra_config=json.loads((ROOT/'config_stage02e2_extra.json').read_text(encoding='utf-8'));assert extra_config==e['config']
extra_entry_manifest={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in ['main_stage02e2_extra.py','config_stage02e2_extra.json']}
(EV/'追加入口交付指纹.json').write_text(json.dumps(dict(origin='独立交付审核时记录',files=extra_entry_manifest,config_matches_actual_run=True,run=str(EXTRA)),ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 02E-2完成记录：弱正则化数值稳定性复核','','本批由Codex在Windows CPU实际实现、运行及独立审核。范围仍是Jeon光学编码复现＋基础重建验证，非原论文网络、非双孔径。',
'', '## 固定条件与目的','',
'02E-1小α结果优于原基线，但原停止门槛留下数值不确定性。本批固定continuous/crop、420/540/660 nm三单色、6.22 μm像元积分核、32×32场景支持、96×96测量、单位辐射响应、无噪声及实际α=基线α×0.01≈1.0589016188494556e−6。保持同一DOE高度、同一A/y、非负约束、零初始化，不重新设计/旋转/镜像PSF，不归一化核。真值只用于最终评价。',
'',
'只读来源02E-1 run_20261002_224704，并递归复核02D-3/PG/成像/光学四层，共五层来源SHA及源码指纹。02E-1包含一个未收敛严格诊断，加载器允许计算与实现验收已完成的诊断源，但逐组验证状态，拒绝伪造总优化标志或篡改α。旧源码/配置/结果未修改。',
'', '## 运行设计与两个验收条件','',
'首轮仅共点源和线条两组，梯度门槛1e−7、目标相对变化1e−9，最大15000轮，预算300秒在完整实验之间检查；预算不强制中断正在运行的单组。与旧结果比较恢复变化、目标、cube/SAM、总功率及空间泄漏。',
'',
'分别报告solver_status与certificate_passed。第一项需要投影梯度映射及目标变化同时过门槛；第二项需要最优解距离理论上界≤当前恢复范数的1%。stability_validation_passed要求来源/算子验收、所有计划实验收敛以及所有界通过。未达界不等于未达到优化停止门槛，两个标志不得混同。',
'',
'沿用02E-1推导：精确正X处s=g，精确零处s=min(g,0)，g=A*(AX−y)+αX。α强凸性给出||X−X*||≤||s||/α、目标差≤||s||²/(2α)。相对界除以||X||，不除以真值；不把小正值视为零。此界描述固定离散正则化目标的最优解，不是真实cube误差，浮点计算也未作区间舍入包络。',
'', '## 首轮与一次追加诊断','',
'|批次/场景|状态|轮数|cube相对误差|最优解距离相对上界|1%界通过|相对02E-1恢复变化|',
'|---|---|---:|---:|---:|---|---:|']
for batch,report in [('首轮',f),('追加',e)]:
    for r in report['records']:lines.append('|%s/%s|%s|%d|%.8g|%.8g|%s|%.8g|'%(batch,r['scene'],r['solver_status'],r['iterations'],r['evaluation']['cube_relative_l2'],r['certificate']['distance_upper_bound_relative'],r['certificate_passed'],r['baseline_reconstruction_relative_change']))
lines += ['',
'首轮两组都converged，但线条界约1.07049%，略大于预设1%；首轮总stability_validation_passed=false、退出2，结果原样保留。按规划只追加线条一次，门槛1e−8/1e−10、仍15000轮/300秒、零初始化、同一α/A/y，不调整1%阈值。追加9792轮converged，界约0.06615%，退出0。没有无限重跑。',
'',
'共点源最终cube误差约18.129%，相对最优解距离上界约0.3977%；线条最终cube误差约7.929%，界约0.06615%。相对于02E-1原门槛恢复，共点变化约2.432%，线条约0.546%。线条误差由约7.832%变为7.929%，目标更优不保证更接近真值；它说明原结果并非已经精确解完。',
'',
'首轮线条与追加线条恢复相对变化为%.8g。两组最终数值目标均达到既定可信度条件，但依然有非零真值误差。总功率比仍接近1，不能代表能量正确落在每个像元/波段；不据此断言剩余误差全部来自光学编码，也不把弱α指定为真实相机最佳参数。'%line_change,
'', '## 实際执行与审核','',
'- 首轮正式run：results/stage02e2/run_20261002_230253，CPU64.332秒，2组（与回归并行测得）；追加run：results/stage02e2/run_20261002_230506，CPU55.923秒，1组。时间随负载变化。首轮退出2、追加退出0，各自状态真实保存。',
'- 新增6项测试含真实iteration_limit报告JSON、未全优化通过源允许、伪造总标志/α篡改拒绝、非法配置和缺源。完整200项unittest全部通过，151.616秒；此前NNLS/强凸界独立参考仍通过。',
'- 首轮及追加独立审核均通过：不调用生产FFT，用逐点核平移重算拟合、直接转置重算梯度/证书，核对目标、实际α、状态、固定数据与五层来源SHA。首次审核复用字段时误把原baseline_alpha与弱α数组alpha比较，已修正审核字段对应，生产数据和模型未改；修正记录保留。',
'- 正式首轮2份NPZ/轨迹JSON与CSV/3张PNG，追加1份NPZ/轨迹/2张PNG；每份完整重读、核对身份与优化轮数，评价CSV、中文报告、来源证据和源码指纹完整。已实际查看两种场景对照图，统一原始功率色标及绝对误差可读。',
'- 默认入口与追加入口均从项目外实际无参数执行。缺源CLI失败；一轮未收敛与预算截断均退出2；报告写入故障和模拟show异常均留failed.json、不留最终validation.json，图窗关闭。',
'- 默认--no-save实际复核两组，退出2（首轮线条1%界未过），模拟显示分支调用show一次且图窗关闭，项目文件SHA及集合未变化。追加入口的无参数默认运行已验证；其独立--no-save/--show-plots命令尚未单独运行，底层共用流程已经验证。真实PyCharm点击/GUI仍未人工验证。',
'',
'版本实测仍为Python3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9；解释器D:/dev/python/python3.10.4/python.exe；Windows CPU，无新增依赖/GPU要求。追加入口/配置额外交付指纹保存在项目外执行证据，实际config与追加run配置完全一致。',
'', '## Windows/PyCharm运行','',
'打开outputs/jeon2019_optics普通Python项目，配置本地解释器；新环境python -m pip install -r requirements.txt。右键main_stage02e2.py运行首轮两组；右键main_stage02e2_extra.py运行一次追加线条诊断。两者均无需参数或Harness，项目相对路径基于入口位置。',
'', '```powershell','python -B main_stage02e2.py','python -B main_stage02e2_extra.py','python -B main_stage02e2.py --no-save','python -B main_stage02e2.py --show-plots','python -B -m unittest discover -s tests','```','',
'参数分别集中在config_stage02e2.json/config_stage02e2_extra.json。scenes为本批共点或线条，alpha_factor固定0.01；solver为上限/门槛/回溯；certificate_relative_threshold=0.01为距离界条件；budget_seconds为组间预算；其余检查/SAM/dpi/保存/显示集中配置。迁移须保留完整五层来源及相容源码，未使用聊天状态、Docker或Notebook。',
'',
'输出唯一results/stage02e2/run_...，progress逐组保存；arrays保留原始物理数组/坐标/恢复，metrics保存轨迹及评价，figures保存对照图与停止量曲线；最终validation最后写入。退出0要求stability_validation_passed=true；退出2可能是未收敛、未过界或预算截断，先读各标志判断原因。异常则failed.json；不覆盖旧目录。',
'', '## 未验证与下一步','',
'nearest_depth弱α对照、噪声/实际相机、宽波段积分、网络与双孔径尚未验证；论文旋转方向对应仍未解决。此批只验证固定单通道目标的数值稳定性，不补足这些物理事项。下一阶段02E-3研究最近深度级量化器件，对比时区分相同相对α和共同实际α；详细规划已保存，本批未执行该阶段。']
(ROOT/'docs/stage02e2_completed.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (ROOT/'README.md').open('a',encoding='utf-8') as stream:
    stream.write("""
## 2026-10-02：02E-2弱α数值稳定性复核

PyCharm右键main_stage02e2.py无参数运行共点/线条首轮，配置config_stage02e2.json；右键main_stage02e2_extra.py运行一次更严线条诊断，配置config_stage02e2_extra.json。仍是本地CPU、Python3.10.4/NumPy2.2.6/SciPy1.15.3/Matplotlib3.10.9，无新增依赖，路径基于入口位置。安装沿用requirements.txt，迁移需完整五层来源run与相容源码。

固定原α×0.01及同一A/y/非负约束/零初始化，最多15000轮，预算300秒在组间检查。首轮门槛1e-7/1e-9，两组收敛但线条距离界1.07%略高于1%，默认退出2；一次追加线条门槛1e-8/1e-10，9792轮通过，退出0。1%阈值保持不变，不将首轮修改为通过。

最终共点cube误差约18.13%、线条约7.93%；相对于固定目标最优解的距离理论上界约0.40%/0.066%，并非真实cube误差界。首轮/追加CPU约64.3/55.9秒；200项回归与独立直接卷积/转置审核通过。

```powershell
python -B main_stage02e2.py
python -B main_stage02e2_extra.py
python -B main_stage02e2.py --no-save
python -B main_stage02e2.py --show-plots
```

新结果唯一results/stage02e2/run_...，含原始NPZ/优化JSON与CSV、对照PNG、评价、progress及中文报告。退出0要求优化与1%界都通过，退出2需查看各标志；报告保存和显示失败留failed.json且不写完成标记。默认无保存及mock显示验证通过，真实PyCharm/GUI未人工验证；追加入口独立显示/无保存命令尚未单独运行。完整记录见docs/stage02e2_completed.md，仍为Jeon光学编码复现＋基础重建验证。
""")
result=dict(stage='02E-2',implementation_audit_passed=True,stability_validation_passed_for_final_two_cases=True,
    initial_run=str(FIRST),additional_run=str(EXTRA),initial_stability_passed=False,additional_stability_passed=True,
    final_records=[point,line],regression_tests_passed=200,regression_elapsed_s=151.616,
    entry_audit=i,independent_audits=a,source_manifest_matches=True,extra_entry_files=extra_entry_manifest,
    real_PyCharm_and_GUI_verified=False,additional_GUI_no_save_individually_verified=False,
    next_stage_executed=False,next_plan=str(EV/'下一阶段02E3详细规划.md'))
(EV/'阶段最终验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('中文记录/README/最终验收已保存，首轮与追加状态各自保留。')
