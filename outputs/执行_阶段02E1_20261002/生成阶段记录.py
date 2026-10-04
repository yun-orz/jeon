# -*- coding: utf-8 -*-
"""验收结束后新增阶段文档并追加README，保留历史内容。"""
from pathlib import Path
import json,hashlib,sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics');EV=Path(__file__).resolve().parent
RUN=ROOT/'results/stage02e1/run_20261002_224704'
v=json.loads((RUN/'metrics/validation.json').read_text(encoding='utf-8'))
a=json.loads((EV/'独立参数扫描审核.json').read_text(encoding='utf-8'))
i=json.loads((EV/'入口交付验收.json').read_text(encoding='utf-8'))
# 纠正验收证据的字段名：该值为优化标志，退出码另存数值2。
if isinstance(i.get('default_no_save_exit'),bool):
    i['default_no_save_optimization_passed']=i.pop('default_no_save_exit');i['default_no_save_exit']=2
    (EV/'入口交付验收.json').write_text(json.dumps(i,ensure_ascii=False,indent=2),encoding='utf-8')
assert v['validation_passed'] and a['implementation_audit_passed'] and i['no_save_files_unchanged']
assert v['completed_jobs']==v['planned_jobs']==14 and not v['optimization_validation_passed']
lines=['# 02E-1完成记录：固定编码模型的正则化敏感性','','范围为Jeon光学编码复现＋基础重建验证，非论文网络、非双孔径。本批由Codex在Windows CPU实际实现、运行及审核。',
'', '## 目的与固定条件','',
'02D-3优化通过后仍有较大的cube误差，本批检查非负岭正则化是否导致局部峰值损失、扩散和光谱串扰。保持continuous高度DOE、三诊断波长420/540/660 nm、原97×97像元积分核、6.22 μm间距、32×32理想像平面场景、96×96显式crop测量、单位辐射响应和非负约束。使用原线性非相干强度叠加模型，不重新传播/设计/旋转PSF，不将核归一到1。',
'',
'只读来源链：02D-3 run_20261002_222854 → PG run_20261002_210155 → 成像run_20261002_203904 → 光学run_20261002_172338；四层文件SHA与相关源码指纹被核对。前述已验证源码未修改。',
'', '## 参数与运行设计','',
'预先固定α倍数0.01、0.1、1、10，各乘以本场景基线实际α约1.0589016188494556e−4。三场景共12组，从零初始化，不用真值参与优化、初始化或停止。真值只用于评价，全部参数结果报告，不按测试真值逐场景挑选参数。',
'',
'复用加速投影梯度/回溯/单调重启；主扫描归一化投影梯度映射≤1e−5且相邻目标相对变化≤1e−7，最多5000轮。线条场景α×0.01和×1另外各运行一次严格诊断，门槛1e−7与1e−9，最多5000轮。共14组，预算600秒在完整实验之间检查；单组上限为max_iterations，预算不是强制中断正在运行的一组。',
'',
'α×1的三组恢复与02D-3基线逐元素完全一致。为避免把未完成优化误报成来源错误，若该组iteration_limit，仅记录基线差异，不触发精确重现验收。',
'', '## 强凸误差界及推导','',
'G(X)=F(X)+I_{X≥0}，F=0.5||AX−y||²+α||X||²/2；α>0使G在可行域α强凸。光滑梯度g=A*(AX−y)+αX。非负约束的法向在正元素处为0，精确零处可取任意非正值，故最小范数次梯度s逐元素为：X_i>0时s_i=g_i；X_i=0时s_i=min(g_i,0)。不使用阈值把小正值视为零。',
'',
'由强凸次梯度不等式G(X*)≥G(X)+〈s,X*−X〉+α||X*−X||²/2，完成平方得G(X)−G(X*)≤||s||²/(2α)。再由次梯度强单调性〈s−0,X−X*〉≥α||X−X*||²和Cauchy–Schwarz得到||X−X*||≤||s||/α。除以||X||得到本报告的相对距离上界；X=0时相对量未定义，保存null。',
'',
'这属于本项目优化诊断数学推导，不是Jeon论文的光学公式。界对应固定离散目标的最优解，不能当作真实cube误差；小α会使界变松。浮点计算未采用区间算术，该值是按理论公式计算的数值界，不声称包含全部机器舍入误差的形式化认证。',
'', '## 12组主扫描结果','',
'|场景|α倍数|轮数|状态|cube相对L2|数据相对残差|SAM/deg|恢复/真值总功率|相对最优解距离上界|',
'|---|---:|---:|---|---:|---:|---:|---:|---:|']
for r in v['records']:
    if not r['stricter']:
        lines.append('|%s|%g|%d|%s|%.6g|%.6g|%.6g|%.6g|%.6g|'%(r['scene'],r['alpha_factor'],r['iterations'],r['solver_status'],r['evaluation']['cube_relative_l2'],r['final_optimization']['residual_relative'],r['evaluation']['sam_mean_deg'],r['recovered_to_truth_total_power'],r['certificate']['distance_upper_bound_relative']))
lines += ['',
'主扫描12组全部达到原门槛。α×0.01对应共点/分离点/线条的cube误差约18.60%/11.25%/7.83%，基线α×1约86.49%/62.55%/36.96%。小α在这些无噪声合成场景改善恢复，不证明它在真实噪声下最佳，不能将结果推广到全光谱、真实相机或双孔径。',
'',
'各参数总功率大多接近真值，但cube误差可以很大：总和不能反映能量落在哪个像元及波段；岭惩罚抑制较大的平方幅值，可能导致峰值扩散/串扰，并非只降低总通光量。',
'', '## 更严门槛诊断','',
'|α倍数|状态|轮数|cube误差|投影梯度相对量|目标相对变化|相对距离上界|',
'|---|---|---:|---:|---:|---:|---:|']
for r in v['records']:
    if r['stricter']:
        lines.append('|%g|%s|%d|%.7g|%.7g|%.7g|%.7g|'%(r['alpha_factor'],r['solver_status'],r['iterations'],r['evaluation']['cube_relative_l2'],r['final_optimization']['projected_gradient_relative'],r['final_optimization']['objective_relative_change'],r['certificate']['distance_upper_bound_relative']))
lines += ['',
'α×0.01严格诊断的梯度已≤1e−7，但目标变化约1.51e−8仍大于1e−9，因此iteration_limit标签正确。恢复相对于原门槛结果变化约0.785%，cube误差7.83%变为7.89%，相对距离上界由5.27%缩到2.21%。不能因为图像相似就称严格诊断收敛。',
'',
'α×1严格诊断达门槛，恢复变化约0.272%，cube误差36.96%变为37.07%，上界约0.0224%。目标更优不保证更接近真值。共点源小α结果上界约8.2%，下一批优先复核该数值不确定性，不直接研究新器件。',
'', '## 实际验收和证据','',
'- 正式run：results/stage02e1/run_20261002_224704；CPU耗时141.604秒（与回归并行时测得），完整计算14组，CLI退出2。validation_passed=true表示来源/算子/实现检查通过；optimization_validation_passed=false及all_reconstructions_converged=false反映严格诊断仍有一组达到上限，未改为true。',
'- 新增7项测试：独立NNLS参考、强凸界、精确零与小正值区别、非法输入、固定来源、来源α篡改及配置。首次测试替身未提供NPZ的.files接口导致测试失败，修正替身后通过；未改磁盘源数据。完整194项unittest测试通过，152.403秒。',
'- 独立审核对14份数组用逐点卷积重算拟合、直接转置重算梯度/证书/停止量；核、测量、物理坐标、真实cube与02D-3逐字段一致，α×1恢复一致，四层源SHA与源码指纹相容。',
'- 14份NPZ完整重读、14份轨迹JSON与CSV核对、评价CSV、3张统一功率色标场景图及1张参数趋势图、中文运行报告和来源证据均保存。已实际查看线条场景图，标题/统一色标可读。',
'- 真实CLI从项目外启动：缺失源非零失败；未收敛退出2；预算截断保留一组完整结果且退出2；报告写入故障留failed.json、不写最终完成标记。旧结果与失败证据均保留。',
'- 默认14组--no-save真实执行退出2（同样严格诊断上限，不是执行异常）；模拟显示用三组α×1，show调用一次且图窗关闭。项目前后文件SHA与集合不变。真实PyCharm点击/GUI尚未人工验证。',
'',
'实测版本仍为Python3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9；CPU，无新增依赖。解释器D:/dev/python/python3.10.4/python.exe。证据在项目外outputs/执行_阶段02E1_20261002，完整日志含运行失败/修正记录，不覆盖历史。',
'', '## Windows/PyCharm手动运行','',
'打开outputs/jeon2019_optics项目，选择本地解释器；新环境python -m pip install -r requirements.txt，右键main_stage02e1.py无参数运行。无需Harness、聊天状态、Notebook、Docker或GPU。',
'', '```powershell','python -B main_stage02e1.py','python -B main_stage02e1.py --no-save','python -B main_stage02e1.py --show-plots','python -B -m unittest discover -s tests','```','',
'配置集中在config_stage02e1.json：source_run为只读02D-3来源；alpha_factors为预先指定倍数且含1基线；solver为主扫描门槛/上限/回溯；stricter为代表场景与更严门槛；certificate_relative_threshold=0.05只用于标记界是否小于恢复量5%，不是新的停止条件；budget_seconds为实验间预算；其余检查、SAM阈值、dpi、保存和显示集中配置。路径基于入口__file__，更换电脑需保留完整四层来源及相容源码。',
'',
'每次新建唯一results/stage02e1/run_...；progress.json逐组记录，arrays保存完整原始数组与坐标，metrics保存轨迹/评价/验收，figures保存形状与趋势。退出0要求所有计划实验优化通过；退出2表示完整或预算截断计算中的优化验收未全通过；异常留failed.json。查看结果时先读状态，再读误差。',
'', '## 未验证与下一阶段','',
'nearest_depth的α扫描、噪声/真实相机、宽波段积分、原论文网络、双孔径及旋转方向对应仍未验证。本批单通道结果不支持双通道互补性结论。下一阶段02E-2对小α共点和线条做有界严格复核，规划已单独保存；本批未执行02E-2。']
(ROOT/'docs/stage02e1_completed.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (ROOT/'README.md').open('a',encoding='utf-8') as f:
    f.write("""
## 2026-10-02：02E-1固定模型α敏感性

在PyCharm右键`main_stage02e1.py`无参数运行，CPU，无新增依赖，实测Python3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9；安装沿用requirements.txt。路径基于入口文件，迁移需完整保留02D-3及递归PG/成像/光学来源。

配置集中在config_stage02e1.json：固定α倍数[0.01,0.1,1,10]，continuous/crop三个场景12组主扫描；线条另外两组更严诊断；max_iterations=5000，预算600秒在实验之间检查；证书阈值5%只作标记。真值不参与优化或参数选择。

```powershell
python -B main_stage02e1.py
python -B main_stage02e1.py --no-save
python -B main_stage02e1.py --show-plots
```

12组主扫描均达原门槛；一组更严诊断仍iteration_limit，因此默认退出2、总优化验收false，属于已完成的诊断结果。α×0.01线条cube误差约7.83%，基线约36.96%，但严格诊断和共点源仍有数值不确定性；无噪声结果不能用来指定真实相机最优参数。总功率接近真值不代表光谱/空间分布正确。

正式CPU运行约141.6秒，194项回归及独立审核通过。输出唯一results/stage02e1/run_...，14份原始NPZ/轨迹、4张PNG、评价CSV、来源SHA、逐组progress和中文报告。真实验证项目外启动、缺失源/未收敛/预算/保存故障及--no-save文件SHA不变；mock显示通过，真实PyCharm点击及GUI仍未人工验证。完整推导、数值与限制见docs/stage02e1_completed.md。仍为Jeon光学编码复现＋基础重建验证，非原论文网络或双孔径。
""")
manifest=json.loads((RUN/'source_manifest.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==s for k,s in manifest.items())
result=dict(stage='02E-1',formal_run=str(RUN),implementation_audit_passed=True,main_sweep_converged_count=12,
    main_sweep_count=12,stricter_converged_count=1,stricter_count=2,optimization_validation_passed=False,
    diagnostic_delivery_complete=True,regression_tests_passed=194,regression_elapsed_s=152.403,
    formal_elapsed_s=v['elapsed_s'],independent_audit=a,entry_audit=i,source_manifest_matches=True,
    next_stage_executed=False,real_GUI_verified=False,next_plan=str(EV/'下一阶段02E2详细规划.md'))
(EV/'阶段最终验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('阶段记录/README已保存，诊断状态保持真实，源码指纹匹配。')
