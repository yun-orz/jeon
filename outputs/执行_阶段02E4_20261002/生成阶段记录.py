# -*- coding: utf-8 -*-
"""依据实测共同α结果新增中文记录并追加README，保留历史。"""
from pathlib import Path
import json,hashlib,re,sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics');EV=Path(__file__).resolve().parent
summary=json.loads([s for s in (EV/'正式运行.log').read_text(encoding='utf-8').splitlines() if s.startswith('{')][-1]);RUN=Path(summary['run_dir'])
v=json.loads((RUN/'metrics/validation.json').read_text(encoding='utf-8'))
a=json.loads((EV/('独立共同α审核_'+RUN.name+'.json')).read_text(encoding='utf-8'));i=json.loads((EV/'入口交付验收.json').read_text(encoding='utf-8'))
assert v['stability_validation_passed'] and a['implementation_audit_passed'] and i['no_save_files_unchanged']
assert v['reused_count']==v['computed_count']==3 and v['completed_jobs']==6
match=re.search(r'Ran (\d+) tests in ([0-9.]+)s',(EV/'回归.log').read_text(encoding='utf-8'));count=int(match[1]);seconds=float(match[2]);assert count==214
manifest=json.loads((RUN/'source_manifest.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==s for k,s in manifest.items())
lines=['# 02E-4完成记录：固定共同实际α的器件对照','','本批由Codex在Windows CPU实际实现、运行与独立审核。仍为Jeon光学编码复现＋基础重建验证，非原论文网络、非双孔径。',
'', '## 目的与物理条件','',
'02E-3采用各器件自己的相对α，实际α略不同，影响比较。02E-4预先固定共同实际α为continuous D3 crop基线×0.01，从只读来源读取精确值，未按真值选参数。共同α=%.17g。'%(v['common_alpha']),
'',
'保持continuous和nearest_depth两张既定完整单孔径DOE高度图的三单色420/540/660 nm响应、q8像元积分功率核、6.22 μm统一探测器网格、32×32场景支持与96×96 crop。非相干功率相加、单位辐射响应、无噪声。未改材料/传播、未重新旋转或设计PSF、未将有限窗口通光核归一到1。',
'',
'连续器件相对于自身D3 α倍数为0.01，量化器件约0.01055492381297267，实际α逐元素完全相等。数组身份前缀沿用E3来源名，common_alpha后缀与实际alpha/alpha_factor字段标明当前策略，不能从旧身份中的0p01推断量化器件仍用原相对α。',
'', '## 复用和新增计算','',
'只读源为E3 run_20261002_231844及递归D3/PG/成像/光学/E1/E2，8份唯一源run。检查固定数据、实际α、源码指纹、拟合、轨迹、评价、优化状态、证书和源SHA。三组continuous只有实际α/完整solver配置/零初始化/收敛与本批距离界条件均相容时才复用；不相容则明确重算。篡改源数据是来源失败，不以重算掩盖。',
'',
'本次复用3组continuous，重新零初始化计算3组nearest_depth。复用NPZ逐字段与E3一致，复用历史轨迹完全一致。execution=reused_verified、new_iterations=0、当前优化elapsed_s=0；source_elapsed_s及轨迹中的历史elapsed_s保留原时长，不当作本轮计算。computed记录才是当前迭代。整个入口总耗时包括来源审核、实际计算、绘图与保存。',
'',
'默认门槛1e−8/1e−10、max_iterations=15000、1%最优解距离理论界，预算300秒在完整组间检查，不强制中断当前组。复用条件因门槛/轮数/开关变化失效时，continuous也会重算，预算与计算组数相应变化。真值仅用于评价，未参与初始化、迭代或停止。',
'',
'目标F=0.5||AX−y||²+α||X||²/2、X≥0；复用加速投影梯度/回溯/单调重启。对旧E3结果的目标变化同样按本批共同α重算，避免比较不同目标。单位仍为原相对功率，不是相机电子数。',
'', '## 六组完整结果（含三组复用）','',
'|器件/场景|执行|优化轮数（历史或新）|新增轮数|cube相对L2|数据残差|SAM/deg|相对最优解距离上界|',
'|---|---|---:|---:|---:|---:|---:|---:|']
for r in v['records']:lines.append('|%s/%s|%s|%d|%d|%.8g|%.8g|%.8g|%.8g|'%(r['device'],r['scene'],r['execution'],r['iterations'],r['new_iterations'],r['evaluation']['cube_relative_l2'],r['final_optimization']['residual_relative'],r['evaluation']['sam_mean_deg'],r['certificate']['distance_upper_bound_relative']))
lines += ['',
'六组均converged且通过1%距离理论界；实现、优化、证书及stability_validation_passed均true，正式退出0。无额外更严诊断或换阈值。相对最优解距离界不是对真实cube的误差界，仍有约8%–19%的真实合成场景恢复误差。',
'', '## 器件差异及其不确定性','',
'α相同消除了显式正则化尺度差异，当前理想模型中的器件仍有不同物理核和各自成像测量。设e=||X−T||/||T||、最优解距离界d=||s||/α，s为非负约束最小范数次梯度。由三角不等式，充分优化解的cube误差属于[max(0,e−d/||T||),e+d/||T||]；量化−连续误差差δ的区间为δ±(d_cont+d_near)/||T||。未对浮点舍入作区间包络，属于按理论公式计算的数值界。',
'', '|场景|量化−连续cube误差|理论差值区间|界支持排序|','|---|---:|---|---|']
for p in v['pairs']:lines.append('|%s|%.8g|[%.8g, %.8g]|%s|'%(p['scene'],p['nearest_minus_continuous_cube_error'],p['lower'],p['upper'],p['ordering_supported_by_theoretical_bounds']))
lines += ['',
'共点源和分离点源区间严格为正，在当前共同α/三单色/无噪声模型中，量化器件的充分优化cube误差更高；这是这两个具体场景的模型结果，不是普遍量化性能结论。线条场景点估计为量化稍高，但区间跨0，不能可靠排序。与E3使用不同实际α时点估计的顺序变化，说明公平比较必须明确正则化策略。',
'',
'总功率比接近1不能代替局部峰值/空间泄漏/光谱误差，小测量残差也不是恢复准确性证明。较小α不能由这批无噪声结果指定为真实相机最佳参数。差异界PNG/SVG和原始数值另存项目外执行证据，已实际查看。',
'', '## 实际运行与审核','',
'- 正式run：%s；总CPU入口耗时%.3f秒（与回归并行时测得），3复用+3新增、退出0。'%(RUN.relative_to(ROOT).as_posix(),v['elapsed_s']),
'- 新增8项检查：复用不调用优化器/逐条件失效、真实一轮回退到六组重新计算、缓存恢复篡改拒绝、非法配置、共同α与只读来源、独立向量三角界/跨零/非法区间。完整%d项unittest通过，%.3f秒；已有物理/NNLS/KKT检查仍通过。'%(count,seconds),
'- 独立审核逐字段比较复用NPZ与历史轨迹，直接逐点核平移及直接转置重算六组拟合/梯度/目标/距离界/状态；不调用生产FFT或差异区间函数，独立核对共同α和三场景区间。8份源run SHA及源码指纹一致。',
'- 6份原始NPZ、6组轨迹JSON/CSV、评价CSV、6张历史/当前图、3张统一功率器件图、1张优化曲线、来源/源码证据、progress及中文报告全部保存重读。缓存曲线和原时间明确为历史。实际查看线条成对图与差异界图，原始色标及说明可读。',
'- 项目外真实无参数运行；缺源失败、一轮（复用条件失效）和预算截断退出2；报告保存/模拟show故障留failed.json且无最终完成标记，关闭图窗。默认--no-save与mock显示已运行，项目前后%s个文件SHA和集合未变；真实PyCharm点击/GUI仍未人工验证。'%i['files_checked'],
'',
'实测Python3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9；解释器D:/dev/python/python3.10.4/python.exe，Windows CPU，无新增依赖或GPU。没有删除/覆盖旧结果。',
'', '## Windows/PyCharm手动运行','',
'打开outputs/jeon2019_optics普通项目，配置上述本地解释器；新环境python -m pip install -r requirements.txt。右键main_stage02e4.py即可无参数CPU运行，不依赖Harness/聊天状态/Notebook/Docker/云端。',
'', '```powershell','python -B main_stage02e4.py','python -B main_stage02e4.py --no-save','python -B main_stage02e4.py --show-plots','python -B -m unittest discover -s tests','```','',
'配置集中在config_stage02e4.json：source_run为E3只读来源；alpha_strategy/reference_alpha_factor确定共同α，不接受按测试真值另选参数；reuse_continuous为明确复用开关；devices/scenes为本阶段子集；solver、1%距离界、组间预算、算子检查、SAM、dpi和保存/显示集中设置。修改solver会使缓存失效，源仍不能被篡改。相对路径基于入口__file__，迁移保留8份源run和相容源码。',
'',
'每次新建唯一results/stage02e4/run_...，progress逐组记录；arrays/metrics/figures保存原始物理数据、执行身份、轨迹与对照；最终validation最后写。退出0要求所有计划组优化与距离界验收通过，退出2查未收敛/证书/预算各标志；异常failed.json。缓存组不是本轮重新迭代。',
'', '## 未验证及下一阶段','',
'尚未引入噪声/读出电子响应、真实相机或宽波段，未复现网络、未扩展双孔径，原论文旋转方向对应仍未解决。本批仅排除显式α混杂并验证固定模型恢复。下一阶段02F-1先建立带电子数单位的合成光子/读出噪声测量，统计验收通过后再规划噪声重建；G/QE等预先假设不当作论文相机标定，本批未执行。']
(ROOT/'docs/stage02e4_completed.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (ROOT/'README.md').open('a',encoding='utf-8') as f:
    f.write("""
## 2026-10-02：02E-4固定共同实际α对照

PyCharm右键main_stage02e4.py无参数运行，集中配置config_stage02e4.json，共同实际α从continuous D3基线×0.01读取，约1.0589016188494555e-6。本批审核复用三组continuous，重新计算三组nearest_depth；α相同，核/测量各自保持物理通光量。配置不相容时明确重算，不伪装成复用。

复用记录execution=reused_verified、new_iterations=0，历史轨迹时间保留；computed才是当前迭代。默认门槛1e-8/1e-10、15000轮、1%距离界及组间300秒预算。三组新增均收敛，六组完整优化/界验收通过；总入口约134.8秒，214项回归和独立复用/卷积/转置/区间审核通过。

```powershell
python -B main_stage02e4.py
python -B main_stage02e4.py --no-save
python -B main_stage02e4.py --show-plots
```

连续/量化cube误差：共点约18.08%/18.78%，分离点约11.08%/11.23%，线条约7.93%/7.99%。线条差异的优化理论区间跨零，不能可靠排序；所有结果仅针对当前无噪声三单色模型。

新结果唯一results/stage02e4/run_...，含6份NPZ/轨迹、10张PNG、评价/执行身份/来源指纹/progress和中文报告。退出0要求全部计划组优化及距离界通过，退出2查各状态；异常留failed.json。实测Python3.10.4/NumPy2.2.6/SciPy1.15.3/Matplotlib3.10.9，CPU，无新增依赖；安装沿用requirements.txt，路径基于入口文件，迁移保留8份源run和相容源码。真实CLI/无保存SHA/mock显示已验证，真实PyCharm/GUI仍未人工验证。详细推导、缓存条件和记录见docs/stage02e4_completed.md；仍为Jeon光学编码复现＋基础重建验证。
""")
result=dict(stage='02E-4',formal_run=str(RUN),implementation_audit_passed=True,stability_validation_passed=True,
    common_alpha=v['common_alpha'],reused_count=3,computed_count=3,regression_tests_passed=count,regression_elapsed_s=seconds,
    formal_elapsed_s=v['elapsed_s'],independent_audit=a,entry_audit=i,pairs=v['pairs'],source_manifest_matches=True,
    real_PyCharm_and_GUI_verified=False,next_stage_executed=False,next_plan=str(EV/'下一阶段02F1详细规划.md'))
(EV/'阶段最终验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('共同α阶段记录/README/最终验收已保存；正式源码指纹仍匹配。')
