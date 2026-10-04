# -*- coding: utf-8 -*-
"""图示器件差值及优化理论界，所有图存项目外证据目录。"""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(r'D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics');EV=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from optics.stage02_runtime import configure_plotting
plt,backend=configure_plotting(False)
run=ROOT/'results/stage02e4/run_20261002_233752'
v=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(figsize=(9,4.3));names={'coincident_points':'共点源','separated_points':'分离点源','lines_and_square':'线条与方块'}
rows=[]
for k,pair in enumerate(v['pairs']):
    group=[r for r in v['records'] if r['scene']==pair['scene']]
    with np.load(run/'arrays'/(group[0]['identity']+'.npz')) as z:norm=float(np.linalg.norm(z['truth']))
    radius=sum(r['certificate']['distance_upper_bound'] for r in group)/norm
    delta=pair['nearest_minus_continuous_cube_error'];cross=delta-radius<=0<=delta+radius
    ax.errorbar(delta*100,k,xerr=radius*100,fmt='o',capsize=6,color='#c56a19' if cross else '#256ba0')
    rows.append(dict(scene=pair['scene'],delta=delta,radius=radius,lower=delta-radius,upper=delta+radius,crosses_zero=cross))
ax.axvline(0,color='black',ls='--',lw=1)
ax.set_yticks(range(3),[names[p['scene']] for p in v['pairs']]);ax.invert_yaxis()
ax.set_xlabel('cube相对L2误差差（量化−连续）/百分点')
ax.set_title('误差差值与优化理论界：线条场景区间跨零，不能据点估计排序')
ax.grid(axis='x',alpha=.3)
fig.text(.5,.02,'固定共同实际α；理论数值界未作区间舍入包络，仅针对当前理想模型',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.06,1,1))
fig.savefig(EV/'器件差异优化误差界.png',dpi=180);fig.savefig(EV/'器件差异优化误差界.svg');plt.close(fig)
(EV/'差异界图数值.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('独立差异界图和数值已保存')
