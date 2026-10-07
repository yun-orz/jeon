# LightPipes填充与核单元平均拆分

本阶段为固定Jeon式CW诊断DOE的传播检查，不是作者2019实现，不改库源码/DOE，不替换G2–G5核或训练模型。

## PyCharm运行与参数

直接运行main_g1_lp_sampling.py，配置config_g1_lp_sampling.json集中管理来源、独立库目录、预设填充3072/4096、控制波长420/540/660nm及阈值。所有路径按入口解析。依赖沿用requirements.txt及requirements_lightpipes.txt，原CPython3.10.4/NumPy2.2.6/SciPy1.15.3/Matplotlib3.10.9/LightPipes2.1.5；LightPipes默认从项目work/dependencies/lightpipes_2_1_5导入。

CPU使用普通本地文件，不需GPU、Notebook、Docker、Harness或会话。4096复数组及FFT临时数组可能占用数GB内存，按单波长/单传播器顺序计算并释放临时数组，不同时执行多个大域任务。--no-save只读复算；--show-plots请求弹窗。默认保存results/g1_lp_sampling/run_*，不覆盖旧结果，图和关键数值自动保存。人工PyCharm点击/GUI和GPU未验证。

## 实验与公式

输入直接复用已审核上一阶段的三个u1复场、固定高度指纹和参考核，不插值或重新设计DOE。Forvard扩大空白域到3072/4096格点，dx仍1µm；复用旧1536/2048数据，报告相邻填充变化以及对参考的误差。1%阈值保留，不根据结果更改。有限两个大域接近不证明无限域收敛。

LightPipes2.1.5 Fresnel内部legacy分支用size/(N−1)。本批将Begin的size明确改为(N−1)×dx，以便内部积分采用原物理dx；仅适配接口，不修改第三方源码。Begin声明的dx变为(N−1)dx/N，不用它作为本批物理轴或功率面积，实际有效积分轴另记录为1µm。同一数值输入放在相同中心索引，按有效积分轴解释，器件不变；这是针对已核对2.1.5实现的适配，不是作者2019参数。

Fresnel还对卷积核作单元积分和相邻输出组合。本批推导并检验其等效每方向核权重：E(x,x′)=1/2∫从−dx到+dx exp[iπ(x−x′+t)²/(λz)]dt。二维输出为Σu1 E_y E_x/(iλz)，全局相位另对齐。用scipy.special.fresnel原函数实现积分；对应宽2dx的中心核平均，不能将其误称为普通宽dx物体单元积分或移动DOE高度。小dx极限E≈dx·exp(iπ位移²/(λz))，回到点采样Fresnel。

分别保存原接口Fresnel、步长适配Fresnel、独立核平均和点采样参考，避免把多个变化合并后只报一个“修好了”。复场只允许单位全局相位对齐；强度及相机核不缩放。原生609方格共用物理坐标，再按上一阶段相同强度线性插值/6.22µm/q8中点积分映射97相机像元。插值经验误差沿用前阶段，尚非严格上界；全域Forvard能量守恒不证明局部无卷绕。

首轮1536适配域在靠外的609节点处仍与独立核平均不一致，结果run_20261006_132838保留。其有限卷积核约覆盖±768µm位移，而输入孔径至输出外侧可能达到约804µm，因此新增fresnel_padding_sizes=[1536,2048]，同时报告两域；2048覆盖约±1024µm，覆盖全部有效输入到该输出域的位移。单元模型验收只针对完整覆盖的2048域，1536差异及首轮失败仍记录，不放宽1e−10门槛。

## 输出与验收

arrays/sampling.npz保留输入、参考、所有相机核及新原生场、独立核平均场和物理轴；metrics/validation.json含填充变化、单元模型检查、步长/接口适配状态及版本；figures/sampling_decomposition.png保存拆分曲线；来源/代码SHA和数组重载检查保留。数值健康、1%算法接近、1e−10单元模型验证分开记录；未通过项目照实保留，paper_alignment_passed=false。

tests/test_g1_lp_sampling.py包含与独立24点Gauss求积的积分核检查、小单元极限和非法物理输入检查。实际运行与独立场/像元核对、无保存完整性结果见outputs/执行_G1LP采样拆分_20261006，最终审核生成前不算阶段审核完成。

来源：项目内LightPipes2.1.5的propagators.py中_field_Fresnel与Forvard，包SHA由前阶段锁定；[官方源码库](https://github.com/opticspy/lightpipes)用于出处，不能将可变化master作为固定版本。正文脚注仅给工具名，未公开作者传播器/版本。整图或无限域收敛、论文图3尺寸对齐和真实实验仍未验证。
