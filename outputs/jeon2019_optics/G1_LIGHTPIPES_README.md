# 同一固定DOE的LightPipes交叉核对

本阶段属于Jeon光学编码复现的诊断，不是作者2019原脚本或完整论文复现。正文第5页脚注只给LightPipes工具名，没有版本及传播器调用，本批使用现代Python LightPipes2.1.5，不能将结果直接归因于作者设置。

## PyCharm运行

选择已有CPython3.10.4解释器，直接运行main_g1_lightpipes.py。参数集中config_g1_lightpipes.json，所有路径按入口位置解析。LightPipes独立安装在项目work/dependencies/lightpipes_2_1_5，入口按配置加入普通本地Python导入路径，不依赖聊天/Agent/Harness。无需Notebook、Docker或GPU。

实际环境NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9，新增LightPipes2.1.5。原环境缺此包，pip联网TLS失败后，使用系统HTTPS从官方PyPI取得wheel并核对官方SHA，再以--no-index --no-deps --no-compile安装至专用目录，旧环境依赖版本未替换。下载wheel及来源SHA保留work/dependencies/downloads。迁移到另一台电脑，可保留相对目录，或在解释器终端运行python -m pip install -r requirements_lightpipes.txt；也可python -m pip install --no-deps --target ../../work/dependencies/lightpipes_2_1_5 LightPipes==2.1.5。原基础依赖需先安装。

默认CPU保存PNG、NPZ及JSON，不弹窗；--no-save只读重算，--show-plots请求弹窗。结果在results/g1_lightpipes/run_*，重复运行不覆盖旧结果。人工PyCharm点击和GUI须另外验证。

## 固定器件与公共物理坐标

直接读取已审核CW诊断高度、掩码及局部设计波长；指纹逐字节重现后再计算420/540/660nm透过复场。无旋转/镜像图片，没有每波长重设计器件。原1101方格、1µm间距完整孔径D1mm、z=f50mm。把同一复场放入1536/2048的中心，外部补零；保持中心索引和物理步长，不插值高度或改孔径。

LightPipes Begin声明dx=size/N，与本工程索引中心约定数值一致。其行坐标默认显示向下，本实验明确映射为本项目y递增并用origin=lower；直接传入数值阵列，不能借显示约定差异推断作者旋转方向。Forvard是周期FFT的近轴传播，两种零填充用来诊断周期卷绕。Fresnel是卷积积分形式，但2.1.5源码内部legacy=True时将dx改成size/(N−1)，与Begin声明值不同；本实验如实记录两者，不悄悄改源码补偿。

参考为当前可分离Fresnel积分在−304:1:304µm的609×609原生节点值。各库输出都在这些相同整数µm节点上取中心场；原始强度差不做任何幅度缩放。复场仅另报最佳单位全局相位对齐后的差异，不能把这项指标与未缩放强度差混淆。Forvard/Fresnel源码使用近似π常数的全局相位约定，可能影响复场直接差而不影响强度。

为比较已审核97×97相机核，将原生1µm网格强度线性插值到公共6.22µm像元的q8中点，再积分并除Pin。插值不生成波长旋转；其误差本批未单独界定，因此探测器差异同时包含传播器离散差和插值差。原生节点强度比较可帮助区分两者。核和为窗口效率，不强制归一到1；图形峰值归一只用于形状观察。

## 审核与解释

预设相对L1/L2均1%仅是工程交叉比较阈值，不是论文标准；每个传播器/填充尺寸分别记录结果，不为通过更换阈值。数值健康（有限、非负、合理能量）与算法接近程度分别报告，周期传播保持全网格功率并不证明局部PSF正确或无卷绕。

arrays/crosscheck.npz保留三波长输入复场、参考场、各库原生中心场/相机核，metrics/validation.json保留误差/效率/间距及版本，source_manifest/evidence保留源码、库、来源SHA。逐元素保存重载核对；实际结果、独立FFT审核和无保存完整性记录见outputs/执行_G1LightPipes_20261006，最终审核生成前不算审核完成。不切换下游G2–G5核，paper_alignment_passed=false。

官方来源：[LightPipes命令文档](https://opticspy.github.io/lightpipes/command-reference.html)、[官方源码库](https://github.com/opticspy/lightpipes)。本批实测以项目内2.1.5安装源码为准；官方master随时间变化，不当作固定版本快照。
