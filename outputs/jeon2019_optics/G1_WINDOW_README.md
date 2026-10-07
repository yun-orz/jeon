# G1固定高度的探测窗口与设计公式补核

本阶段为Jeon光学编码复现的诊断，不是完整论文复现；不修改旧G0/G1、顺时针补核高度或G2–G5模型。

PyCharm选择原CPython3.10.4解释器，直接运行main_g1_window.py。参数集中config_g1_window.json，路径按入口解析。CPU依赖沿用requirements.txt；已安装NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9。无Notebook、Docker、Harness或GPU依赖。--no-save只读复算，--show-plots尝试弹窗；人工PyCharm点击和GUI须另验。

固定正文图3九波长420:30:660nm；孔径1mm、f=z=50mm、像元6.22µm与正文PDF第7页一致。器件使用已审核CW诊断高度，1101×1101输入/1µm间距，所有波长共享同一高度，不重新设计器件。把探测器97×97扩大到161×161，保持像元中心及边界对齐，8×8中点积分为主、4×4参比，共18次真实复振幅Fresnel传播。保留输入功率Pin，K=Ppixel/Pin，和为有限窗口效率；不补偿归一化。绘图峰值归一化仅观察形状。

复场传播为正文式(4)的近轴标量Fresnel模型，强度abs(u2)^2；坐标x右/y上，数组行y列x，长度内部m。NA约0.01，薄透射器件、单色轴上平面波，不含吸收、反射、矢量/加工误差。公共任意输出坐标积分不受FFT原生波长间距变化影响。输出场540nm原始复数与求积节点保存，可独立核对。

正文第4页式(7)–(9)、第5页式(10)–(12)给出δ=sqrt(r²+f²)−f、h=(mλdesign−δ)/(n(λdesign)−1)。本程序检查孔径内[δ+(n−1)h]/λdesign是否整数。m是绕回整数，不是折射率；n(λdesign)采用既有Malitson色散，仍是实施解释。基底常量只改变全局相位，不能改变理想强度。正文第7页明确材料熔石英、16层/100nm及0.5mm基底；图3的连续/量化高度、材料模型、图像处理细节未明确，不能把制造量化自动视为图3设置。

扩大窗口后中心97核应与旧CW核一致，同时统计两套尺寸：R50/R80以输入总功率或窗口功率为分母。窗口分母随捕获效率改变，不能与输入分母混用；径向统计限完整圆域，未达到阈值写null。固定150µm域的峰值阈值面积等效半径、RMS半径也保存，不等同翼尖/FWHM。工程采样容差沿用0.5%相对L1、1%相对L2，不是论文尺寸容差。大窗口未重做细输入网格验证，九波长也不是新25波段重建核，不切换下游算子。

结果保存results/g1_window/run_*：arrays/window_bank.npz含新旧核及物理坐标，control_540nm.npz含复场/像元功率/节点/Pin；metrics/validation.json含尺寸、效率、求积及公式残差，figures两图自动保存，来源及代码SHA留档，不覆盖旧run。实际执行和独立审核见工程外outputs/执行_G1窗口补核_20261006；最终审核生成前不算阶段验收完成。

作者项目页：https://vccimaging.org/Publications/Jeon2019Hyperspectral/ 。本地原论文work/papers/main.pdf、supplement.pdf，来源SHA见work/papers/stage01_check/source_manifest.json。正文图3和补充第4页未提供PSF物理标尺/定量容差，故paper_alignment_passed=false；本阶段只能区分截断窗口与当前解析模型中的变化，不能反推唯一作者实现。
