# 阶段02D-1实际运行报告

本批验证单通道非相干成像及伴随；尚未执行基础重建。
验证状态：True；源run只读且三高度/九组q8场重验通过。

模型：y=CΣ_b R_b(X_b*K_b)，K_b=Ppixel_b/Pin_b；非相干按功率相加。
full零延拓线性卷积，伴随先裁剪零回填再与翻转核作valid卷积。

|器件|场景|full通量|crop保留比例|通量相对误差|直接叠加相对误差|
|---|---|---:|---:|---:|---:|
|continuous|coincident_points|3.2104|0.999144|0|3.24e-16|
|continuous|separated_points|3.2104|0.984356|0|4.04e-16|
|continuous|lines_and_square|141.903|0.994022|0|4.89e-16|
|nearest_depth|coincident_points|3.12306|0.999144|0|3.2e-16|
|nearest_depth|separated_points|3.12306|0.984395|1.42e-16|4.02e-16|
|nearest_depth|lines_and_square|138.378|0.994039|2.05e-16|5.6e-16|

## 假设与未验证项

- 理想像平面点源功率格点，平移不变卷积；无物距/放大率或离轴像差模型。
- 三离散单色诊断波长；不是宽波段积分或每nm谱密度。
- PSF核=像元功率/Pin，不重复乘像元面积、不将通量强制归一到1。
- 满填充、单位辐射响应；无CFA/QE、光子电子换算或噪声。
- full保留所选有限PSF核的通量；核外衍射尾部仍截断。
- FFT机器舍入级负数不剪裁，以保持线性/伴随；它们不是物理负功率。
- 正文旋转方向对应仍未确定；未镜像PSF。真实PyCharm点击/图窗尚未人工验证。

详细内积/梯度检查和环境见metrics/validation.json。原始数组在arrays，结果图在figures。
