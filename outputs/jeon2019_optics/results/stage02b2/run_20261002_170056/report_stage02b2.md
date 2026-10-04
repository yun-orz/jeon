# 02B-2采样与视场报告

计算状态：completed；科学筛查：default_within_tested_tolerances

36组原始场载荷，18次主传播及Jeon540原生点独立核验。比较容差取自config_effective.json。

同网格比较原始强度；输出加密以双线性强度插值作诊断，未用插值生成器件PSF。

| 比较 | 器件 | λ/nm | L1/% | L2/% | 全部指标达标 |
| --- | --- | --- | --- | --- | --- |
| input_coarse->baseline | fresnel | 420 | 0.2334 | 0.2819 | True |
| input_coarse->baseline | fresnel | 540 | 0.0366 | 0.0148 | True |
| input_coarse->baseline | fresnel | 660 | 0.0836 | 0.0780 | True |
| input_coarse->baseline | jeon | 420 | 0.2274 | 0.1776 | True |
| input_coarse->baseline | jeon | 540 | 0.1272 | 0.0853 | True |
| input_coarse->baseline | jeon | 660 | 0.1673 | 0.1533 | True |
| baseline->input_fine | fresnel | 420 | 0.0569 | 0.0586 | True |
| baseline->input_fine | fresnel | 540 | 0.0141 | 0.0078 | True |
| baseline->input_fine | fresnel | 660 | 0.0261 | 0.0219 | True |
| baseline->input_fine | jeon | 420 | 0.1090 | 0.0941 | True |
| baseline->input_fine | jeon | 540 | 0.0626 | 0.0440 | True |
| baseline->input_fine | jeon | 660 | 0.0783 | 0.0709 | True |
| output_coarse->input_fine | fresnel | 420 | 0.2563 | 0.2918 | True |
| output_coarse->input_fine | fresnel | 540 | 0.4263 | 0.3721 | True |
| output_coarse->input_fine | fresnel | 660 | 0.1031 | 0.1298 | True |
| output_coarse->input_fine | jeon | 420 | 0.3019 | 0.2856 | True |
| output_coarse->input_fine | jeon | 540 | 0.2178 | 0.2216 | True |
| output_coarse->input_fine | jeon | 660 | 0.1252 | 0.1446 | True |
| input_fine->output_fine | fresnel | 420 | 0.0643 | 0.0731 | True |
| input_fine->output_fine | fresnel | 540 | 0.1068 | 0.0932 | True |
| input_fine->output_fine | fresnel | 660 | 0.0258 | 0.0325 | True |
| input_fine->output_fine | jeon | 420 | 0.0757 | 0.0716 | True |
| input_fine->output_fine | jeon | 540 | 0.0546 | 0.0555 | True |
| input_fine->output_fine | jeon | 660 | 0.0314 | 0.0362 | True |

| 器件 | λ/nm | η150 | η300 | 扩窗新增 | R80_150/μm | R80_300/μm |
| --- | --- | --- | --- | --- | --- | --- |
| fresnel | 420 | 0.729070 | 0.872227 | 0.143157 | None | 215.9178084364511 |
| fresnel | 540 | 0.965937 | 0.982685 | 0.016748 | 25.656383221334995 | 25.656383221334995 |
| fresnel | 660 | 0.857728 | 0.909025 | 0.051297 | 120.07497657713701 | 120.07497657713701 |
| jeon | 420 | 0.794913 | 0.894884 | 0.099971 | None | 166.01204775557707 |
| jeon | 540 | 0.888439 | 0.941139 | 0.052700 | 105.65036677645752 | 105.65036677645752 |
| jeon | 660 | 0.725099 | 0.864067 | 0.138968 | None | 221.42549537033895 |

本批最细网格不是解析真值。未进行0.25μm输入加密、制造量化、6.22μm像元积分或重建。
原文方向坐标对应仍待解释。原始I与Pin未经归一化，窗口未捕获量不是制造损耗。
参数/公式来源沿用02A/02B-1，D/f应用于图3、材料与网格为已记录实施假设。