# G3 HQS结构与梯度验证

Jeon光学编码公式实现＋合成响应＋HQS结构验证。使用PyTorch，原文TensorFlow；无作者权重或真实HSI训练。3阶段、64特征、4层，各阶段先验及参数独立。未公开细节为实施假设。

随机初始化输出及2步合成单patch输出均不是正式重建基线，不能评价论文PSNR/SSIM/SAM。详细中文说明见G3_README.md。
