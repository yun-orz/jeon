"""M2相对能量RGB适配层：零延拓卷积和严格伴随。"""
import numpy as np
from scipy.signal import convolve2d


def overlap_matrix(native_n=97, pitch=6.22e-6):
    """把源像元质量按物理面积重叠分配；目标边缘多出的面积保持空白。"""
    target_n = (native_n + 1) // 2
    source = (np.arange(native_n + 1) - native_n / 2) * pitch
    target = (np.arange(target_n + 1) - target_n / 2) * 2 * pitch
    overlap = np.maximum(0, np.minimum(target[1:, None], source[None, 1:])
                         - np.maximum(target[:-1, None], source[None, :-1]))
    matrix = overlap / pitch
    empty_area = (2 * pitch)**2 - np.outer(overlap.sum(1), overlap.sum(1))
    return matrix, source, target, empty_area


def remap(kernels, matrix):
    return np.stack([matrix @ k @ matrix.T for k in kernels])


class EnergyRGB:
    """输入[N,25,H,W]为相对谱密度/nm；输出RGB只计一次10nm权重。

    物理模式不应用训练增益。最终模式要求M3提供共同拟合增益，pending时拒绝。
    """
    def __init__(self, kernels, response, weights, measurement_gain=None, final=False):
        self.k = np.asarray(kernels, dtype=np.float64)
        self.coeff = np.asarray(response, dtype=np.float64) * np.asarray(weights)[None, :]
        if self.k.ndim != 3 or self.k.shape[0] != 25 or self.coeff.shape != (3, 25):
            raise ValueError("核或响应形状不符合25波段协议")
        if not np.isfinite(self.k).all() or not np.isfinite(self.coeff).all():
            raise ValueError("核与响应必须有限")
        if min(self.k.shape[1:]) < 1 or any(n % 2 != 1 for n in self.k.shape[1:]):
            raise ValueError("卷积核必须为奇数尺寸")
        if final:
            if measurement_gain is None or not np.isfinite(measurement_gain) or measurement_gain <= 0:
                raise ValueError("共同训练增益pending，不能调用最终训练算子")
            self.coeff = self.coeff * measurement_gain

    def numpy_forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 4 or x.shape[1] != 25:
            raise ValueError("输入应为[N,25,H,W]")
        blurred = np.array([[convolve2d(b, k, mode="same") for b, k in zip(scene, self.k)] for scene in x])
        return np.einsum("cl,nlhw->nchw", self.coeff, blurred)

    def numpy_adjoint(self, y):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 4 or y.shape[1] != 3:
            raise ValueError("RGB应为[N,3,H,W]")
        mixed = np.einsum("cl,nchw->nlhw", self.coeff, y)
        return np.array([[convolve2d(b, k[::-1, ::-1], mode="same")
                          for b, k in zip(scene, self.k)] for scene in mixed])

    def torch_forward(self, x):
        import torch
        from torch.nn.functional import conv2d
        k = torch.as_tensor(self.k[:, ::-1, ::-1].copy(), dtype=x.dtype, device=x.device)[:, None]
        c = torch.as_tensor(self.coeff, dtype=x.dtype, device=x.device)
        blurred = conv2d(x, k, padding=(k.shape[-2]//2, k.shape[-1]//2), groups=25)
        return torch.einsum("cl,nlhw->nchw", c, blurred)

    def torch_adjoint(self, y):
        import torch
        from torch.nn.functional import conv_transpose2d
        k = torch.as_tensor(self.k[:, ::-1, ::-1].copy(), dtype=y.dtype, device=y.device)[:, None]
        c = torch.as_tensor(self.coeff, dtype=y.dtype, device=y.device)
        mixed = torch.einsum("cl,nchw->nlhw", c, y)
        return conv_transpose2d(mixed, k, padding=(k.shape[-2]//2, k.shape[-1]//2), groups=25)


def numerical_audit(kernels, response, weights):
    """用独立直接卷积、内积和自动微分核对物理算子，包含非对称真实核。"""
    import torch
    torch.set_num_threads(2)
    op = EnergyRGB(kernels, response, weights)
    rng = np.random.default_rng(20261007)
    x = rng.normal(size=(1,25,13,15)); y = rng.normal(size=(1,3,13,15))
    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(y, dtype=torch.float64)
    forward = op.numpy_forward(x); adjoint = op.numpy_adjoint(y)
    tf = op.torch_forward(tx); ta = op.torch_adjoint(ty)
    grad = torch.autograd.grad((tf*ty).sum(), tx)[0].detach().numpy()
    def rel(a,b):
        return float(np.linalg.norm(a-b) / max(np.linalg.norm(b), np.finfo(float).tiny))
    lhs = float(np.sum(forward*y)); rhs = float(np.sum(x*adjoint))
    rows = {"forward_relative": rel(tf.detach().numpy(),forward),
            "adjoint_relative": rel(ta.numpy(),adjoint), "autograd_relative": rel(grad,adjoint),
            "inner_product_relative": abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-300)}
    # 中央差分验证标量损失方向导数，独立于伴随实现。
    direction = rng.normal(size=x.shape); eps = 1e-4
    fd = float(np.sum((op.numpy_forward(x+eps*direction)-op.numpy_forward(x-eps*direction))*y)/(2*eps))
    analytic = float(np.sum(grad*direction))
    rows["directional_derivative_relative"] = abs(fd-analytic)/max(abs(fd),abs(analytic),1e-300)
    rows["passed"] = all(v <= 1e-10 for v in rows.values())
    return rows


def mapping_audit(kernels, mapped, matrix, pitch):
    """独立像元矩形遍历以及点质量、对称核、质心案例。"""
    x = (np.arange(97)-48)*pitch; t = (np.arange(49)-24)*2*pitch
    independent = np.zeros_like(matrix)
    for j, center_t in enumerate(t):
        for i, center_s in enumerate(x):
            independent[j,i] = max(0, min(center_t+pitch,center_s+pitch/2)
                                   - max(center_t-pitch,center_s-pitch/2))/pitch
    mass_error = float(np.max(np.abs(kernels.sum((1,2))-mapped.sum((1,2)))))
    cases = []
    for iy,ix in [(48,48),(48,49),(0,0),(96,96)]:
        point = np.zeros((97,97)); point[iy,ix] = 1
        target = matrix @ point @ matrix.T
        cases.append({"source_index_yx":[iy,ix], "mass_error":abs(float(target.sum())-1),
                      "centroid_error_m":max(abs(float(target.sum(0)@t)-x[ix]),
                                              abs(float(target.sum(1)@t)-x[iy]))})
    symmetric = np.exp(-(x[:,None]**2+x[None,:]**2)/(2*(20*pitch)**2))
    sym = matrix @ symmetric @ matrix.T
    rows = {"matrix_independent_max_error":float(np.max(np.abs(matrix-independent))),
            "column_mass_max_error":float(np.max(np.abs(matrix.sum(0)-1))),
            "native_to_half_mass_max_error":mass_error,
            "symmetry_relative_error":float(np.linalg.norm(sym-sym[::-1,::-1])/np.linalg.norm(sym)),
            "symmetric_centroid_m":float(max(abs(sym.sum(0)@t),abs(sym.sum(1)@t))/sym.sum()),
            "point_cases":cases}
    # 面积映射保持质量和几何原点，任意离散质心允许最多半个源像元的量化偏差。
    native_centers = np.column_stack([kernels.sum(1)@x,kernels.sum(2)@x])/kernels.sum((1,2))[:,None]
    half_centers = np.column_stack([mapped.sum(1)@t,mapped.sum(2)@t])/mapped.sum((1,2))[:,None]
    rows["actual_centroid_shift_m"] = np.max(abs(native_centers-half_centers),axis=1).tolist()
    rows["passed"] = bool(mass_error < 1e-12 and rows["matrix_independent_max_error"]<1e-12
                          and rows["column_mass_max_error"]<1e-12 and rows["symmetry_relative_error"]<1e-12
                          and max(rows["actual_centroid_shift_m"]) <= pitch/2+1e-15
                          and all(c["mass_error"]<1e-12 and c["centroid_error_m"]<=pitch/2+1e-15 for c in cases))
    return rows
