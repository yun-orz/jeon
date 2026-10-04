# ======================================================================================
# D. 三类彼此不可替代的检查：输入采样收敛 / 输出采样（填充）收敛 / 离散算法恒等
# ======================================================================================
def _intensity_l1_common(a: np.ndarray, b: np.ndarray, dA: float) -> Dict[str, float]:
    """同一物理网格上两张**未归一化**强度的比较（第二轮修正定义的三种量）。

    ``raw_intensity_L1``  : ``Σ|Ia−Ib|ΔA / Σ Ib ΔA`` —— 主收敛指标，保留绝对通量信息。
    ``shape_L1``          : ``Σ|Ia/Pa − Ib/Pb|ΔA`` —— 只比较形状，``Pa/Pb`` 是各自
                            在同一观察窗内的强度积分；**不用于**主验收。
    ``field_relL2``       : ``sqrt(Σ|ua−ub|²ΔA / Σ|ub|²ΔA)`` —— 复场相对 L2，
                            单独存放，**不得**塞进名为 L1 的字段。
    """
    pa = float(np.sum(a) * dA)
    pb = float(np.sum(b) * dA)
    raw = float(np.sum(np.abs(a - b)) * dA / pb) if pb > 0 else float("nan")
    if pa > 0 and pb > 0:
        shape = float(np.sum(np.abs(a / pa - b / pb)) * dA)
    else:
        shape = float("nan")
    return {"raw_intensity_L1": raw, "shape_L1": shape, "P_a": pa, "P_b": pb}


def study_sampling(cfg: Dict[str, Any], run_dir: Optional[Path], logger, show: bool
                   ) -> Dict[str, Any]:
    """D. 把「离散算法恒等」「输入积分收敛」「输出采样/填充收敛」当作**不同指标**分别验收。

    * **D1 输入采样收敛**：固定同一个连续器件（理想圆孔 + 理想薄透镜）、固定公共输出
      坐标，只改变输入采样间距。每级都按**同一个连续函数**取样，不把粗网格图像放大。
    * **D2 输出采样收敛**：固定同一个离散输入，只改变 FFT 长度 M，把原生**强度**
      映射到同一个公共探测器网格，并与该输入在公共坐标上的完整核参考比较。
    * **D3 离散算法恒等**：同一离散输入、同一原生网格上 FFT 与完整核求和必须相等。
      它**不能**代替 D1/D2 的物理收敛。

    三种量的定义不同，阈值也不同，报告与 JSON 里分开存放。
    """
    s = cfg["studies"]["sampling"]
    lam = U.nm(s["wavelength_nm"])
    z = U.mm(s["propagation_distance_mm"])
    D = U.mm(s["aperture_diameter_mm"])
    W = U.um(s["input_half_width_um"])
    r1 = M.airy_dark_ring_radii(lam, z, D, 1)[0]
    tol_in = cfg["acceptance"]["input_sampling_intensity_L1_max"]
    tol_pad = cfg["acceptance"]["padding_sampling_intensity_L1_max"]
    tol_id = cfg["acceptance"]["discrete_identity_rel_l2_max"]

    # 公共输出坐标（所有输入等级与所有 M 都映射到这一套坐标）
    half_out = float(s["common_output_half_width_over_r1"]) * r1
    out_dx = U.um(s["common_output_spacing_um"])
    k_out = 2 * int(np.ceil(half_out / out_dx)) + 1
    x_common = (np.arange(k_out, dtype=np.float64) - k_out // 2) * out_dx
    g_common = Grid2D(Axis1D(k_out, out_dx, "x"), Axis1D(k_out, out_dx, "y"),
                      label="common detector grid")
    dA = g_common.cell_area
    common = {"x_m": x_common, "y_m": x_common, "spacing_m": out_dx, "points": k_out,
              "half_width_m": half_out}
    logger.info("【D】公共输出网格：%d×%d，间距 %.4g μm，半宽 %.4g μm（= %.2f·r1，r1=%.4g μm）",
                k_out, k_out, U.to_um(out_dx), U.to_um(half_out), s["common_output_half_width_over_r1"],
                U.to_um(r1))

    figs: List[Path] = []
    rows_in: List[List[Any]] = []
    rows_pad: List[List[Any]] = []
    rows_id: List[List[Any]] = []
    saved: Dict[str, Any] = {"x_common_m": x_common}

    def build_level(dx_in: float) -> Dict[str, Any]:
        """按**同一个连续器件函数**在给定间距上取样。"""
        n = 2 * int(np.ceil(W / dx_in)) + 1        # 奇数，覆盖同一边界
        g = Grid2D(Axis1D(n, dx_in, "x'"), Axis1D(n, dx_in, "y'"))
        u1, ap, lens = make_lens_aperture_field(D, z, lam, g)
        p_in = float(np.sum(np.abs(u1) ** 2) * g.cell_area)
        ap_area = float(np.sum(ap) * g.cell_area)
        # 公共坐标上用完整位移核的**可分离**等价形式求值
        u2 = fresnel_kernel_separable(u1, g, lam, z, x_common, x_common)
        I = np.abs(u2) ** 2
        # 该级与解析 Airy 的中心截线误差（沿用与 A 组相同的口径）
        line = I[k_out // 2, :] / max(float(I[k_out // 2, :].max()), 1e-300)
        airy = M.airy_intensity(np.abs(x_common), lam, z, D)
        line_l1 = float(np.sum(np.abs(line - airy)) / np.sum(airy))
        return {"n": n, "dx_m": dx_in, "grid": g, "u1": u1, "ap": ap, "lens": lens,
                "p_in": p_in, "ap_area": ap_area, "u2": u2, "I": I,
                "line_L1_vs_airy": line_l1}

    # ---------------- D1: 输入采样收敛 ----------------
    levels = [build_level(U.um(lv["input_spacing_um"])) for lv in s["input_levels"]]
    for lv, d in zip(s["input_levels"], levels):
        logger.info("【D1】输入采样 %s：N=%d，Δx′=%.4g μm，P_in=%.10e，孔径离散面积=%.6e "
                    "（理想 πD²/4=%.6e，相对差 %+.3e），中心截线 L1=%.4e",
                    lv["name"], d["n"], U.to_um(d["dx_m"]), d["p_in"], d["ap_area"],
                    np.pi * (0.5 * D) ** 2,
                    d["ap_area"] / (np.pi * (0.5 * D) ** 2) - 1.0, d["line_L1_vs_airy"])

    level_metrics = []
    for i in range(len(levels) - 1):
        a, b = levels[i], levels[i + 1]
        cmp_ib = _intensity_l1_common(a["I"], b["I"], dA)
        cmp_ba = _intensity_l1_common(b["I"], a["I"], dA)
        f_l2 = float(np.sqrt(np.sum(np.abs(a["u2"] - b["u2"]) ** 2) * dA
                             / (np.sum(np.abs(b["u2"]) ** 2) * dA)))
        m = {"pair": "%s->%s" % (s["input_levels"][i]["name"], s["input_levels"][i + 1]["name"]),
             "left": s["input_levels"][i]["name"], "right": s["input_levels"][i + 1]["name"],
             "raw_intensity_L1": cmp_ib["raw_intensity_L1"],
             "raw_intensity_L1_reverse": cmp_ba["raw_intensity_L1"],
             "shape_L1": cmp_ib["shape_L1"],
             "field_relL2": f_l2,
             "P_left_window": cmp_ib["P_a"], "P_right_window": cmp_ib["P_b"]}
        level_metrics.append(m)
        rows_in.append([m["pair"], gnum(levels[i]["n"]), gnum(levels[i + 1]["n"]),
                        gnum(U.to_um(a["dx_m"])), gnum(U.to_um(b["dx_m"])),
                        gnum(m["raw_intensity_L1"]), gnum(m["shape_L1"]),
                        gnum(m["field_relL2"]),
                        gnum(levels[i]["line_L1_vs_airy"]),
                        gnum(levels[i + 1]["line_L1_vs_airy"])])
        logger.info("【D1】%s：原始强度相对 L1=%.6e（反向 %.6e），形状 L1=%.6e，"
                    "复场相对 L2=%.6e（目标 ≤%.3g）", m["pair"], m["raw_intensity_L1"],
                    m["raw_intensity_L1_reverse"], m["shape_L1"], m["field_relL2"], tol_in)

    # 预设判据：用**最后两级**判定最终精度；粗级不满足就如实标出，不删除
    input_last_pair = level_metrics[-1] if level_metrics else None
    worst_input_L1 = max((m["raw_intensity_L1"] for m in level_metrics), default=float("nan"))
    for lv, d in zip(s["input_levels"], levels):
        rows_in.append(["level:%s" % lv["name"], gnum(d["n"]), "", gnum(U.to_um(d["dx_m"])), "",
                        "", "", "", gnum(d["line_L1_vs_airy"]),
                        gnum(d["ap_area"] / (np.pi * (0.5 * D) ** 2) - 1.0)])
        saved["u_%s" % lv["name"]] = d["u2"]
        saved["I_%s" % lv["name"]] = d["I"]
        saved["ap_%s" % lv["name"]] = d["ap"]
        saved["u1_%s" % lv["name"]] = d["u1"]

    # ---------------- D2: 输出采样（填充）收敛 ----------------
    dx_pad = U.um(s["padding_input_spacing_um"])
    n_pad = 2 * int(np.ceil(W / dx_pad)) + 1
    g_pad = Grid2D(Axis1D(n_pad, dx_pad, "x'"), Axis1D(n_pad, dx_pad, "y'"))
    u1_pad, _, _ = make_lens_aperture_field(D, z, lam, g_pad)
    p_in_pad = float(np.sum(np.abs(u1_pad) ** 2) * g_pad.cell_area)
    # 公共坐标上的完整核参考（不插值复相位，只在公共坐标直接求值）
    u_ref = fresnel_kernel_separable(u1_pad, g_pad, lam, z, x_common, x_common)
    I_ref = np.abs(u_ref) ** 2
    logger.info("【D2】固定输入：N=%d，Δx′=%.4g μm，P_in=%.10e；公共坐标参考由完整核"
                "（可分离等价形式）直接求值，不插值复相位", n_pad, U.to_um(dx_pad), p_in_pad)

    pad_records = []
    for lv in s["padding_levels"]:
        m_fft = int(lv["fft_size"])
        with Timer() as t:
            r = fresnel_fft(u1_pad, g_pad, FresnelConfig(lam, z, m_fft))
        # 3) 原生网格上的离散恒等（归入算法检查）
        ref_native = fresnel_kernel_matrix(u1_pad, g_pad, lam, z,
                                           r.grid.x.coords, r.grid.y.coords)
        ident = rel_l2(r.field, ref_native)
        p_in = float(np.sum(np.abs(u1_pad) ** 2) * g_pad.cell_area)
        p_err = abs(r.power / p_in - 1.0)
        # 4) 原生**强度**双线性插值到公共探测器网格（不插值高速变化的复相位）
        I_native = np.abs(r.field) ** 2
        I_map = M.resample_intensity_bilinear(I_native, r.grid.x.coords, r.grid.y.coords,
                                              x_common, x_common)
        # 越界点必须显式排除，不能补零参与主比较
        inb = (np.abs(x_common) <= np.abs(r.grid.x.coords).max()) & \
              (np.abs(x_common) <= np.abs(r.grid.y.coords).max())
        mask = inb[None, :] & inb[:, None]
        cmp_ref = _intensity_l1_common(np.where(mask, I_map, 0.0),
                                       np.where(mask, I_ref, 0.0), dA)
        # 通量偏差：公共窗口内的映射强度积分对参考积分
        flux = cmp_ref["P_a"] / cmp_ref["P_b"] - 1.0 if cmp_ref["P_b"] > 0 else float("nan")
        rec = {"name": lv["name"], "M": m_fft,
               "native_output_points": int(r.grid.x.n),
               "native_output_spacing_m": float(r.grid.dx),
               "native_output_spacing_formula_m": float(lam * z / (m_fft * dx_pad)),
               "identity_rel_L2": ident, "power_rel_err": p_err,
               "common_ref_intensity_L1": cmp_ref["raw_intensity_L1"],
               "common_shape_L1": cmp_ref["shape_L1"],
               "common_window_flux_bias": flux,
               "common_points_used": int(mask.sum()),
               "common_points_excluded": int(mask.size - mask.sum()),
               "wall_time_s": t.elapsed,
               "window_covered": bool(mask.all()),
               "I_map": I_map, "I_native": I_native, "mask": mask}
        pad_records.append(rec)
        logger.info("【D2】%s：M=%d，原生输出 %d 点，原生间距 %.4g μm（公式 %.4g μm，一致=%s）；"
                    "离散恒等 rel L2=%.4e；功率误差=%.4e；公共网格原始强度 L1=%.6e"
                    "（目标 ≤%.3g）；形状 L1=%.4e；窗口通量偏差 %+.3e；剔除越界点 %d",
                    lv["name"], m_fft, r.grid.x.n, U.to_um(r.grid.dx),
                    U.to_um(rec["native_output_spacing_formula_m"]),
                    abs(r.grid.dx - rec["native_output_spacing_formula_m"]) <= 1e-9 * r.grid.dx,
                    ident, p_err, cmp_ref["raw_intensity_L1"], tol_pad, cmp_ref["shape_L1"],
                    flux, rec["common_points_excluded"])
        rows_pad.append([lv["name"], gnum(m_fft), gnum(r.grid.x.n),
                         gnum(U.to_um(r.grid.dx)), gnum(ident), gnum(p_err),
                         gnum(cmp_ref["raw_intensity_L1"]), gnum(cmp_ref["shape_L1"]),
                         gnum(flux), gnum(t.elapsed)])
        saved["I_native_%s" % lv["name"]] = I_native
        saved["x_native_%s" % lv["name"]] = r.grid.x.coords
        saved["I_map_%s" % lv["name"]] = I_map

    # 两个 M 在公共网格上的实际 L1
    pad_pair = None
    if len(pad_records) >= 2:
        a_rec, b_rec = pad_records[0], pad_records[-1]
        both = a_rec["mask"] & b_rec["mask"]
        cmp_two = _intensity_l1_common(np.where(both, a_rec["I_map"], 0.0),
                                       np.where(both, b_rec["I_map"], 0.0), dA)
        pad_pair = {"a": a_rec["name"], "b": b_rec["name"],
                    "M_a": a_rec["M"], "M_b": b_rec["M"],
                    "raw_intensity_L1_between_M": cmp_two["raw_intensity_L1"],
                    "shape_L1_between_M": cmp_two["shape_L1"],
                    "points": int(both.sum())}
        logger.info("【D2】%s vs %s：公共网格上两者的原始强度 L1=%.6e，形状 L1=%.4e（%d 点）",
                    a_rec["name"], b_rec["name"], cmp_two["raw_intensity_L1"],
                    cmp_two["shape_L1"], pad_pair["points"])
        # 嵌套原生坐标上的相同样点一致性（可作一致性检查，但不能证明分辨率收敛）
        common_pts = np.intersect1d(np.round(a_rec.get("x_native", x_common), 12),
                                    np.round(b_rec.get("x_native", x_common), 12))

    saved.update({"u_reference_common": u_ref, "I_reference_common": I_ref,
                  "u1_padding_input": u1_pad,
                  "x_padding_input_m": g_pad.x.coords, "y_padding_input_m": g_pad.y.coords})

    # ---------------- D3: 离散算法恒等（与物理收敛分开） ----------------
    worst_identity = 0.0
    for case in s.get("identity_cases", []):
        n_i = int(case["input_samples"])
        dx_i = U.um(case["input_spacing_um"])
        m_i = int(case["fft_size"])
        g_i = Grid2D(Axis1D(n_i, dx_i, "x'"), Axis1D(n_i, dx_i, "y'"))
        u_i = make_asymmetric_input(g_i, lam)
        p_in_i = float(np.sum(np.abs(u_i) ** 2) * g_i.cell_area)
        r_i = fresnel_fft(u_i, g_i, FresnelConfig(lam, z, m_i))
        ref_i = fresnel_kernel_matrix(u_i, g_i, lam, z, r_i.grid.x.coords, r_i.grid.y.coords)
        e_i = rel_l2(r_i.field, ref_i)
        pwr_i = abs(r_i.power / p_in_i - 1.0)
        worst_identity = max(worst_identity, e_i)
        logger.info("【D3】%s：N=%d，Δx′=%.4g μm，M=%d → FFT vs 完整核 rel L2=%.4e；"
                    "功率误差=%.4e", case["name"], n_i, U.to_um(dx_i), m_i, e_i, pwr_i)
        rows_id.append([case["name"], gnum(n_i), gnum(U.to_um(dx_i)), gnum(m_i),
                        gnum(e_i), gnum(pwr_i)])
        saved["u1_%s" % case["name"]] = u_i
        saved["x_%s" % case["name"]] = r_i.grid.x.coords

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5))
    for lv, d in zip(s["input_levels"], levels):
        axes[0].plot(U.to_um(x_common), d["I"][k_out // 2, :] / d["I"].max(),
                     lw=1.0, label="%s（Δx′=%.3g μm）" % (lv["name"], U.to_um(d["dx_m"])))
    axes[0].plot(U.to_um(x_common),
                 M.airy_intensity(np.abs(x_common), lam, z, D), "k--", lw=1.0,
                 label="解析 Airy")
    axes[0].set_xlabel("x (μm)")
    axes[0].set_ylabel("归一化强度")
    axes[0].set_title("D1 三个输入采样等级的中心截线\n（公共输出坐标，各自归一化）",
                      fontsize=9.5)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    names = [lv["name"] for lv in s["input_levels"]]
    axes[1].semilogy(range(len(levels)), [d["line_L1_vs_airy"] for d in levels], "o-",
                     label="截线对解析 Airy 的 L1")
    if level_metrics:
        axes[1].semilogy(range(1, len(levels)),
                         [m["raw_intensity_L1"] for m in level_metrics], "s--",
                         label="相邻等级原始强度 L1")
    axes[1].axhline(tol_in, color="r", ls=":", label="阈值 %.3g" % tol_in)
    axes[1].set_xticks(range(len(levels)))
    axes[1].set_xticklabels(names)
    axes[1].set_xlabel("输入采样等级")
    axes[1].set_ylabel("误差")
    axes[1].set_title("D1 输入采样误差趋势", fontsize=9.5)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, which="both")

    for rec in pad_records:
        axes[2].plot(U.to_um(x_common), rec["I_map"][k_out // 2, :] / rec["I_map"].max(),
                     lw=1.0, label="%s（M=%d，Δ=%.3g μm）"
                     % (rec["name"], rec["M"], U.to_um(rec["native_output_spacing_m"])))
    axes[2].plot(U.to_um(x_common), I_ref[k_out // 2, :] / I_ref.max(), "k--", lw=1.1,
                 label="公共坐标完整核参考")
    axes[2].set_xlabel("x (μm)")
    axes[2].set_ylabel("归一化强度")
    axes[2].set_title("D2 两个填充输出的公共网格中心截线", fontsize=9.5)
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    p = save_fig(fig, run_dir, "D_sampling_convergence.png", cfg["plots"]["dpi"], show)
    if p is not None:
        figs.append(p)

    if pad_records:
        ref_rec = pad_records[-1]
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5))
        imshow_grid(axes[0], ref_rec["I_map"], g_common, cfg["plots"]["colormap_intensity"],
                    "D2 %s 映射到公共网格的强度" % ref_rec["name"], cbar_label="a.u.")
        imshow_grid(axes[1], I_ref, g_common, cfg["plots"]["colormap_intensity"],
                    "D2 公共坐标完整核参考", cbar_label="a.u.")
        diff = np.where(ref_rec["mask"],
                        ref_rec["I_map"] / max(ref_rec["I_map"].max(), 1e-300)
                        - I_ref / max(I_ref.max(), 1e-300), np.nan)
        imshow_grid(axes[2], diff, g_common, cfg["plots"]["colormap_error"],
                    "D2 归一化强度差（L1=%.3e）" % ref_rec["common_ref_intensity_L1"],
                    cbar_label="差")
        fig.tight_layout()
        p = save_fig(fig, run_dir, "D_padding_common_grid.png", cfg["plots"]["dpi"], show)
        if p is not None:
            figs.append(p)

    # ---------------- 输出清单 ----------------
    write_csv_maybe(run_dir, "sampling_input_levels.csv",
                    ["比较或等级", "N_left", "N_right", "dx_left_um", "dx_right_um",
                     "原始强度L1", "形状L1", "复场相对L2", "左级截线L1_或孔径面积相对差",
                     "右级截线L1_或空"], rows_in)
    write_csv_maybe(run_dir, "sampling_padding_levels.csv",
                    ["等级", "M", "原生输出点数", "原生输出间距_um", "离散恒等relL2",
                     "功率相对误差", "公共参考原始强度L1", "公共形状L1", "公共窗口通量偏差",
                     "耗时_s"], rows_pad)
    write_csv_maybe(run_dir, "sampling_identity_levels.csv",
                    ["算例", "N", "dx_um", "M", "离散恒等relL2", "功率相对误差"], rows_id)
    save_npz(run_dir, "sampling_common_grid.npz", **saved)

    in_last_ok = bool(input_last_pair and input_last_pair["raw_intensity_L1"] <= tol_in)
    pad_ok = bool(pad_records and min(r["common_ref_intensity_L1"] for r in pad_records) <= tol_pad)
    pad_trend_ok = True
    if len(pad_records) >= 2:
        pad_trend_ok = bool(pad_records[-1]["common_ref_intensity_L1"]
                            <= pad_records[0]["common_ref_intensity_L1"] * 1.5 + 1e-12)
    res = {
        "input_levels": [{"name": lv["name"], "n": d["n"], "dx_m": d["dx_m"],
                          "dx_um": U.to_um(d["dx_m"]), "P_in": d["p_in"],
                          "aperture_area_discrete_m2": d["ap_area"],
                          "aperture_area_ideal_m2": float(np.pi * (0.5 * D) ** 2),
                          "aperture_area_rel_err": d["ap_area"] / (np.pi * (0.5 * D) ** 2) - 1.0,
                          "line_L1_vs_airy": d["line_L1_vs_airy"]}
                         for lv, d in zip(s["input_levels"], levels)],
        "input_pair_metrics": level_metrics,
        "input_sampling_L1": {"last_pair": input_last_pair["pair"] if input_last_pair else None,
                              "last_pair_raw_intensity_L1":
                                  input_last_pair["raw_intensity_L1"] if input_last_pair else None,
                              "worst_pair_raw_intensity_L1": worst_input_L1,
                              "tolerance": tol_in, "pass": in_last_ok,
                              "definition": "raw_intensity_L1(a,b)=Σ|Ia−Ib|ΔA/ΣIbΔA，公共输出坐标"},
        "padding_levels": [{k: v for k, v in r.items()
                            if k not in ("I_map", "I_native", "mask")} for r in pad_records],
        "padding_pair": pad_pair,
        "padding_sampling_L1": {
            "measure": ("公共探测器网格上，各 M 的映射强度对公共坐标完整核参考的"
                        "原始强度 L1"),
            "per_level": {r["name"]: r["common_ref_intensity_L1"] for r in pad_records},
            "best": (min((r["common_ref_intensity_L1"] for r in pad_records), default=None)),
            "tolerance": tol_pad, "pass": pad_ok,
            "finer_not_worse": pad_trend_ok,
            "definition": "raw_intensity_L1(a,b)=Σ|Ia−Ib|ΔA/ΣIbΔA，公共探测器坐标"},
        "identity": {"worst_rel_L2": worst_identity, "tolerance": tol_id,
                     "pass": bool(worst_identity <= tol_id)},
        "common_output_grid": common,
        "metrics_are_distinct": ("input_sampling_L1（输入收敛）、padding_sampling_L1（输出/填充"
                                "收敛）、identity（离散算法恒等）是三种不同指标，分开存放。"),
        "图": [str(p.relative_to(run_dir)) if run_dir else str(p) for p in figs],
    }
    return res
