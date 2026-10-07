# -*- coding: utf-8 -*-
"""02B-2：CPU输入/输出采样与视场检查，PyCharm直接运行入口。"""
import argparse
import csv
import gc
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import main_stage02b as B
from optics.coordinates import make_grid
from optics.convergence import common_indices, compare_intensities, compare_metrics
from optics.psf_analysis import psf_metrics, absolute_encircled_energy
from optics.runutil import environment_info
from optics.stage02_runtime import configure_plotting, effective_runtime, resolve_project_path

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def physical_grid(half_width, spacing):
    steps = round(float(half_width)/float(spacing))
    if steps < 1 or not np.isclose(steps*spacing, half_width, rtol=1e-12, atol=0):
        raise ValueError("半宽必须为采样间距的整数倍")
    return make_grid(2*steps+1, float(spacing))


def load_config(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    opt = cfg["optical"]
    for key in ("diameter_m", "focal_length_m", "distance_m", "lambda_min_m", "lambda_max_m",
                "conventional_lambda_m"):
        if not np.isfinite(opt[key]) or opt[key] <= 0:
            raise ValueError("光学参数必须为有限正数："+key)
    if opt["wings_N"] != 3 or opt["lambda_min_m"] >= opt["lambda_max_m"]:
        raise ValueError("本批使用N=3且设计波长范围必须递增")
    lambdas = opt["wavelengths_m"]
    if len(lambdas)!=3 or lambdas != sorted(set(lambdas)):
        raise ValueError("本批必须有三个递增互异波长")
    for lam in lambdas:
        B.refractive_index_fused_silica(lam)
    names = [c["name"] for c in cfg["cases"]]
    if set(names) != {"input_coarse", "baseline", "input_fine", "output_coarse", "output_fine", "fov_large"} or len(names)!=6:
        raise ValueError("六组实验身份必须齐全且不重复")
    if cfg["input_half_width_m"] < opt["diameter_m"]/2:
        raise ValueError("输入网格必须覆盖完整孔径")
    for c in cfg["cases"]:
        for key in ("input_spacing_m", "output_spacing_m", "output_half_width_m"):
            if not np.isfinite(c[key]) or c[key]<=0:
                raise ValueError("网格参数必须为有限正数")
        physical_grid(cfg["input_half_width_m"], c["input_spacing_m"])
        physical_grid(c["output_half_width_m"], c["output_spacing_m"])
        if c["output_half_width_m"]<150e-6:
            raise ValueError("本批固定150μm评价圆盘必须被完整覆盖")
    # 检查一次只改变一个因素，以及同输入组的最大输出能包含全部原生点。
    by = {c["name"]:c for c in cfg["cases"]}
    for name in ("input_coarse", "baseline", "input_fine"):
        c = by[name]
        if (c["output_spacing_m"],c["output_half_width_m"]) != (by["baseline"]["output_spacing_m"],by["baseline"]["output_half_width_m"]):
            raise ValueError("输入收敛比较必须保持相同输出网格")
    for name in ("output_coarse", "output_fine", "fov_large"):
        if by[name]["input_spacing_m"] != by["input_fine"]["input_spacing_m"]:
            raise ValueError("输出/视场比较必须保持同一输入间距")
    for key,value in cfg["thresholds"].items():
        if not np.isfinite(value) or value<=0:
            raise ValueError("比较容差必须为有限正数："+key)
    return cfg


def run(cfg, run_dir, keep_figures=False):
    # 直接调用也不得覆盖已有场或指标；main只分配新唯一目录。
    if run_dir:
        for name in ("arrays", "metrics", "figures"):
            if any((run_dir/name).iterdir()):
                raise ValueError("拒绝覆盖非空结果子目录："+name)
    opt = cfg["optical"]
    base = B.load_and_validate_config(ROOT/"config_stage02b.json")
    records, payloads, comparisons, profiles_saved = {}, {}, [], []
    state = {"stage":"stage02b2", "status":"in_progress", "fields_completed":[],
             "scientific_status":"not_evaluated", "limitations":[
                 "最细0.5μm输入为离散参考，不是解析真值；未做0.25μm加密。",
                 "旋转与原文顺时针表述的观察面对应尚未解决，本批不改变角向公式。"]}
    log = logging.getLogger("stage02b2")
    started = time.perf_counter()
    if run_dir:
        write_json(run_dir/"checkpoint.json",state)
    cases = cfg["cases"]
    # 按输入网格串行处理；同组先传播最大最密输出，其余仅抽取原生坐标点。
    spacings = sorted(set(c["input_spacing_m"] for c in cases),reverse=True)
    for dx in spacings:
        gi = physical_grid(cfg["input_half_width_m"],dx)
        group = [c for c in cases if c["input_spacing_m"]==dx]
        master = max(group,key=lambda c:physical_grid(c["output_half_width_m"],c["output_spacing_m"]).x.n)
        gm = physical_grid(master["output_half_width_m"],master["output_spacing_m"])
        for key in ("fresnel","jeon"):
            log.info("输入dx=%.3fμm，器件%s，输入%s，直接输出%s",dx*1e6,key,gi.shape,gm.shape)
            profile = B.design_conventional_fresnel_height(gi,opt["diameter_m"],opt["focal_length_m"],opt["conventional_lambda_m"]) \
                if key=="fresnel" else B.design_jeon2019_spiral_height(gi,opt["diameter_m"],opt["focal_length_m"],3,opt["lambda_min_m"],opt["lambda_max_m"])
            fingerprint = profile.compute_fingerprint()
            if run_dir:
                height_name = "height_%s_dx%gnm.npz" % (key,dx*1e9)
                np.savez_compressed(run_dir/"arrays"/height_name, delta_h_m=profile.delta_h,
                                    mask=profile.mask,lambda_design_m=profile.lambda_design,
                                    x_in_m=gi.x.coords,y_in_m=gi.y.coords,dx_m=dx,
                                    fingerprint=fingerprint,design_type=profile.design_type,
                                    params_json=json.dumps(profile.params,sort_keys=True))
                profiles_saved.append(height_name)
            for lam in opt["wavelengths_m"]:
                nm = round(lam*1e9)
                log.info("开始直接传播：%s %dnm，dx=%.3fμm",key,nm,dx*1e6)
                t = time.perf_counter()
                field = B.propagate_device(profile,lam,gi,gm,True,1.,0.,opt["distance_m"],log,"02B2")
                elapsed = time.perf_counter()-t
                health = B.field_health_check(field["u1"],field["u2"],field["intensity"],gi,gm,1.02)
                if health["status"]!="pass":
                    raise RuntimeError("直接传播健康检查失败："+str(health))
                if profile.compute_fingerprint()!=fingerprint:
                    raise RuntimeError("同输入网格的固定高度在不同λ下发生改变")
                for c in group:
                    go = physical_grid(c["output_half_width_m"],c["output_spacing_m"])
                    ix = common_indices(gm.x.coords,go.x.coords)
                    iy = common_indices(gm.y.coords,go.y.coords)
                    u = field["u2"][np.ix_(iy,ix)]
                    I = np.abs(u)**2
                    h = B.field_health_check(field["u1"],u,I,gi,go,1.02)
                    if h["status"]!="pass":
                        raise RuntimeError("抽取场健康检查失败")
                    ident = "%s_%s_%dnm" % (c["name"],key,nm)
                    m = psf_metrics(I,go,h["Pin"],h["Pwindow"],base["rotation_bands"],base["size_roi"],base["energy"],.05,.9*np.pi)
                    m.update(identity=ident,case=c["name"],device=key,wavelength_nm=nm,
                             input_spacing_m=dx,output_spacing_m=go.dx,output_half_width_m=c["output_half_width_m"],
                             height_fingerprint=fingerprint,propagation_s=elapsed,
                             native_source_case=master["name"])
                    if c["name"]=="fov_large":
                        m["Eabs_300um"] = absolute_encircled_energy(I,go,h["Pin"],[300e-6])[0]["Eabs"]
                    if key=="jeon" and nm==540 and c["name"]=="input_fine":
                        direct = B.propagate_device(profile,lam,gi,go,True,1.,0.,opt["distance_m"],log,"原生点独立核验")
                        error = float(np.linalg.norm(direct["u2"]-u)/np.linalg.norm(u))
                        state["native_extraction_relative_l2"] = error
                        if error>1e-11:
                            raise RuntimeError("最大网格原生点抽取与独立传播不一致")
                        del direct
                    if run_dir:
                        path=run_dir/"arrays"/(ident+".npz")
                        np.savez_compressed(path,u2_complex=u,intensity_raw=I,x_out_m=go.x.coords,y_out_m=go.y.coords,
                                            wavelength_m=lam,device_key=key,case_name=c["name"],
                                            device_fingerprint=fingerprint,Pin=h["Pin"],Pwindow=h["Pwindow"],
                                            propagation_distance_m=opt["distance_m"],input_spacing_m=dx,
                                            output_spacing_m=go.dx,native_source_case=master["name"],
                                            refractive_index=B.refractive_index_fused_silica(lam),
                                            design_focal_length_m=opt["focal_length_m"],include_global_phase=True)
                        # 保存即重读：不把文件数够了当成科学验收。
                        with np.load(path) as a:
                            if not np.array_equal(a["u2_complex"],u) or not np.array_equal(a["intensity_raw"],I) \
                                    or str(a["device_fingerprint"])!=fingerprint \
                                    or float(a["Pin"])!=h["Pin"] or float(a["Pwindow"])!=float(a["intensity_raw"].sum()*go.cell_area):
                                raise RuntimeError("保存后重读不一致："+ident)
                    records[ident]=m
                    payloads[ident]=(I,go)
                    state["fields_completed"].append(ident)
                    if run_dir:
                        write_json(run_dir/"checkpoint.json",state)
                del field,u,I
                gc.collect()
                log.info("本λ完成，直接传播%.2fs，总进度%d/36",elapsed,len(records))
            del profile
            gc.collect()
    expected={"%s_%s_%dnm"%(c["name"],key,round(lam*1e9)) for c in cases for key in ("fresnel","jeon") for lam in opt["wavelengths_m"]}
    if set(records)!=expected:
        raise RuntimeError("36组身份集合不完整")
    pairs=[("input_coarse","baseline"),("baseline","input_fine"),
           ("output_coarse","input_fine"),("input_fine","output_fine")]
    for coarse,fine in pairs:
        for key in ("fresnel","jeon"):
            for lam in opt["wavelengths_m"]:
                nm=round(lam*1e9)
                a,b=("%s_%s_%dnm"%(case,key,nm) for case in (coarse,fine))
                ia,ga=payloads[a];ib,gb=payloads[b]
                delta=compare_intensities(ia,ga,ib,gb)
                met=compare_metrics(records[a],records[b],cfg["thresholds"])
                checks=dict(met["checks"],intensity_l1=delta["intensity_l1"]<=cfg["thresholds"]["intensity_l1"],
                            intensity_l2=delta["intensity_l2"]<=cfg["thresholds"]["intensity_l2"])
                comparisons.append(dict(pair=coarse+"->"+fine,device=key,wavelength_nm=nm,
                                        **delta,**met,all_checks=checks,within_tolerance=all(checks.values())))
    window=[]
    for key in ("fresnel","jeon"):
        for lam in opt["wavelengths_m"]:
            nm=round(lam*1e9);small="output_fine_%s_%dnm"%(key,nm);large="fov_large_%s_%dnm"%(key,nm)
            ia,ga=payloads[small];ib,gb=payloads[large]
            sub=ib[np.ix_(common_indices(gb.y.coords,ga.y.coords),common_indices(gb.x.coords,ga.x.coords))]
            if not np.array_equal(ia,sub):
                raise RuntimeError("扩窗改变了公共区域强度")
            window.append(dict(device=key,wavelength_nm=nm,eta_150=records[small]["eta_window"],
                               eta_300=records[large]["eta_window"],extra_capture=records[large]["eta_window"]-records[small]["eta_window"],
                               R80_150_um=records[small]["R80"]["radius_um"],R80_300_um=records[large]["R80"]["radius_um"],
                               Eabs_300um=records[large]["Eabs_300um"],common_region_exact=True))
    key_pairs=[c for c in comparisons if c["pair"] in ("baseline->input_fine","input_fine->output_fine")]
    state.update(status="completed",scientific_status="default_within_tested_tolerances" if all(c["within_tolerance"] for c in key_pairs) else "further_refinement_needed",
                 comparisons_within_tolerance=sum(c["within_tolerance"] for c in comparisons),comparisons_total=len(comparisons),
                 elapsed_s=time.perf_counter()-started)
    result=dict(state=state,metrics=records,comparisons=comparisons,window=window)
    figures=[]
    if run_dir:
        write_json(run_dir/"metrics"/"convergence.json",result)
        with (run_dir/"metrics"/"comparisons.csv").open("w",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,fieldnames=["pair","device","wavelength_nm","intensity_l1","intensity_l2","within_tolerance"])
            writer.writeheader();writer.writerows({k:c[k] for k in writer.fieldnames} for c in comparisons)
    if run_dir or keep_figures:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        labels=[c["pair"]+"\n"+c["device"]+" "+str(c["wavelength_nm"]) for c in comparisons]
        for ax,field,threshold in zip(axes,["intensity_l1","intensity_l2"],
                                      [cfg["thresholds"]["intensity_l1"],cfg["thresholds"]["intensity_l2"]]):
            ax.bar(range(len(comparisons)),[c[field]*100 for c in comparisons])
            ax.axhline(threshold*100,color="red",ls="--",label="配置筛查容差")
            ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=6)
            ax.set_ylabel(field+" (%)");ax.legend();ax.grid(axis="y",alpha=.25)
        fig.tight_layout()
        if run_dir:fig.savefig(run_dir/"figures"/"sampling_errors.png",dpi=cfg["plots"]["dpi"])
        figures.append(fig)
        fig,ax=plt.subplots(figsize=(8,4.5))
        for key in ("fresnel","jeon"):
            w=[v for v in window if v["device"]==key]
            for radius,style in [(150,"o-"),(300,"s--")]:
                ax.plot([v["wavelength_nm"] for v in w],[v["eta_%d"%radius] for v in w],style,label=key+" ±%dμm"%radius)
        ax.set_xlabel("波长(nm)");ax.set_ylabel("方窗捕获能量/Pin");ax.legend();ax.grid(alpha=.3);fig.tight_layout()
        if run_dir:fig.savefig(run_dir/"figures"/"window_energy.png",dpi=cfg["plots"]["dpi"])
        figures.append(fig)
        if not keep_figures:
            for fig in figures:plt.close(fig)
            figures=[]
    if run_dir:
        lines=["# 02B-2采样与视场报告","", "计算状态：completed；科学筛查："+state["scientific_status"],
               "", "36组原始场载荷，18次主传播及Jeon540原生点独立核验。比较容差取自config_effective.json。",
               "", "同网格比较原始强度；输出加密以双线性强度插值作诊断，未用插值生成器件PSF。",
               "", "| 比较 | 器件 | λ/nm | L1/% | L2/% | 全部指标达标 |", "| --- | --- | --- | --- | --- | --- |"]
        for c in comparisons:
            lines.append("| %s | %s | %d | %.4f | %.4f | %s |"%(c["pair"],c["device"],c["wavelength_nm"],100*c["intensity_l1"],100*c["intensity_l2"],c["within_tolerance"]))
        lines += ["","| 器件 | λ/nm | η150 | η300 | 扩窗新增 | R80_150/μm | R80_300/μm |","| --- | --- | --- | --- | --- | --- | --- |"]
        for w in window:
            lines.append("| %s | %d | %.6f | %.6f | %.6f | %s | %s |"%(w["device"],w["wavelength_nm"],w["eta_150"],w["eta_300"],w["extra_capture"],w["R80_150_um"],w["R80_300_um"]))
        lines += ["","本批最细网格不是解析真值。未进行0.25μm输入加密、制造量化、6.22μm像元积分或重建。",
                  "原文方向坐标对应仍待解释。原始I与Pin未经归一化，窗口未捕获量不是制造损耗。",
                  "参数/公式来源沿用02A/02B-1，D/f应用于图3、材料与网格为已记录实施假设。"]
        (run_dir/"report_stage02b2.md").write_text("\n".join(lines),encoding="utf-8")
        needed=["metrics/convergence.json","metrics/comparisons.csv","figures/sampling_errors.png","figures/window_energy.png","report_stage02b2.md"]
        found={p.stem for p in (run_dir/"arrays").glob("*.npz") if not p.name.startswith("height_")}
        if found!=expected or any(not (run_dir/name).exists() for name in needed):
            raise RuntimeError("必需产物缺失")
        write_json(run_dir/"completion.json",state);write_json(run_dir/"checkpoint.json",state)
    return result,figures


def main(argv=None):
    parser=argparse.ArgumentParser(description="02B-2 CPU采样与视场验证")
    parser.add_argument("--config",default="config_stage02b2.json")
    parser.add_argument("--no-save",action="store_true");parser.add_argument("--show-plots",action="store_true")
    args=parser.parse_args(argv)
    cfg=load_config(resolve_project_path(args.config,ROOT))
    runtime=effective_runtime(cfg,args.no_save,args.show_plots)
    plt,backend=configure_plotting(runtime["show_plots"])
    run_dir=B.allocate_unique_run_dir("stage02b2") if runtime["save_results"] else None
    log=B.setup_logger(run_dir);logging.getLogger("stage02b2").handlers=log.handlers
    logging.getLogger("stage02b2").setLevel(logging.INFO);logging.getLogger("stage02b2").propagate=False
    if run_dir:
        write_json(run_dir/"config_effective.json",cfg);write_json(run_dir/"environment.json",environment_info())
        files=["main_stage02b2.py","config_stage02b2.json","optics/convergence.py","main_stage02b.py",
               "optics/psf_analysis.py","optics/doe.py","optics/materials.py","optics/propagation.py","optics/coordinates.py"]
        write_json(run_dir/"source_manifest.json",{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    log.info("02B-2开始，CPU，保存=%s，输出=%s",runtime["save_results"],run_dir)
    try:
        result,figures=run(cfg,run_dir,runtime["show_plots"])
        log.info("本批完成：%s",result["state"])
        if runtime["show_plots"] and figures:plt.show()
        return 0
    except Exception as exc:
        log.exception("02B-2失败，保留所有证据")
        if run_dir:write_json(run_dir/"completion.json",{"status":"failed","error":str(exc)})
        return 1


if __name__=="__main__":
    raise SystemExit(main())
