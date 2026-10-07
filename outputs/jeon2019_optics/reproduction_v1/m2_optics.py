"""M2正式光学冻结；每次执行写新目录，采样失败保持原始阈值和逐波段记录。"""
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import uuid
from pathlib import Path
import numpy as np
from audit import HERE, ROOT, read_json, safe_path, sha256, validate_records
from m2_operator import overlap_matrix, remap, mapping_audit, numerical_audit

FIXED_HEIGHT = "3469c7a4717a765d31e0433d9180c015520e00b99d08a0c656e791ac2b917f51"


def protection_check(manifest):
    """保留M0长度/SHA检查，单次stat并报告进度，便于定位慢文件。"""
    records = validate_records(manifest)
    failures=[]; checked_bytes=0
    print(f"M2 历史保护：{len(records)}个文件，开始读取",flush=True)
    for i,item in enumerate(records):
        path=safe_path(item["path"])
        try:
            actual=path.stat().st_size
            count=item["bytes"]
            length_ok=actual>=count if item["mode"]=="append_only" else actual==count
            if not length_ok:failures.append({"path":item["path"],"error":"长度不符"})
            elif sha256(path,count if item["mode"]=="append_only" else None)!=item["sha256"]:
                failures.append({"path":item["path"],"error":"SHA不符"})
            checked_bytes+=count
        except OSError as exc:failures.append({"path":item["path"],"error":repr(exc)})
        if (i+1)%1000==0:print(f"M2 历史保护：{i+1}/{len(records)}",flush=True)
    return {"passed":not failures,"checked":len(records),"checked_bytes":checked_bytes,"failures":failures}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def response_source(dest, frozen=None):
    """冻结实际下载内容和许可证，保留文件共同相对尺度，不作通道归一化。"""
    entry = next(s for s in read_json(HERE/"sources.json")["sources"] if s["id"]=="rawtoaces_canon_5d_mark_iii")
    raw = dest/"Canon_EOS_5D_Mark_III_380_780_5.json"
    license_file = dest/"RAWtoACES_LICENSE"
    license_url = "https://raw.githubusercontent.com/AcademySoftwareFoundation/rawtoaces-data/main/LICENSE"
    for url,target,name in [(entry["url"],raw,raw.name),(license_url,license_file,license_file.name)]:
        if frozen:
            shutil.copyfile(Path(frozen)/name, target)
        else:
            with urllib.request.urlopen(url, timeout=60) as stream:
                target.write_bytes(stream.read())
    data = read_json(raw)
    if (data["header"]["model"] != "EOS 5D Mark III" or data["header"]["license"]!="Apache-2.0"
            or data["spectral_data"]["units"]!="relative" or data["spectral_data"]["index"]["main"]!=["R","G","B"]):
        raise ValueError("响应型号、许可、单位或通道顺序不符")
    table = data["spectral_data"]["data"]["main"]
    if sorted(map(int, table)) != list(range(380,781,5)):
        raise ValueError("响应实际表必须为380–780nm/5nm")
    response = np.array([table[str(w)] for w in range(420,661,10)], dtype=np.float64).T
    if not np.isfinite(response).all() or np.any(response<0):
        raise ValueError("相对响应包含非法数值")
    return response, {"url":entry["url"], "raw_file":raw.name, "raw_sha256":sha256(raw),
                      "license_url":license_url,"license_file":license_file.name,"license_sha256":sha256(license_file),
                      "license":"Apache-2.0", "access_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                      "version":data["header"]["schema_version"], "header":data["header"],
                      "version_freeze":"访问内容SHA，未取得提交号", "common_scale_factor":1.0,
                      "author_calibration":False, "absolute_QE":False}


def load_physics():
    sys.path.insert(0,str(HERE.parent))
    from optics.g1_forward import profile_for, measure, compare, native_parseval
    from optics.g1_alignment import clockwise_profile
    from optics.doe import design_conventional_fresnel_height
    from optics.materials import get_material_model_metadata
    return profile_for,measure,compare,native_parseval,clockwise_profile,design_conventional_fresnel_height,get_material_model_metadata


def run(config_path, resume=None):
    from main import load_m0
    index,_ = load_m0()
    protection = protection_check(read_json(safe_path(index["protection"])))
    if not protection["passed"]:
        raise ValueError("历史保护检查失败")
    c = read_json(config_path)
    if set(c)-{"response_source_dir"} != {"schema","budget_seconds","reserve_bytes","spectral_input"} or c["schema"]!="jeon2019-m2-config-v1":
        raise ValueError("M2配置字段不符，物理参数由M0历史来源冻结")
    if (isinstance(c["budget_seconds"],bool) or not 0<c["budget_seconds"]<=28800
            or type(c["reserve_bytes"]) is not int or c["reserve_bytes"]<512*1024**2
            or c["spectral_input"]!="relative_energy_density_per_nm"):
        raise ValueError("预算、空间预留或谱密度协议不符")
    history = read_json(safe_path(index["stages"]))["stages"]
    cw = next(s for s in history if s["stage"]=="G1_CW")
    source_dir = safe_path(cw["source_run"])
    source_config = read_json(source_dir/"config_effective.json")["source_config"]
    profile_for,measure,compare,parseval,clockwise,fresnel,material = load_physics()
    profiles = {}
    for refined in [False,True]:
        reference = profile_for(source_config,refined)
        profiles[("jeon",refined)] = clockwise(reference)
        profiles[("fresnel",refined)] = fresnel(reference.grid,.001,.05,550e-9)
    if profiles[("jeon",False)].compute_fingerprint()!=FIXED_HEIGHT:
        raise ValueError("CW固定高度指纹不符")
    dest = ROOT/"outputs/jeon2019_optics/results/reproduction_v1/m2"/("run_"+time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
    dest.mkdir(parents=True)
    command = [sys.executable,str(HERE/"main.py"),"optics","--config",str(Path(config_path).resolve())]
    if resume: command += ["--resume",str(Path(resume).resolve())]
    frozen_sources = {str(p.relative_to(ROOT)).replace("\\","/"):sha256(p)
                      for p in list((HERE.parent/"optics").glob("*.py"))+[HERE/"main.py",HERE/"m2_optics.py",HERE/"m2_operator.py",Path(config_path).resolve(),HERE/"contracts.json",HERE/"sources.json",source_dir/"config_effective.json",source_dir/"source_manifest.json",safe_path(cw["audit"])]}
    effective = {"execution":c,"physics":source_config,"chirality":"CW","origin_deg":0,
                 "fresnel_design_nm":550,"material":material(),
                 "formula":"n²=1+sum(Bi*L²/(L²-Ci)), L以μm计；delta=r²/(sqrt(r²+f²)+f)；h=(floor(delta/lambda_design)*lambda_design-delta)/(n(lambda_design)-1)；U=mask*exp(i*2pi*(n(lambda)-1)*h/lambda)",
                 "propagation":"已审核直接可分离Fresnel积分；全局相位保留；相机像元q4/q8中点强度积分", "physics_source":"M0历史索引G1_CW source_config，历史budget不作为本次执行预算"}
    response_dir = safe_path(c["response_source_dir"]) if c.get("response_source_dir") else None
    if response_dir:
        for name in ["Canon_EOS_5D_Mark_III_380_780_5.json","RAWtoACES_LICENSE"]:
            path=response_dir/name
            frozen_sources[path.relative_to(ROOT).as_posix()]=sha256(path)
    for stage in ["G1_window","G1_lightpipes","G1_lp_sampling","G1_padding"]:
        record=next(s for s in history if s["stage"]==stage)
        path=safe_path(record["audit"])
        if sha256(path)!=record["audit_sha256"]:raise ValueError("历史审核SHA不符")
        frozen_sources[path.relative_to(ROOT).as_posix()]=sha256(path)
    original_height=source_dir/"arrays/height_cw_fixed.npz"
    frozen_sources[original_height.relative_to(ROOT).as_posix()]=sha256(original_height)
    write_json(dest/"effective_config.json",effective)
    write_json(dest/"source_fingerprints.json",frozen_sources)
    prior = None
    if resume:
        prior = Path(resume).resolve()
        prior_sources = read_json(prior/"source_fingerprints.json")
        if prior_sources != frozen_sources or read_json(prior/"effective_config.json")!=effective:
            raise ValueError("续接配置或源码指纹不符")
    report = {"milestone":"M2","status":"dependency_pending","commands":[command],"effective_config":effective,
              "input_fingerprints":frozen_sources,"output_fingerprints":{},"key_numbers":{},"failures":[],
              "acceptance":{"all_passed":False}, "resume_command":command+["--resume",str(dest)] if not resume else command[:-2]+["--resume",str(dest)],
              "next_inputs":["M3数据及共同训练增益拟合，物理包不悄悄改写；最终包须包含父指纹"],"paper_alignment_passed":False,
              "run_dir":str(dest),"bands":[],"historical_protection":protection,"measurement_gain":{"status":"pending","value":None,"fit_scope":"M3仅训练集，两器件共用"},
              "limitations":["替代同型号相对响应，不是作者标定或绝对QE","25波段局部采样检查不证明无限域收敛","旧G1–G5模型仅作历史诊断，后续从头训练"]}
    start = time.perf_counter()
    matrix,source_edges,target_edges,empty = overlap_matrix()
    np.savez_compressed(dest/"area_mapping.npz",matrix=matrix,source_edges_m=source_edges,target_edges_m=target_edges,outer_empty_area_m2=empty)
    try:
        if shutil.disk_usage(ROOT).free<c["reserve_bytes"]:
            report["status"]="space_stopped"
            raise RuntimeError("剩余空间低于M2预留")
        response, response_meta = response_source(dest,prior or response_dir)
        write_json(dest/"response_metadata.json",response_meta)
        np.save(dest/"response.npy",response)
        weights = np.full(25,10.0)
        for method in ["jeon","fresnel"]:
            for nm in range(420,661,10):
                if time.perf_counter()-start>=c["budget_seconds"]:
                    report["status"]="budget_stopped"
                    raise RuntimeError("预算已到，保留每波段记录，续接写新目录")
                if shutil.disk_usage(ROOT).free<c["reserve_bytes"]:
                    report["status"]="space_stopped"
                    raise RuntimeError("剩余空间低于M2预留")
                name=f"{method}_{nm}nm"
                if prior and (prior/(name+".json")).exists():
                    row = read_json(prior/(name+".json"))
                    if sha256(prior/(name+".npz"))!=row["array_sha256"]:
                        raise ValueError("续接波段数组SHA不符")
                    shutil.copyfile(prior/(name+".npz"),dest/(name+".npz"))
                    write_json(dest/(name+".json"),row)
                else:
                    measurements = {}
                    for refined in [False,True]:
                        for q in [4,8]:
                            measurements[(refined,q)] = measure(profiles[(method,refined)],nm*1e-9,source_config,q)
                    comparisons = {"input_q4":compare(measurements[(False,4)],measurements[(True,4)],source_config),
                                   "input_q8":compare(measurements[(False,8)],measurements[(True,8)],source_config),
                                   "quadrature_1um":compare(measurements[(False,4)],measurements[(False,8)],source_config),
                                   "quadrature_0_5um":compare(measurements[(True,4)],measurements[(True,8)],source_config)}
                    pv = parseval(measurements[(True,8)]["u1"],profiles[(method,True)].grid,nm*1e-9,.05)
                    np.savez_compressed(dest/(name+".npz"),**{f"kernel_{'fine' if refined else 'coarse'}_q{q}":m["kernel"] for (refined,q),m in measurements.items()})
                    row = {"method":method,"wavelength_nm":nm,"status":"completed", "comparisons":comparisons,
                           "metrics":{f"{'fine' if refined else 'coarse'}_q{q}":m["metrics"] for (refined,q),m in measurements.items()},
                           "parseval":pv,"array_sha256":sha256(dest/(name+".npz")),"passed":all(v["passed"] for v in comparisons.values()) and pv["passed"]}
                    write_json(dest/(name+".json"),row)
                report["bands"].append(row)
                write_json(dest/"audit.json",report)
                print(f"M2 {method} {nm}nm {'通过' if row['passed'] else '采样失败'}，{time.perf_counter()-start:.1f}s",flush=True)
        for method in ["jeon","fresnel"]:
            native = np.stack([np.load(dest/f"{method}_{nm}nm.npz")["kernel_fine_q8"] for nm in range(420,661,10)])
            training = remap(native,matrix)
            np.save(dest/f"{method}_native.npy",native); np.save(dest/f"{method}_training.npy",training)
            mapping = mapping_audit(native,training,matrix,6.22e-6)
            numeric = {"native":numerical_audit(native,response,weights),"training":numerical_audit(training,response,weights)}
            write_json(dest/f"{method}_mapping_audit.json",mapping); write_json(dest/f"{method}_numeric_audit.json",numeric)
            passed = all(r["passed"] for r in report["bands"] if r["method"]==method) and mapping["passed"] and all(r["passed"] for r in numeric.values())
            package = {"schema":"jeon2019-operator-physical-v1","version":1,"method":method,"wavelengths_nm":list(range(420,661,10)),
                       "kernels_native":f"{method}_native.npy","kernels_training":f"{method}_training.npy",
                       "kernel_sha256":{"native":sha256(dest/f"{method}_native.npy"),"training":sha256(dest/f"{method}_training.npy")},
                       "coordinates_m":{"native":((np.arange(97)-48)*6.22e-6).tolist(),"training":((np.arange(49)-24)*12.44e-6).tolist()},
                       "pixel_pitch_m":{"native":6.22e-6,"training":12.44e-6},"spectral_response":"response.npy","response_sha256":sha256(dest/"response.npy"),
                       "spectral_weights":{"values":weights.tolist(),"units":"nm","applied_once":True},"measurement_gain":report["measurement_gain"],
                       "radiometric_units":{"input":"相对能量谱密度/nm，中心波长分段常数近似","band_energy":"谱密度乘10nm","output":"共同相对响应加权的有限窗口相对能量RGB","kernel":"像元功率/入射孔径功率，无量纲；禁止归一化质量1","photon_conversion":False},
                       "height_sha256":profiles[(method,False)].compute_fingerprint(),
                       "mapping":{"file":"area_mapping.npz","sha256":sha256(dest/"area_mapping.npz"),"mass_preserving":True,"outer_empty_area_m2":float(empty.sum()),"origin_m":[0,0],"audit":mapping},
                       "sources":{"code_and_inputs":frozen_sources,"response":response_meta,"effective_config":fingerprint(effective),"m0":index["artifacts"]},
                       "assumptions":report["limitations"],"audit":{"sampling_passed":all(r["passed"] for r in report["bands"] if r["method"]==method),"numerical":numeric,"physical_package_passed":passed,"final_training_ready":False},
                       "parent_fingerprint":None,"fingerprint":None}
            required = read_json(HERE/"contracts.json")["operator_package"]["required"]
            if not set(required)<=set(package):raise ValueError("算子包合同缺字段")
            package["fingerprint"] = fingerprint({k:v for k,v in package.items() if k!="fingerprint"})
            write_json(dest/f"{method}_package.json",package)
            report["output_fingerprints"][method]=package["fingerprint"]
            report["key_numbers"][method]={"eta_min":float(native.sum((1,2)).min()),"eta_max":float(native.sum((1,2)).max()),"mapping":mapping,"numerical":numeric}
        report["acceptance"] = {"all_passed":all(read_json(dest/f"{method}_package.json")["audit"]["physical_package_passed"] for method in ["jeon","fresnel"]),"completed_bands":len(report["bands"]),"gain_pending_explicit":True}
        report["status"] = "passed" if report["acceptance"]["all_passed"] else "failed"
        report["failures"] = [{"method":r["method"],"wavelength_nm":r["wavelength_nm"],"comparisons":r["comparisons"]} for r in report["bands"] if not r["passed"]]
    except Exception as exc:
        if report["status"] not in ["budget_stopped","space_stopped"]: report["status"]="failed"
        report["failures"].append({"error":repr(exc)})
    report["elapsed_seconds"] = time.perf_counter()-start
    completed = {(r["method"],r["wavelength_nm"]) for r in report["bands"]}
    report["unfinished"] = [{"method":m,"wavelength_nm":w} for m in ["jeon","fresnel"] for w in range(420,661,10) if (m,w) not in completed]
    write_json(dest/"audit.json",report)
    text = f"# M2 光学与RGB算子审核\n\n状态：{report['status']}。已完成{len(completed)}/50器件波段，耗时{report['elapsed_seconds']:.1f}秒。\n\n实际执行命令：\n\n```powershell\n& " + " ".join('"'+str(s)+'"' for s in command) + "\n```\n\n共同训练增益：pending，未拟合；M3须创建带父指纹的新最终包。原生97×97/6.22μm，半尺寸49×49/12.44μm，保留有限窗口质量与几何原点。\n\n参数、Sellmeier公式及系数见 effective_config.json；原始响应、许可证、访问内容SHA见 response_metadata.json；全部输入源码SHA见 source_fingerprints.json。\n\n逐波段四组比较保持G1原定义和原阈值；parseval只检查数值功率，不证明无限域收敛。Fresnel无可靠旋转结构时按G1状态比较，不虚构角度。独立面积映射、点质量、对称核、质心和NumPy/Torch FP64核对见各方法审核文件。\n\n失败与未完成情况见 audit.json；禁止把本次状态解释为论文对齐。下一阶段M3，仅物理包审核通过后推进。\n"
    if report["status"]!="passed":
        text += "\n失败诊断：先查看每波段input_q4/input_q8与quadrature比较定位输入或探测器误差，不放宽阈值。续接命令：\n\n```powershell\n& " + " ".join('"'+str(s)+'"' for s in report["resume_command"]) + "\n```\n"
    (dest/"report.md").write_text(text,encoding="utf-8")
    return report


def check_run(path):
    """检查指定M2结果的合同、源码及内容SHA，不以目录存在推定通过。"""
    dest = Path(path).resolve(); audit = read_json(dest/"audit.json"); failures=[]
    for name,digest in audit["input_fingerprints"].items():
        if sha256(safe_path(name))!=digest:failures.append("输入SHA变化："+name)
    for method in ["jeon","fresnel"]:
        file = dest/f"{method}_package.json"
        if not file.exists():failures.append("未完成算子包："+method);continue
        p = read_json(file)
        if not set(read_json(HERE/"contracts.json")["operator_package"]["required"])<=set(p):failures.append("合同字段缺失")
        if p["fingerprint"]!=fingerprint({k:v for k,v in p.items() if k!="fingerprint"}):failures.append("包指纹变化")
        if p["fingerprint"]!=audit["output_fingerprints"].get(method):failures.append("包指纹与运行审核不一致")
        if p["sources"]["effective_config"]!=fingerprint(read_json(dest/"effective_config.json")):failures.append("有效配置指纹变化")
        for key,kind in [("kernels_native","native"),("kernels_training","training")]:
            if sha256(dest/p[key])!=p["kernel_sha256"][kind]:failures.append("核SHA变化")
        if sha256(dest/p["spectral_response"])!=p["response_sha256"]:failures.append("响应SHA变化")
        for key,sha_key in [("raw_file","raw_sha256"),("license_file","license_sha256")]:
            if sha256(dest/p["sources"]["response"][key])!=p["sources"]["response"][sha_key]:failures.append("响应来源SHA变化")
        if sha256(dest/p["mapping"]["file"])!=p["mapping"]["sha256"]:failures.append("面积映射SHA变化")
        if not p["audit"]["physical_package_passed"]:failures.append("物理审核未通过："+method)
    for r in audit["bands"]:
        if sha256(dest/f"{r['method']}_{r['wavelength_nm']}nm.npz")!=r["array_sha256"]:failures.append("逐波段SHA变化")
    passed = audit["status"]=="passed" and audit["acceptance"]["all_passed"] and len(audit["bands"])==50 and not failures
    return {"status":"passed" if passed else "failed","run_dir":str(dest),"failures":failures,"gain_pending":True,"paper_alignment_passed":False}
