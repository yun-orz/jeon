"""M3规范JSON、M2不可变依赖、预算与来源桥接。"""
import hashlib
import json
import shutil
from pathlib import Path
from audit import HERE, ROOT, read_json, safe_path, sha256


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode("utf-8")


def fingerprint(value):return hashlib.sha256(canonical(value)).hexdigest()


def save_json(path,value):
    """不可变新文件；进度用递增文件名保存，不覆盖旧状态。"""
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def rank(seed,value):return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def freeze_m2_bridge(m2_run,destination):
    """实施M3入口前保存M2登记的旧入口字节，明确记录后续入口演进。"""
    from m2_optics import check_run
    result=check_run(m2_run)
    if result["status"]!="passed":raise ValueError("M2当前核验未通过")
    dest=Path(destination).resolve();dest.mkdir(parents=True,exist_ok=False)
    old=HERE/"main.py";snapshot=dest/"m2_main_frozen.py"
    shutil.copyfile(old,snapshot)
    bridge={"schema":"jeon2019-m3-parent-bridge-v1","m2_run":str(Path(m2_run).resolve()),
            "verified_before_m3":result,"replacements":{old.relative_to(ROOT).as_posix():snapshot.relative_to(ROOT).as_posix()},
            "snapshot_sha256":sha256(snapshot),"policy":"只替代M2冻结的历史入口源码读路径；物理源码、M2配置及产物不变"}
    save_json(dest/"bridge.json",bridge)
    return bridge


def verify_parent(m2_run,bridge_file):
    """核验M2包全部原始指纹；旧入口SHA通过显式保存的原字节验证。"""
    from m2_optics import fingerprint as m2_fingerprint
    dest=Path(m2_run).resolve();bridge=read_json(bridge_file)
    if Path(bridge["m2_run"]).resolve()!=dest:raise ValueError("桥接父目录不一致")
    if set(bridge["replacements"])!={(HERE/"main.py").relative_to(ROOT).as_posix()}:raise ValueError("桥接只允许历史入口")
    snapshot=safe_path(next(iter(bridge["replacements"].values())))
    if sha256(snapshot)!=bridge["snapshot_sha256"]:raise ValueError("旧入口快照SHA变化")
    audit=read_json(dest/"audit.json")
    if audit["status"]!="passed" or not audit["acceptance"]["all_passed"] or len(audit["bands"])!=50:
        raise ValueError("父M2审核不完整")
    for name,digest in audit["input_fingerprints"].items():
        path=safe_path(bridge["replacements"].get(name,name))
        if sha256(path)!=digest:raise ValueError("M2依赖SHA变化："+name)
    packages={}
    for method in ["jeon","fresnel"]:
        p=read_json(dest/f"{method}_package.json")
        if p["fingerprint"]!=m2_fingerprint({k:v for k,v in p.items() if k!="fingerprint"}) or p["fingerprint"]!=audit["output_fingerprints"][method]:raise ValueError("父包指纹错误")
        if not p["audit"]["physical_package_passed"]:raise ValueError("父物理包未通过")
        if p["sources"]["effective_config"]!=m2_fingerprint(read_json(dest/"effective_config.json")):raise ValueError("父有效配置改变")
        for key,kind in [("kernels_native","native"),("kernels_training","training")]:
            if sha256(dest/p[key])!=p["kernel_sha256"][kind]:raise ValueError("父核改变")
        if sha256(dest/p["spectral_response"])!=p["response_sha256"]:raise ValueError("父响应改变")
        if sha256(dest/p["mapping"]["file"])!=p["mapping"]["sha256"]:raise ValueError("父面积映射改变")
        source=p["sources"]["response"]
        for key,sha_key in [("raw_file","raw_sha256"),("license_file","license_sha256")]:
            if sha256(dest/source[key])!=source[sha_key]:raise ValueError("父响应来源改变")
        packages[method]=p
    for r in audit["bands"]:
        if sha256(dest/f"{r['method']}_{r['wavelength_nm']}nm.npz")!=r["array_sha256"]:raise ValueError("父波段数组改变")
    return packages


def space_budget(compressed,unpacked,decoded,processed,checkpoint=8*1024**3,cache=512*1024**2,already_present=0):
    """完整保留下载分块和原始文件；估计包含解码、预处理及checkpoint，不自行清理。"""
    parts={"compressed":int(compressed),"unpacked":int(unpacked),"decoded_mat":int(decoded),
           "preprocessed":int(processed),"checkpoint_retention":int(checkpoint),"cache_reserve":int(cache)}
    if min(parts.values())<0 or already_present<0:raise ValueError("空间估计不能为负")
    required=max(0,sum(parts.values())-already_present)
    safety=int(required*.1);free=shutil.disk_usage(ROOT).free
    return {"components":parts,"already_present_bytes":already_present,"safety_bytes":safety,
            "required_bytes":required+safety,"free_bytes":free,"passed":free>=required+safety,
            "note":"D盘10%余量；不删除历史、不移盘；任何未知项必须另记为未核实"}
