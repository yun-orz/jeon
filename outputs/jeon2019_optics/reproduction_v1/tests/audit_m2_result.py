"""独立读取M2保存数组重算采样门槛；不调用生产比较或面积映射实现。"""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(path):
    dest=Path(path).resolve()
    report=json.loads((dest/"audit.json").read_text(encoding="utf-8"))
    failures=[]; maxima={}; mapping={}
    pairs={"input_q4":("coarse_q4","fine_q4"),"input_q8":("coarse_q8","fine_q8"),
           "quadrature_1um":("coarse_q4","coarse_q8"),"quadrature_0_5um":("fine_q4","fine_q8")}
    for row in report["bands"]:
        data=np.load(dest/f"{row['method']}_{row['wavelength_nm']}nm.npz")
        for name,(a,b) in pairs.items():
            ka=data["kernel_"+a];kb=data["kernel_"+b]
            pa=ka*row["metrics"][a]["pin"];pb=kb*row["metrics"][b]["pin"]
            actual={"pixel_l1":float(abs(pa-pb).sum()/pb.sum()),"pixel_l2":float(np.linalg.norm(pa-pb)/np.linalg.norm(pb)),
                    "kernel_l1":float(abs(ka-kb).sum()/kb.sum()),"kernel_l2":float(np.linalg.norm(ka-kb)/np.linalg.norm(kb)),
                    "eta_absolute":float(abs(ka.sum()-kb.sum()))}
            for key,value in actual.items():
                if abs(value-row["comparisons"][name][key])>1e-12:failures.append(f"保存数组与指标不符：{row['method']}/{row['wavelength_nm']}/{name}/{key}")
                limit=.005 if key.endswith("l1") else .01 if key.endswith("l2") else .001
                if value>limit:failures.append(f"采样阈值失败：{key}={value}")
                maxima[key]=max(maxima.get(key,0),value)
            for band in ["primary","sensitivity"]:
                ma=row["metrics"][a]["rotation_bands"][band];mb=row["metrics"][b]["rotation_bands"][band]
                if ma["reliable"]!=mb["reliable"]:failures.append("角度可靠性状态不一致")
                aa=ma["alpha_wrapped_deg"];ab=mb["alpha_wrapped_deg"]
                if aa is not None and ab is not None:
                    diff=abs((aa-ab+60)%120-60)
                    if diff>.5:failures.append(f"角度差超阈值：{diff}")
                    maxima["angle_deg"]=max(maxima.get("angle_deg",0),diff)
            for radius in ["R50","R80"]:
                ra=row["metrics"][a][radius];rb=row["metrics"][b][radius]
                if ra["status"]!=rb["status"]:failures.append("半径状态不一致")
                if ra["radius_um"] is not None and rb["radius_um"] is not None:
                    difference=abs(ra["radius_um"]-rb["radius_um"])/6.22
                    if difference>1:failures.append("半径差超过一个像元")
                    maxima["radius_pixels"]=max(maxima.get("radius_pixels",0),difference)
    # 独立二维矩形交集，检查所有实际半尺寸核，不复用一维生产矩阵。
    s=(np.arange(97)-48)*6.22e-6;t=(np.arange(49)-24)*12.44e-6
    for method in ["jeon","fresnel"]:
        native=np.load(dest/f"{method}_native.npy");half=np.load(dest/f"{method}_training.npy")
        expected=np.zeros_like(half)
        for iy,y in enumerate(t):
            jy=np.flatnonzero((s+3.11e-6>y-6.22e-6)&(s-3.11e-6<y+6.22e-6))
            for ix,x in enumerate(t):
                jx=np.flatnonzero((s+3.11e-6>x-6.22e-6)&(s-3.11e-6<x+6.22e-6))
                for sy in jy:
                    height=max(0,min(y+6.22e-6,s[sy]+3.11e-6)-max(y-6.22e-6,s[sy]-3.11e-6))
                    for sx in jx:
                        width=max(0,min(x+6.22e-6,s[sx]+3.11e-6)-max(x-6.22e-6,s[sx]-3.11e-6))
                        expected[:,iy,ix]+=native[:,sy,sx]*width*height/(6.22e-6)**2
        mass=float(max(abs(native.sum((1,2))-half.sum((1,2)))))
        error=float(np.linalg.norm(expected-half)/np.linalg.norm(half))
        mapping[method]={"mass_max_absolute_error":mass,"rectangle_overlap_relative_error":error}
        if mass>1e-12 or error>1e-12:failures.append("独立面积映射不通过："+method)
    if len(report["bands"])!=50:failures.append("器件波段检查未完成50项")
    result={"status":"passed" if not failures else "failed","completed_bands":len(report["bands"]),
            "sampling_maxima":maxima,"independent_mapping":mapping,"failures":failures}
    target=dest/"independent_audit.json"
    if target.exists():raise ValueError("独立审核结果已存在，拒绝覆盖")
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("run_dir");args=parser.parse_args()
    result=audit(args.run_dir);print(json.dumps(result,ensure_ascii=True,indent=2))
    raise SystemExit(0 if result["status"]=="passed" else 2)
