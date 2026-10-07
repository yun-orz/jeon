"""M3官方来源只读发现；保存实际页面与访问失败，不绕过账号授权。"""
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

URLS={
    "harvard":"https://vision.seas.harvard.edu/hyperspec/d2x5g3/",
    "icvl":"https://icvl.cs.bgu.ac.il/pages/researches/hyperspectral-imaging.html",
    "icvl_catalog":"https://huggingface.co/api/datasets/ICVL-BGU/ICVL_HS_2016/tree/main/mat?recursive=false&expand=false&limit=1000",
    "icvl_card":"https://huggingface.co/datasets/ICVL-BGU/ICVL_HS_2016/raw/main/README.md",
    "kaist":"https://www.vclab.kaist.ac.kr/siggraphasia2017p1/kaistdataset.html",
    "kaist_example":"https://www.vclab.kaist.ac.kr/siggraphasia2017p1/kaistdataset/example.m"}


def discover(dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    report={"access_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"sources":{},"downloads":[]}
    for key,url in URLS.items():
        try:
            request=urllib.request.Request(url,headers={"User-Agent":"Jeon2019-M3-academic-reproduction/1.0"})
            with urllib.request.urlopen(request,timeout=30) as stream:
                content=stream.read(4*1024*1024)
                if stream.read(1):raise ValueError("来源元数据超过4MiB")
                headers=dict(stream.headers);final_url=stream.url
            path=dest/(key+".txt");path.write_bytes(content)
            report["sources"][key]={"url":url,"final_url":final_url,"status":"available","file":path.name,"sha256":hashlib.sha256(content).hexdigest(),"headers":headers}
            if key in ["harvard","kaist"]:
                html=content.decode("utf-8",errors="replace")
                links=re.findall(r'(?:href|src)\s*=\s*[\"\x27]([^\"\x27]+)',html,re.I)
                for link in links:
                    full=urljoin(url,link)
                    if re.search(r'\.(?:tgz|zip|tar\.gz|exr)(?:\?|$)',full,re.I):
                        try:
                            request=urllib.request.Request(full,method="HEAD")
                            with urllib.request.urlopen(request,timeout=30) as stream:
                                size=stream.headers.get("Content-Length");remote_headers=dict(stream.headers)
                            report["downloads"].append({"dataset":key,"url":full,"bytes":int(size) if size else None,"headers":remote_headers})
                        except Exception as exc:report["downloads"].append({"dataset":key,"url":full,"error":repr(exc)})
        except Exception as exc:report["sources"][key]={"url":url,"status":"dependency_pending","error":repr(exc)}
        print(key,report["sources"][key]["status"],flush=True)
    (dest/"review.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("destination");args=p.parse_args()
    print(json.dumps(discover(args.destination),ensure_ascii=True,indent=2))
