"""M3有界流式下载；断点、源版本和每段SHA登记，不截断或删除失败文件。"""
import hashlib
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path
from audit import sha256
from m3_common import save_json


class DownloadStopped(RuntimeError):
    def __init__(self,status,message):super().__init__(message);self.status=status


def download(url,destination,expected_bytes,deadline,space_check,expected_sha=None):
    """一个连接顺序下载，64MiB登记一次；续接只接受正确Range和同一源版本。"""
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    if expected_bytes<=0:raise ValueError("官方下载大小必须为正")
    logs=sorted(dest.glob("progress_*.json"))
    prior=json.loads(logs[-1].read_text(encoding="utf-8")) if logs else None
    if prior and (prior["url"]!=url or prior["expected_bytes"]!=expected_bytes):raise ValueError("断点来源不一致")
    offset=0;segment_records=[]
    if prior:
        path=dest/prior["file"];segment_records=prior["segments"]
        with path.open("rb") as stream:
            for row in segment_records:
                if row['offset']!=offset or not 0<row['bytes']<=64*1024**2:raise ValueError("断点分段不连续或长度非法")
                value=stream.read(row["bytes"])
                if hashlib.sha256(value).hexdigest()!=row["sha256"]:raise ValueError("断点分段SHA不符，保留文件")
                offset+=row["bytes"]
        if offset!=prior['completed_bytes'] or offset>expected_bytes:raise ValueError("登记断点长度与分段不符")
        if path.stat().st_size!=offset:
            # 不截断失败尾部；复制已核验前缀到新尝试文件，旧文件保留。
            fresh=dest/("payload_"+uuid.uuid4().hex+".bin")
            with path.open("rb") as src,fresh.open("xb") as out:
                remaining=offset
                while remaining:
                    b=src.read(min(1024**2,remaining));out.write(b);remaining-=len(b)
            path=fresh
        if prior.get("status")=="complete" and offset==expected_bytes:
            if sha256(path)!=prior["sha256"]:raise ValueError("已完成下载SHA改变")
            if expected_sha and prior['sha256']!=expected_sha:raise ValueError("已完成下载与来源SHA不符")
            return dict(prior,file=str(path.resolve()))
        if offset==expected_bytes:
            # 最后一段写入后中断时，只核验完整文件，不请求无效的末尾Range。
            actual=sha256(path)
            if expected_sha and actual!=expected_sha:raise ValueError("完整原始文件SHA与来源不符")
            record=dict(prior,file=path.name,status='complete',sha256=actual,upstream_sha_verified=bool(expected_sha))
            save_json(dest/f"progress_{len(list(dest.glob('progress_*.json'))):08d}.json",record)
            return dict(record,file=str(path.resolve()))
    else:path=dest/("payload_"+uuid.uuid4().hex+".bin")
    if time.monotonic()>=deadline:raise DownloadStopped("budget_stopped","下载前预算已到")
    space_check()
    headers={"User-Agent":"Jeon2019-M3-academic-reproduction/1.0","Accept-Encoding":"identity"}
    if offset:headers["Range"]=f"bytes={offset}-"
    request=urllib.request.Request(url,headers=headers)
    with urllib.request.urlopen(request,timeout=45) as response:
        if offset and (response.status!=206 or not response.headers.get("Content-Range","").startswith(f"bytes {offset}-")):
            raise DownloadStopped("dependency_pending","源不支持可靠Range续接，未重复下载全包")
        identity={"etag":response.headers.get("ETag"),"last_modified":response.headers.get("Last-Modified")}
        if prior and prior["source_identity"]!=identity:raise ValueError("远端版本变化，拒绝混接")
        with path.open("ab" if offset else "xb") as stream:
            while offset<expected_bytes:
                if time.monotonic()>=deadline:raise DownloadStopped("budget_stopped","预算已到，保留完整分段断点")
                space_check()
                count=min(64*1024**2,expected_bytes-offset);digest=hashlib.sha256();written=0
                while written<count:
                    if time.monotonic()>=deadline:raise DownloadStopped("budget_stopped","分段下载预算已到，保留失败尾部")
                    value=response.read(min(1024**2,count-written))
                    if not value:raise DownloadStopped("dependency_pending","下载提前EOF，保留已登记断点及失败尾部")
                    stream.write(value);digest.update(value);written+=len(value)
                stream.flush();os.fsync(stream.fileno())
                segment_records.append({"offset":offset,"bytes":written,"sha256":digest.hexdigest()});offset+=written
                record={"url":url,"file":path.name,"expected_bytes":expected_bytes,"source_identity":identity,
                        "segments":segment_records,"completed_bytes":offset,"status":"partial"}
                save_json(dest/f"progress_{len(list(dest.glob('progress_*.json'))):08d}.json",record)
                print(f"下载 {dest.name} {offset}/{expected_bytes}",flush=True)
            if response.read(1):raise ValueError("实际下载超过官方登记大小")
    actual=sha256(path)
    if expected_sha and actual!=expected_sha:raise ValueError("完整原始文件SHA与来源不符")
    record.update(status="complete",sha256=actual,upstream_sha_verified=bool(expected_sha))
    save_json(dest/f"progress_{len(list(dest.glob('progress_*.json'))):08d}.json",record)
    return dict(record,file=str(path.resolve()))
