# -*- coding: utf-8 -*-
"""一次流式请求取得一幅未参与G4的MAT；不修改已有数据、不覆盖文件。"""
from pathlib import Path
import hashlib
import json
import tarfile
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/datasets/harvard_g5_test_20261006_retry1'
OLD=ROOT/'work/datasets/harvard_g4_20261006'
URL='https://vision.seas.harvard.edu/hyperspec/d2x5g3/CZ_hsdbi.tgz'


class Reader:
    def __init__(self,response):self.response=response;self.count=0;self.start=time.monotonic();self.last=self.start
    def read(self,n=-1):
        if self.count>=384*1024**2 or time.monotonic()-self.start>300:raise RuntimeError('达到384MiB/300秒下载预算')
        if n<0:n=1024*1024
        n=min(n,384*1024**2-self.count)
        block=self.response.read(n);self.count+=len(block)
        if time.monotonic()-self.last>20:
            print(f'流式归档已读取 {self.count/1024**2:.1f}MiB',flush=True);self.last=time.monotonic()
        return block


def main():
    excluded={'img3.mat','img4.mat'}
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.iterdir()):raise ValueError('目标非空，拒绝覆盖或重复下载')
    # 校正及说明直接复用已审核数据的字节，不向服务器重复请求。
    reused=[]
    for name in ['calib.txt','README.txt']:
        value=(OLD/name).read_bytes()
        with (OUT/name).open('xb') as stream:stream.write(value)
        reused.append(dict(local_name=name,source=str(OLD/name),sha256=hashlib.sha256(value).hexdigest()))
    records=[];seen=[]
    try:
        request=urllib.request.Request(URL,headers={'User-Agent':'Jeon2019-academic-reproduction/1.0'})
        with urllib.request.urlopen(request,timeout=30) as response:
            reader=Reader(response)
            with tarfile.open(fileobj=reader,mode='r|gz') as archive:
                for member in archive:
                    seen.append(member.name);name=Path(member.name).name
                    if not member.isfile() or not name.endswith('.mat') or name in excluded:continue
                    if member.size>250*1024**2:raise ValueError('单MAT超过250MiB限制')
                    destination=OUT/name;digest=hashlib.sha256();count=0
                    with archive.extractfile(member) as source,destination.open('xb') as stream:
                        while True:
                            block=source.read(1024*1024)
                            if not block:break
                            stream.write(block);digest.update(block);count+=len(block)
                    if count!=member.size:raise ValueError('MAT长度核对失败')
                    records.append(dict(member=member.name,local_name=name,bytes=count,sha256=digest.hexdigest()))
                    print('新测试候选文件已保存：'+name,flush=True);break
            if len(records)!=1:raise ValueError('未找到新的MAT')
        report=dict(url=URL,date='2026-10-06',files=records,reused_files=reused,members_seen=seen,
            compressed_bytes_read=reader.count,excluded_scenes=sorted(excluded),
            method='一次读取归档到首个新MAT即停止；gzip流须经过已有前缀，但不再次保存旧MAT',
            license_page='https://vision.seas.harvard.edu/hyperspec/download.html',usage='non-commercial academic research')
        with (OUT/'download_manifest.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    except Exception as exc:
        with (OUT/'download_failed.json').open('x',encoding='utf-8') as stream:json.dump(dict(error=repr(exc),url=URL),stream,ensure_ascii=False,indent=2)
        raise


if __name__=='__main__':main()
