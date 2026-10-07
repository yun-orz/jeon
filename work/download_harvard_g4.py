# -*- coding: utf-8 -*-
"""单次流式读取官方归档前两幅MAT及说明；不展开未知路径、不删文件。"""
from pathlib import Path
import hashlib
import json
import tarfile
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/datasets/harvard_g4_20261006'
URL='https://vision.seas.harvard.edu/hyperspec/d2x5g3/CZ_hsdbi.tgz'

class LimitedReader:
    def __init__(self,response):self.response=response;self.count=0;self.start=time.monotonic()
    def read(self,n=-1):
        if self.count>550*1024**2 or time.monotonic()-self.start>180:raise RuntimeError('到达本批下载字节或时间预算')
        block=self.response.read(n);self.count+=len(block);return block

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.iterdir()):raise ValueError('下载目标非空，拒绝覆盖或重复下载')
    records=[];members_seen=[];request=urllib.request.Request(URL,headers={'User-Agent':'Jeon2019-academic-reproduction/1.0'})
    with urllib.request.urlopen(request,timeout=30) as response:
        reader=LimitedReader(response)
        with tarfile.open(fileobj=reader,mode='r|gz') as archive:
            for member in archive:
                members_seen.append(member.name)
                name=Path(member.name).name
                selected=member.isfile() and (name.lower().endswith('.mat') or 'readme' in name.lower() or name.lower()=='calib.txt')
                if not selected:continue
                if member.size>250*1024**2:raise ValueError('单文件超过本批预算')
                destination=OUT/name
                if destination.exists():raise ValueError('重复归档文件名，拒绝覆盖')
                source=archive.extractfile(member);digest=hashlib.sha256();count=0
                with destination.open('xb') as stream:
                    while True:
                        block=source.read(1024*1024)
                        if not block:break
                        stream.write(block);digest.update(block);count+=len(block)
                assert count==member.size
                records.append(dict(member=member.name,local_name=name,bytes=count,sha256=digest.hexdigest()))
                print('已保存：',name,count,flush=True)
                if sum(r['local_name'].lower().endswith('.mat') for r in records)==2:break
        downloaded=reader.count
    result=dict(url=URL,date='2026-10-06',method='一次请求流式读取前两幅MAT后关闭，不保存完整归档',compressed_bytes_read=downloaded,
        files=records,members_seen=members_seen,license_page='https://vision.seas.harvard.edu/hyperspec/download.html',usage='non-commercial academic research')
    with (OUT/'download_manifest.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
