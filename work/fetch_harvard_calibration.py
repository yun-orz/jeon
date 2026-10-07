# -*- coding: utf-8 -*-
"""只补读官方归档开头的校正向量，不重复下载两幅MAT。"""
from pathlib import Path
import hashlib
import json
import tarfile
import urllib.request

OUT=Path(__file__).resolve().parent/'datasets/harvard_g4_20261006'
URL='https://vision.seas.harvard.edu/hyperspec/d2x5g3/CZ_hsdbi.tgz'

class Prefix:
    def __init__(self,response):self.response=response;self.count=0
    def read(self,n=-1):
        if self.count>1024*1024:raise RuntimeError('补读前缀超过1MiB预算')
        result=self.response.read(n);self.count+=len(result);return result

if __name__=='__main__':
    request=urllib.request.Request(URL,headers={'Range':'bytes=0-1048575','User-Agent':'Jeon2019-academic-reproduction/1.0'})
    with urllib.request.urlopen(request,timeout=30) as response:
        reader=Prefix(response)
        with tarfile.open(fileobj=reader,mode='r|gz') as archive:
            for member in archive:
                if member.isfile() and Path(member.name).name=='calib.txt':
                    if member.size>10000:raise ValueError('校正文件异常大')
                    value=archive.extractfile(member).read()
                    with (OUT/'calib.txt').open('xb') as stream:stream.write(value)
                    result=dict(url=URL,member=member.name,compressed_bytes_read=reader.count,bytes=len(value),sha256=hashlib.sha256(value).hexdigest(),date='2026-10-06')
                    break
            else:raise RuntimeError('未找到校正向量')
    with (OUT/'calibration_download_manifest.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
