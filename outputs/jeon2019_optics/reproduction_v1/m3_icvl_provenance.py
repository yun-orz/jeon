"""核对用户完成授权后取得的固定官方ICVL目录、清单和真实远端摘要。"""
import re
from pathlib import Path
from audit import read_json


def load_pinned_inventory(provenance):
    source=read_json(provenance['source'])
    manifest=read_json(provenance['manifest'])
    tree=read_json(provenance['tree'])
    revision=source['resolved_revision']
    if not re.fullmatch('[0-9a-f]{40}',revision) or manifest['revision']!=revision:
        raise ValueError('ICVL必须对应相同固定官方commit')
    if source['repo_id']!='ICVL-BGU/ICVL_HS_2016' or manifest['repo_id']!=source['repo_id']:
        raise ValueError('ICVL官方仓库标识不符')
    remote={r['path']:r for r in tree if r['type']=='file' and r['path'].startswith('mat/') and r['path'].endswith('.mat')}
    if len(remote)!=202 or len(manifest['files'])!=202:raise ValueError('ICVL完整交接目录应含202幅MAT')
    result={}
    for row in manifest['files']:
        path=row['relative_path'];entry=remote[path];digest=entry['lfs']['oid']
        if not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('ICVL远端SHA缺失或被星号遮蔽')
        if row['sha256']!=row['expected_sha256'] or row['sha256']!=digest:
            raise ValueError('ICVL交接SHA与固定目录远端摘要不符：'+path)
        if row['size_bytes']!=entry['size'] or row['size_bytes']!=row['expected_size_bytes']:
            raise ValueError('ICVL固定目录文件长度不符：'+path)
        name=Path(path).name
        if name in result:raise ValueError('ICVL交接文件名重复')
        result[name]={'sha256':digest,'bytes':entry['size'],'revision':revision,'remote_path':path}
    return result
