"""M3来源预估与独立下载阶段；ICVL授权未取得时只登记真实缺口。"""
import json
import copy
import shutil
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path,PurePosixPath
from audit import HERE,ROOT,read_json,safe_path,sha256
from m3_common import save_json,space_budget
from m3_download import download,DownloadStopped


def estimate(review_dir,already_present=0):
    review=read_json(Path(review_dir)/'review.json')
    objects=review['downloads']
    if len([r for r in objects if r['dataset']=='harvard'])!=2 or len([r for r in objects if r['dataset']=='kaist'])!=30:
        raise ValueError('官方归档/EXR清单不完整')
    if any(not isinstance(r.get('bytes'),int) or r['bytes']<=0 for r in objects):raise ValueError('下载尺寸未核实，不批量下载')
    catalog=read_json(Path(review_dir)/'icvl_catalog.txt')
    if not catalog or any(not r['path'].startswith('mat/') or not r['path'].endswith('.mat') for r in catalog):raise ValueError('ICVL公开实际目录不符')
    harvard=sum(r['bytes'] for r in objects if r['dataset']=='harvard')
    kaist=sum(r['bytes'] for r in objects if r['dataset']=='kaist')
    icvl=sum(r['size'] for r in catalog)
    # Harvard实际旧样本元数据1040×1392；ICVL官方1392×1300；KAIST暂用保守3376×2704上界。
    processed=77*(1040//2)*(1392//2)*101+len(catalog)*(1392//2)*(1300//2)*101+30*(3376//2)*(2704//2)*101
    budget=space_budget(harvard+kaist+icvl,int(harvard*1.15),0,processed,already_present=already_present)
    budget['evidence']={'harvard_bytes':harvard,'kaist_bytes':kaist,'icvl_public_mat_count':len(catalog),'icvl_bytes':icvl,
                        'harvard_unpacked_upper_factor':1.15,'kaist_shape_bound':[2704,3376],
                        'decoded_mat_disk_bytes':0,'decoded_mat_reason':'MAT-v5按波段64列块流式解码，无整ref缓存文件；已用实际旧样本验证',
                        'processed_policy':'估计含全部候选；真正只预处理训练/验证，测试半尺寸推迟M7',
                        'remaining_unknowns':['新Harvard/KAIST实际形状须逐场景确认；超过预估即重新检查空间并停下']}
    return budget,objects


def extract_harvard(blob,destination,deadline,space_check):
    """只提取普通MAT/README/calib，逐文件校验；不接受路径逃逸或覆盖。"""
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    files=[];archive_sha=sha256(blob)
    complete=dest/'extraction_complete.json'
    if complete.exists():
        old=read_json(complete)
        if old['archive_sha256']!=archive_sha:raise ValueError('已提取归档SHA改变')
        for row in old['files']:
            if sha256(row['file'])!=row['sha256']:raise ValueError('已提取成员SHA改变')
        return old['files']
    with tarfile.open(blob,'r|gz') as archive:
        for member in archive:
            if time.monotonic()>=deadline:raise DownloadStopped('budget_stopped','解压预算到期')
            pure=PurePosixPath(member.name)
            if pure.is_absolute() or '..' in pure.parts or '\\' in member.name:raise ValueError('归档路径非法')
            if member.issym() or member.islnk():raise ValueError('归档链接不接受')
            name=pure.name
            if not member.isfile() or not (name.endswith('.mat') or name.lower() in ['readme.txt','calib.txt']):continue
            relative=Path(*pure.parts);target=dest/relative
            record=target.with_name(target.name+'.source.json')
            if record.exists():
                row=read_json(record)
                if not target.exists() or target.stat().st_size!=member.size or sha256(target)!=row['sha256']:raise ValueError('已提取源文件变化')
                files.append(row);continue
            if target.exists():
                # 失败文件保留，用新尝试目录保存完整对象，绝不覆盖或截断。
                target=dest/('attempt_'+str(time.time_ns()))/relative
                record=target.with_name(target.name+'.source.json')
            target.parent.mkdir(parents=True,exist_ok=True);space_check(member.size)
            source=archive.extractfile(member);count=0
            with target.open('xb') as stream:
                while True:
                    block=source.read(1024**2)
                    if not block:break
                    stream.write(block);count+=len(block)
            if count!=member.size:raise ValueError('归档成员长度不符')
            row={'file':str(target.resolve()),'member':member.name,'bytes':count,'sha256':sha256(target),'archive_sha256':archive_sha}
            save_json(record,row);files.append(row)
            print('Harvard已提取',name,count,flush=True)
    save_json(complete,{'archive_sha256':archive_sha,'files':files,'mat_count':sum(Path(r['file']).suffix=='.mat' for r in files),'complete_archive_stream':True})
    return files


def existing_bytes(data_dir):
    """仅扣除已登记的有效断点、原始文件和预处理，不把失败副本重复计入计划。"""
    data_dir=Path(data_dir);total=0
    for directory in (data_dir/'downloads').glob('*/*'):
        logs=sorted(directory.glob('progress_*.json'))
        if logs:total+=read_json(logs[-1])['completed_bytes']
    total+=sum(read_json(p)['bytes'] for p in (data_dir/'raw').rglob('*.source.json')) if (data_dir/'raw').exists() else 0
    for record in (data_dir/'preprocessed').rglob('record.json'):
        for name in ['base.npy','mask.npy']:
            if (record.parent/name).exists():total+=(record.parent/name).stat().st_size
    return total


def run(review_dir,data_dir,budget_seconds=28800,workers=3):
    review_dir=Path(review_dir).resolve();data_dir=Path(data_dir).resolve()
    data_dir.mkdir(parents=True,exist_ok=True)
    present=existing_bytes(data_dir)
    estimate_report,objects=estimate(review_dir,present)
    path=data_dir/('acquisition_'+str(time.time_ns())+'.json')
    status={'status':'dependency_pending','space':estimate_report,'datasets':{'harvard':[],'kaist':[],'icvl':[]},
            'dependency':'ICVL官方访问授权/合法本地原始MAT尚未提供','source_review':str(review_dir),'budget_seconds':budget_seconds}
    if not estimate_report['passed']:
        status['status']='space_stopped';save_json(path,status);return status
    deadline=time.monotonic()+budget_seconds
    def guard(extra=0):
        reserve=8*1024**3+512*1024**2+extra
        if shutil.disk_usage(ROOT).free<reserve:raise DownloadStopped('space_stopped','低于checkpoint/cache及本对象空间保留')
    lock=threading.Lock()
    def job(obj):
        try:
            dataset=obj['dataset'];stem=Path(obj['url']).stem
            folder=data_dir/'downloads'/dataset/stem
            record=download(obj['url'],folder,obj['bytes'],deadline,guard)
            record['stem']=stem
            if dataset=='harvard':
                files=extract_harvard(record['file'],data_dir/'raw/harvard'/stem,deadline,guard)
                record['extracted_files']=files
            with lock:
                status['datasets'][dataset].append(record)
                save_json(data_dir/('acquisition_progress_'+str(time.time_ns())+'.json'),copy.deepcopy(status))
        except DownloadStopped as exc:
            with lock:
                status['status']=exc.status;status.setdefault('errors',[]).append({'url':obj['url'],'error':str(exc)})
        except Exception as exc:
            with lock:status.setdefault('errors',[]).append({'url':obj['url'],'error':repr(exc)})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(job,obj) for obj in objects]
        for future in as_completed(futures):future.result()
    status['completed_source_objects']=sum(len(v) for v in status['datasets'].values())
    status['expected_source_objects']=32
    save_json(path,status)
    return status


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--data-dir',required=True);p.add_argument('--budget-seconds',type=float,default=28800);p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();result=run(args.review,args.data_dir,args.budget_seconds,args.workers)
    print(json.dumps(result,ensure_ascii=True,indent=2))
    raise SystemExit(0 if result['status']=='passed' else 2)
