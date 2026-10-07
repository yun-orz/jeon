"""M3正式数据流水线，未取得真实数据绝不生成passed/最终增益/假场景。"""
import copy
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
import numpy as np
from audit import HERE,ROOT,read_json,safe_path,sha256,verify_manifest
from m3_common import save_json,fingerprint,canonical,verify_parent
from m3_acquire import estimate,existing_bytes,run as acquire
from m3_download import DownloadStopped
from m3_readers import Harvard,ICVL,KAIST,inspect_scene,preprocess
from m3_index import scene_groups,split_groups,normalization_scale,build_index,fit_gain,BoundedReader


def reader_for(record):
    if record['dataset']=='harvard':return Harvard(record['raw_file'],record['calibration_file'],record['readme_file'])
    if record['dataset']=='icvl':return ICVL(record['raw_file'])
    if record['dataset']=='kaist':return KAIST(record['raw_file'])
    raise ValueError('未支持数据集')


def source_records(data_dir,review_dir,local_icvl=None,icvl_provenance=None):
    """实际完整下载/已核验本地文件才入场景清单，不从文件名造数据。"""
    records=[];availability={'harvard_archives':0,'kaist_files':0,'icvl_files':0,'icvl_authorization_pending':not bool(local_icvl)}
    for collection in ['CZ_hsdb','CZ_hsdbi']:
        directory=data_dir/'downloads/harvard'/collection
        logs=sorted(directory.glob('progress_*.json'))
        if not logs or read_json(logs[-1]).get('status')!='complete':continue
        raw=data_dir/'raw/harvard'/collection
        if (raw/'extraction_complete.json').exists():availability['harvard_archives']+=1
        existing=[read_json(p) for p in raw.rglob('*.source.json')]
        # 同归档成员有失败旧尝试时，以最新完整登记为准，原文件仍保留。
        members={r['member']:r for r in existing}
        calibration=[r for r in members.values() if Path(r['member']).name.lower()=='calib.txt']
        readmes=[r for r in members.values() if Path(r['member']).name.lower()=='readme.txt']
        if len(calibration)!=1 or len(readmes)!=1:continue
        for item in members.values():
            path=Path(item['file'])
            if path.suffix!='.mat':continue
            if sha256(path)!=item['sha256']:raise ValueError('Harvard原始源SHA变化')
            source_files=[item,calibration[0],readmes[0]]
            for source in source_files:
                if sha256(source['file'])!=source['sha256']:raise ValueError('Harvard标定/说明源SHA变化')
            records.append({'dataset':'harvard','scene_id':f'harvard:{collection}:{path.stem}','stem':path.stem,
                            'raw_file':str(path),'raw_sha256':item['sha256'],'source_files':[r['file'] for r in source_files],
                            'source_sha256':{r['file']:r['sha256'] for r in source_files},'calibration_file':calibration[0]['file'],
                            'readme_file':readmes[0]['file'],'legacy_diagnostic':path.stem in ['img3','img4','img5'],
                            'source_collection':collection})
    for directory in sorted((data_dir/'downloads/kaist').glob('*')):
        logs=sorted(directory.glob('progress_*.json'))
        if not logs:continue
        last=read_json(logs[-1])
        if last.get('status')!='complete':continue
        path=directory/last['file']
        if sha256(path)!=last['sha256']:raise ValueError('KAIST原始EXR SHA变化')
        stem=directory.name
        records.append({'dataset':'kaist','scene_id':'kaist:'+stem,'stem':stem,'raw_file':str(path.resolve()),
                        'raw_sha256':last['sha256'],'source_files':[str(path.resolve())],'source_sha256':{str(path.resolve()):last['sha256']}})
        availability['kaist_files']+=1
    catalog={Path(r['path']).name:r for r in read_json(review_dir/'icvl_catalog.txt')}
    if local_icvl:
        pinned=None
        if icvl_provenance:
            from m3_icvl_provenance import load_pinned_inventory
            pinned=load_pinned_inventory({key:safe_path(value) for key,value in icvl_provenance.items()})
        directory=Path(local_icvl).resolve()
        if not directory.is_dir():raise ValueError('指定ICVL本地目录不存在')
        for path in sorted(directory.rglob('*.mat')):
            if path.name not in catalog:raise ValueError('ICVL文件不在官方公开MAT目录：'+path.name)
            if path.stat().st_size!=catalog[path.name]['size']:raise ValueError('ICVL文件长度与官方目录不符：'+path.name)
            actual=sha256(path)
            if pinned and (path.name not in pinned or actual!=pinned[path.name]['sha256']):raise ValueError('ICVL实际文件SHA与官方固定目录不符：'+path.name)
            records.append({'dataset':'icvl','scene_id':'icvl:'+path.stem,'stem':path.stem,'raw_file':str(path),
                            'raw_sha256':actual,'source_files':[str(path)],'source_sha256':{str(path):actual},
                            'catalogue_file':catalog[path.name]['path'],'upstream_sha_unavailable':not bool(pinned),
                            'source_revision':pinned[path.name]['revision'] if pinned else None,
                            'provenance':'已授权官方固定版本，实际SHA逐文件匹配仓库LFS摘要' if pinned else '用户提供合法官方MAT；文件名/大小核对官方公开目录，实际内容SHA冻结，未冒称来源端SHA已取得'})
            availability['icvl_files']+=1
    if len({r['scene_id'] for r in records})!=len(records):raise ValueError('场景ID重复')
    return records,availability


def write_index(path,index):
    with Path(path).open('xb') as stream:
        for row in index:stream.write(canonical(row)+b'\n')
    return sha256(path)


def final_packages(parents,m2_dir,dest,data_fingerprint,gain,gain_audit):
    """创建新最终包，父M2文件不修改；数据版本与共同增益显式绑定。"""
    result={}
    for method,parent in parents.items():
        folder=dest/('operator_'+method);folder.mkdir()
        package=copy.deepcopy(parent)
        source=parent['sources']['response']
        names={parent['kernels_native'],parent['kernels_training'],parent['spectral_response'],parent['mapping']['file'],source['raw_file'],source['license_file']}
        for name in names:shutil.copyfile(m2_dir/name,folder/name)
        package.update(schema='jeon2019-operator-final-v1',version=2,parent_fingerprint=parent['fingerprint'],data_fingerprint=data_fingerprint)
        package['measurement_gain']={'status':'fitted','value':gain,'fit_scope':'M3仅训练集，两器件共同单一增益','data_fingerprint':data_fingerprint,'audit_sha256':sha256(gain_audit)}
        package['audit'].update(final_training_ready=True,gain_fit_passed=True)
        package['sources']['parent_m2_run']=str(m2_dir)
        package['fingerprint']=fingerprint({k:v for k,v in package.items() if k!='fingerprint'})
        save_json(folder/'package.json',package);result[method]={'file':str(folder/'package.json'),'fingerprint':package['fingerprint']}
    return result


def run(config_path,phase='all',resume=None):
    from main import load_m0
    index,_=load_m0();config=read_json(config_path)
    if config['schema']!='jeon2019-m3-config-v1' or not 0<config['budget_seconds']<=28800:raise ValueError('M3配置/预算不符')
    if not 0<config['cache_max_bytes']<=512*1024**2:raise ValueError('缓存超过512MiB')
    if config['acquisition_mode'] not in ['existing','download']:raise ValueError('下载模式不符')
    data_dir=safe_path(config['data_dir']);review_dir=safe_path(config['source_review']);m2_dir=safe_path(config['m2_run'])
    bridge=safe_path(config['parent_bridge']);protocol=read_json(safe_path(config['protocol']))
    if protocol['seed']!=20261006 or protocol['patch_count']!=30000 or protocol['patch_size']!=256:raise ValueError('预登记协议与M0不符')
    base=ROOT/'outputs/jeon2019_optics/results/reproduction_v1/m3'
    dest=base/('run_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);dest.mkdir(parents=True)
    command=[sys.executable,str(HERE/'main.py'),'data','--config',str(Path(config_path).resolve()),'--phase',phase]
    if resume:command+=['--resume',str(Path(resume).resolve())]
    source_paths=list(HERE.glob('m3_*.py'))+[HERE/'main.py',Path(config_path).resolve(),safe_path(config['protocol']),bridge,HERE/'contracts.json']
    source_paths += [safe_path(value) for value in config.get('icvl_provenance',{}).values()]
    inputs={p.relative_to(ROOT).as_posix():sha256(p) for p in source_paths}
    if resume:
        prior=read_json(Path(resume)/'audit.json')
        # 未冻结的获取路径可以改变；物理包与处理协议必须保持一致。
        if prior['effective_config']['protocol']!=protocol or prior['effective_config']['config']['m2_run']!=config['m2_run']:raise ValueError('续接父包/预登记协议变化')
    report={'milestone':'M3','status':'dependency_pending','commands':[command],
            'effective_config':{'config':config,'protocol':protocol,'phase':phase},'input_fingerprints':inputs,'output_fingerprints':{},
            'key_numbers':{},'failures':[],'acceptance':{'all_passed':False},
            'resume_command':command[:-2]+['--resume',str(dest)] if resume else command+['--resume',str(dest)],
            'next_inputs':['全部验收通过后M4；目前不允许进入M4'],'paper_alignment_passed':False,'run_dir':str(dest),
            'scene_progress':[],'measurement_gain':{'status':'pending','value':None},'test_seal_status':'not_finalized'}
    save_json(dest/'preregistered_protocol.json',protocol)
    deadline=time.monotonic()+config['budget_seconds'];start=time.monotonic()
    def guard():
        if time.monotonic()>=deadline:raise DownloadStopped('budget_stopped','M3预算已到，保留场景进度')
        if shutil.disk_usage(ROOT).free<8*1024**3+512*1024**2:raise DownloadStopped('space_stopped','checkpoint/cache保留空间不足')
    try:
        parents=verify_parent(m2_dir,bridge)
        protection=verify_manifest(read_json(safe_path(index['protection'])))
        if not protection['passed']:raise ValueError('M0历史保护失败')
        report['historical_protection']=protection
        present=existing_bytes(data_dir) if data_dir.exists() else 0
        budget,_=estimate(review_dir,present);report['space']=budget
        if not budget['passed']:raise DownloadStopped('space_stopped','完整数据/预处理/checkpoint峰值预算不满足')
        if phase=='download' or config['acquisition_mode']=='download':
            report['acquisition']=acquire(review_dir,data_dir,max(1,deadline-time.monotonic()))
            if phase=='download':raise DownloadStopped(report['acquisition']['status'],'独立下载阶段完成或暂停；尚未进行全部M3验收')
        records,availability=source_records(data_dir,review_dir,config['local_icvl_dir'],config.get('icvl_provenance'))
        report['availability']=availability
        save_json(dest/'raw_source_records.json',records)
        if not records:raise DownloadStopped('dependency_pending','没有已完整取得并校验的正式场景；官方下载/ICVL授权尚未完成')
        descriptors={};reader_version=sha256(HERE/'m3_readers.py')
        for row in records:
            guard()
            cache=data_dir/'metadata'/fingerprint({'raw':row['source_sha256'],'reader':reader_version})
            if (cache/'record.json').exists():
                stored=read_json(cache/'record.json')
                descriptor=np.load(cache/'descriptor.npy')
                if sha256(cache/'descriptor.npy')!=stored['descriptor_sha256']:raise ValueError('重复元数据SHA变化')
                row['metadata']=stored['metadata']
            else:
                if cache.exists() and any(cache.iterdir()):cache=cache/('attempt_'+uuid.uuid4().hex[:8])
                cache.mkdir(parents=True,exist_ok=True);reader=reader_for(row)
                try:descriptor,meta=inspect_scene(reader)
                finally:reader.close()
                np.save(cache/'descriptor.npy',descriptor)
                save_json(cache/'record.json',{'metadata':meta,'source_sha256':row['source_sha256'],'reader_sha256':reader_version,'descriptor_sha256':sha256(cache/'descriptor.npy')})
                row['metadata']=meta
            descriptors[row['scene_id']]=descriptor
            report['scene_progress'].append({'scene_id':row['scene_id'],'metadata':'passed','raw_sha256':row['raw_sha256']})
            save_json(dest/f'progress_{len(report["scene_progress"]):06d}.json',report)
            print('元数据已核对',row['scene_id'],flush=True)
        if phase=='metadata':raise DownloadStopped('dependency_pending','元数据阶段结束，未执行划分/拟合/冻结')
        grouped,reasons=scene_groups(records,descriptors)
        targets=read_json(HERE/'protocol.json')['data']['targets']
        selected,counts=split_groups(grouped,targets,protocol['seed'])
        report['key_numbers']['counts']=counts
        save_json(dest/'group_audit.json',{'source_records':grouped,'relationships':reasons,'counts':counts,'group_crossing':False,'known_legacy_policy':protocol['legacy_policy']})
        if phase=='split':raise DownloadStopped('dependency_pending','划分阶段结束，未执行尺度/共同增益拟合或正式测试封条')
        # 全部来源到齐后才处理训练/验证，避免暂定划分变化将已处理场景归入封存测试。
        ready=availability['harvard_archives']==2 and availability['kaist_files']==30 and availability['icvl_files']>=150
        report['acceptance']['source_acquisition_complete']=ready
        if not ready:raise DownloadStopped('dependency_pending','全来源尚未到齐；只审核元数据及暂定组，不预处理，防止后续划分变化污染测试封存')
        for row in selected:
            guard()
            if row['split']=='test':
                row.update(data_file=None,mask_file=None,preprocessing={'status':'deferred_to_M7','reason':protocol['test_policy']})
                continue
            folder=data_dir/'preprocessed'/fingerprint({'source':row['source_sha256'],'reader':reader_version})
            if (folder/'record.json').exists():
                old=read_json(folder/'record.json')
                for name in ['base.npy','mask.npy']:
                    if sha256(folder/name)!=old['sha256'][name]:raise ValueError('半尺寸预处理SHA变化')
                prep=old['preprocessing']
            else:
                if folder.exists() and any(folder.iterdir()):folder=folder/('attempt_'+uuid.uuid4().hex[:8])
                folder.mkdir(parents=True,exist_ok=True);reader=reader_for(row)
                try:prep=preprocess(reader,folder/'base.npy',folder/'mask.npy')
                finally:reader.close()
                save_json(folder/'record.json',{'source_sha256':row['source_sha256'],'preprocessing':prep,'sha256':{name:sha256(folder/name) for name in ['base.npy','mask.npy']}})
            row.update(data_file=str(folder/'base.npy'),mask_file=str(folder/'mask.npy'),preprocessing=prep,preprocessed_sha256=read_json(folder/'record.json')['sha256'])
            print('半尺寸已生成',row['scene_id'],row['split'],flush=True)
        save_json(dest/'prepared_scenes.json',selected)
        ready=availability['harvard_archives']==2 and availability['kaist_files']==30 and availability['icvl_files']>=150
        report['acceptance']['source_acquisition_complete']=ready
        if phase=='preprocess':raise DownloadStopped('dependency_pending','预处理阶段结束，未执行训练尺度/共同增益/最终冻结')
        if not ready:
            raise DownloadStopped('dependency_pending','真实来源未完整取得：ICVL需要官方授权后提供不少于150幅官方MAT，Harvard两归档及KAIST30幅也须取得')
        if not all(counts[d]['train']>0 and counts[d]['validation']>0 for d in targets) or counts['kaist']['test']!=10:
            raise DownloadStopped('dependency_pending','分组优先后缺训练/验证或10幅封存测试；不从测试补足')
        scales,scale_audit=normalization_scale(selected)
        save_json(dest/'normalization_audit.json',{'preregistered_protocol_sha256':sha256(dest/'preregistered_protocol.json'),'scales':scales,'datasets':scale_audit})
        guard();patches=build_index(selected,scales,protocol['seed'],30000)
        first=write_index(dest/'patch_index.jsonl',patches)
        replay=build_index(selected,scales,protocol['seed'],30000)
        second=write_index(dest/'patch_index_replay.jsonl',replay)
        if first!=second:raise ValueError('同种子3万条索引SHA不能复现')
        report['key_numbers']['index']={'count':len(patches),'sha256':first,'replay_sha256':second,'copied_patch_files':0}
        if phase=='index':raise DownloadStopped('dependency_pending','索引阶段完成；共同增益和最终封存还未审核')
        kernels={m:np.load(m2_dir/p['kernels_training']) for m,p in parents.items()}
        response=np.load(m2_dir/parents['jeon']['spectral_response']);weights=parents['jeon']['spectral_weights']['values']
        gain,gain_audit=fit_gain(selected,scales,patches,kernels,response,weights,protocol['gain_target_rms'])
        save_json(dest/'gain_audit.json',gain_audit)
        core={'schema':'jeon2019-m3-data-version-v1','version':config['data_version'],'seed':protocol['seed'],'protocol':protocol,
              'preprocessed_sha256':{r['scene_id']:r['preprocessed_sha256'] for r in selected if r['split']!='test'},
              'sources':[{'scene_id':r['scene_id'],'scene_group_id':r['scene_group_id'],'split':r['split'],'source_sha256':{Path(k).name:v for k,v in r['source_sha256'].items()},'source_revision':r.get('source_revision'),'metadata':r['metadata'],'preprocessing':r['preprocessing']} for r in selected],
              'normalization_training_fit':scale_audit,'patch_index_sha256':first,'parent_operator_fingerprints':{m:p['fingerprint'] for m,p in parents.items()}}
        core['fingerprint']=fingerprint(core);save_json(dest/'data_version.json',core)
        operators=final_packages(parents,m2_dir,dest,core['fingerprint'],gain,dest/'gain_audit.json')
        manifests=[]
        for row in selected:
            record={'schema':'jeon2019-data-scene-v1','version':config['data_version'],'dataset':row['dataset'],'scene_id':row['scene_id'],
                    'scene_group_id':row['scene_group_id'],'source_files':row['source_files'],'source_sha256':row['source_sha256'],
                    'wavelength_mapping':row['metadata'],'radiometric_units':row['metadata']['units'],'valid_mask':{'file':row['mask_file'],'rule':protocol['valid_mask'],'test_deferred':row['split']=='test'},
                    'calibration':row.get('calibration_file','官方发布rad/reflectance，未额外逐谱带校正'),'preprocessing_version':{'version':'m3-energy-area-v1','reader_sha256':reader_version},
                    'split':row['split'],'normalization_training_fit':{'dataset_scale':scales[row['dataset']],'audit':str(dest/'normalization_audit.json'),'training_only':True},
                    'base_resolution':{'scale':.5,'pixel_pitch_m':12.44e-6,'data_file':row['data_file'],'details':row['preprocessing'],'sha256':row.get('preprocessed_sha256')},
                    'augmentation_scales':[.5,1.,2.],'patch_index':{'file':str(dest/'patch_index.jsonl'),'sha256':first,'train_only':True},
                    'operator_fingerprint':{m:v['fingerprint'] for m,v in operators.items()},'data_version_fingerprint':core['fingerprint']}
            required=read_json(HERE/'contracts.json')['data_manifest']['required']
            if not set(required)-{'fingerprint'}<=set(record):raise ValueError('场景清单合同缺字段')
            record['fingerprint']=fingerprint(record);manifests.append(record)
        save_json(dest/'scene_manifest.json',manifests)
        test=[r for r in manifests if r['split']=='test']
        seal={'schema':'jeon2019-test-seal-v1','open_milestone':'M7','test_scene_count':len(test),'scene_ids':[r['scene_id'] for r in test],
              'scene_groups':[r['scene_group_id'] for r in test],'source_sha256':{r['scene_id']:r['source_sha256'] for r in test},
              'data_version_fingerprint':core['fingerprint'],'operator_fingerprints':{m:v['fingerprint'] for m,v in operators.items()},
              'prohibited_before_M7':['预测','成绩','ROI选择','尺度/增益拟合','测试patch读取'],'preprocessed_before_M7':False}
        seal['fingerprint']=fingerprint(seal);save_json(dest/'test_seal.json',seal)
        measured=BoundedReader(selected,scales,config['cache_max_bytes'])
        for row in patches[:64]:measured.read(row['scene_id'],row['scale'],row['y'],row['x'])
        save_json(dest/'cache_audit.json',measured.metrics())
        from m3_validate import validate_artifacts
        independent=validate_artifacts(dest,parents,operators)
        save_json(dest/'independent_audit.json',independent)
        report['measurement_gain']={'status':'fitted','value':gain,'data_fingerprint':core['fingerprint']}
        report['operators']=operators;report['test_seal_status']='sealed_until_M7'
        report['output_fingerprints']={'data':core['fingerprint'],'test_seal':seal['fingerprint'],**{m:v['fingerprint'] for m,v in operators.items()}}
        report['acceptance']={'all_passed':True,'sources_acquired':True,'groups_disjoint':True,'index_reproducible':True,'cache_bound_measured':True,'test_sealed':True,'common_gain_fitted_training_only':True}
        report['status']='passed';report['next_inputs']=['M4：最终算子包、数据版本、训练索引及封存测试清单']
    except DownloadStopped as exc:
        report['status']=exc.status;report['failures'].append({'reason':str(exc)})
    except Exception as exc:
        report['status']='failed';report['failures'].append({'error':repr(exc)})
    report['elapsed_seconds']=time.monotonic()-start
    report['unfinished']=[k for k,v in report['acceptance'].items() if not v]
    if report['status']!='passed':report['unfinished']+=['全部真实来源验收','仅训练尺度/共同增益正式冻结','最终算子/数据指纹','不可变测试封条','完整独立审核及main.py check']
    save_json(dest/'audit.json',report)
    save_json(dest/'result_files.json',{p.name:sha256(p) for p in dest.iterdir() if p.is_file()})
    lines=['# M3 数据流水线运行报告','',f"状态：{report['status']}。未全部通过前不进入M4。",'',
           '实际执行命令：','','```powershell','& '+' '.join('"'+str(s)+'"' for s in command),'```','',
           '续接命令（写新运行目录、保留已有源数据和场景进度）：','','```powershell','& '+' '.join('"'+str(s)+'"' for s in report['resume_command']),'```','',
           '原始文件列表见raw_source_records.json，来源页面/实际HEAD尺寸见source_review目录；预登记估计器见preregistered_protocol.json。',
           '分组/真实规模差异、每场景状态、失败及未完成事项见audit.json/group_audit.json。未取得数据不能用fixture代替，不生成假scene或passed。',
           '测试只允许来源/通道/完整性/重复元数据核对，半尺寸预处理推迟M7。Harvard历史img3/img4/img5及其关联组明确排除正式训练/验证/新测试。',
           '校正为ref/相对灵敏度，不作λ/540光子换算；ICVL按实际bands，KAIST按w{波长}nm名称并保持反射率。跨数据集尺度与照明谱代理均为实施假设，不是绝对标定。',
           'M2原包保持不变；最终包须带父指纹和数据版本，未拟合前共同增益保持pending。','','失败/缺口：',json.dumps(report['failures'],ensure_ascii=False,indent=2)]
    (dest/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def check_run(path):
    """检查正式M3结果；pending仅报告一致的未完成状态，不冒称验收通过。"""
    dest=Path(path).resolve();audit=read_json(dest/'audit.json');failures=[]
    for name,digest in audit['input_fingerprints'].items():
        if sha256(safe_path(name))!=digest:failures.append('M3源码/协议/配置改变：'+name)
    config=audit['effective_config']['config']
    verify_parent(safe_path(config['m2_run']),safe_path(config['parent_bridge']))
    inventory=read_json(dest/'result_files.json')
    for name,digest in inventory.items():
        if sha256(dest/name)!=digest:failures.append('M3结果内容SHA改变：'+name)
    if audit['status']!='passed':
        return {'status':audit['status'] if not failures else 'failed','acceptance_passed':False,'run_dir':str(dest),'failures':failures,'unfinished':audit['unfinished'],'M4_ready':False}
    from m3_validate import validate_artifacts
    try:
        validate_artifacts(dest,verify_parent(safe_path(config['m2_run']),safe_path(config['parent_bridge'])),audit['operators'])
    except Exception as exc:
        failures.append(str(exc))
    core=read_json(dest/'data_version.json');body={k:v for k,v in core.items() if k!='fingerprint'}
    if core['fingerprint']!=fingerprint(body):failures.append('数据版本指纹改变')
    scenes=read_json(dest/'scene_manifest.json');groups={}
    for row in scenes:
        required=read_json(HERE/'contracts.json')['data_manifest']['required']
        if not set(required)<=set(row) or row['fingerprint']!=fingerprint({k:v for k,v in row.items() if k!='fingerprint'}):failures.append('场景合同或指纹错误')
        groups.setdefault(row['scene_group_id'],set()).add(row['split'])
        for file,digest in row['source_sha256'].items():
            if sha256(file)!=digest:failures.append('原始场景源SHA改变')
    if any(len(v)>1 for v in groups.values()):failures.append('场景组跨集')
    seal=read_json(dest/'test_seal.json')
    if seal['fingerprint']!=fingerprint({k:v for k,v in seal.items() if k!='fingerprint'}) or seal['test_scene_count']!=10:failures.append('测试封条错误')
    if sha256(dest/'patch_index.jsonl')!=sha256(dest/'patch_index_replay.jsonl'):failures.append('同种子索引不一致')
    patch_rows=[json.loads(line) for line in (dest/'patch_index.jsonl').read_text(encoding='utf-8').splitlines()]
    training={r['scene_id'] for r in scenes if r['split']=='train'}
    if len(patch_rows)!=30000 or not {r['scene_id'] for r in patch_rows}<=training:failures.append('索引数量或训练归属错误')
    gain=read_json(dest/'gain_audit.json')['value'];gains=[]
    for method,info in audit['operators'].items():
        p=read_json(info['file']);gains.append(p['measurement_gain']['value'])
        if p['fingerprint']!=fingerprint({k:v for k,v in p.items() if k!='fingerprint'}) or p['data_fingerprint']!=core['fingerprint']:failures.append('最终算子绑定错误')
        parent=read_json(safe_path(config['m2_run'])/(method+'_package.json'))
        if p['parent_fingerprint']!=parent['fingerprint'] or p['kernel_sha256']!=parent['kernel_sha256']:failures.append('物理父包改变')
        for key,kind in [('kernels_native','native'),('kernels_training','training')]:
            if sha256(Path(info['file']).parent/p[key])!=p['kernel_sha256'][kind]:failures.append('最终核SHA改变')
    if any(v!=gain for v in gains) or not np.isfinite(gain) or gain<=0:failures.append('共同增益错误')
    cache=read_json(dest/'cache_audit.json')
    if cache['peak_cache_bytes']>512*1024**2 or cache['observed_process_rss_bytes']<=0:failures.append('真实缓存审核不符')
    return {'status':'passed' if not failures else 'failed','acceptance_passed':not failures,'run_dir':str(dest),'failures':failures,'M4_ready':not failures}
