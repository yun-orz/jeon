"""正式M3外部独立审核；直接格式索引和标量计算，不读取测试patch。"""
import argparse
import collections
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
import numpy as np
from scipy.io import loadmat,whosmat
from scipy.signal import fftconvolve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import ROOT,HERE,read_json,sha256,safe_path
from m3_common import save_json,fingerprint
from m3_readers import matlab_ref_blocks
from m3_index import BoundedReader


def direct_rgb_points(image,kernels,coeff,points):
    """按每个核元素直接计算零边界卷积，独立于生产FFT路径。"""
    _,height,width=image.shape;kh,kw=kernels.shape[1:];values=[]
    for y,x in points:
        sy=y+kh//2-np.arange(kh);sx=x+kw//2-np.arange(kw)
        valid=((sy>=0)&(sy<height))[:,None]&((sx>=0)&(sx<width))[None,:]
        patch=np.asarray(image[:,np.clip(sy,0,height-1)[:,None],np.clip(sx,0,width-1)[None,:]],dtype=np.float64)
        blurred=np.sum(patch*kernels*valid[None],axis=(1,2),dtype=np.float64)
        values.append(coeff@blurred)
    return np.asarray(values).T


def independent_index_masks(prepared,index):
    """逐场景独立整数积分图核对全部索引的有效像元，不读取测试或光谱补丁。"""
    lookup={r['scene_id']:r for r in prepared if r['split']=='train'}
    grouped=collections.defaultdict(list)
    for row in index:grouped[(row['scene_id'],row['scale'])].append(row)
    if {scene for scene,_ in grouped}!=set(lookup):raise ValueError('冻结索引未覆盖全部训练场景')
    minimum=256**2;partial=0;checked=0
    for (scene,scale),rows in grouped.items():
        mask=np.load(lookup[scene]['mask_file'],mmap_mode='r');h,w=mask.shape
        if scale==1:valid=np.asarray(mask)
        elif scale==.5:valid=np.asarray(mask[:h//2*2,:w//2*2]).reshape(h//2,2,w//2,2).all((1,3))
        else:
            # 双倍像元中心的四个权重均非零；有效当且仅当四个邻点全部有效。
            y=np.floor((np.arange(h*2)-.5)/2).astype(int);x=np.floor((np.arange(w*2)-.5)/2).astype(int)
            y0=np.clip(y,0,h-1);y1=np.clip(y+1,0,h-1);x0=np.clip(x,0,w-1);x1=np.clip(x+1,0,w-1)
            valid=mask[y0[:,None],x0[None,:]]&mask[y0[:,None],x1[None,:]]&mask[y1[:,None],x0[None,:]]&mask[y1[:,None],x1[None,:]]
        integral=np.pad(valid.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
        for row in rows:
            y,x=row['y'],row['x'];count=int(integral[y+256,x+256]-integral[y,x+256]-integral[y+256,x]+integral[y,x])
            if count<=0:raise ValueError('冻结索引含全无效补丁')
            minimum=min(minimum,count);partial+=count<256**2;checked+=1
    return {'checked_indices':checked,'training_scenes_covered':len(lookup),'minimum_valid_pixels':minimum,'partially_masked_indices':partial,'test_masks_read':False}


def independent_augmented_pixel(data,mask,scale,y,x,normalization):
    """标量中心映射与四邻域计算，不调用生产增强函数。"""
    if scale==1:
        value=np.asarray(data[:,y,x]);valid=bool(mask[y,x])
    elif scale==.5:
        value=np.asarray(data[:,2*y:2*y+2,2*x:2*x+2],dtype=np.float64).mean((1,2))
        valid=bool(mask[2*y:2*y+2,2*x:2*x+2].all())
    elif scale==2:
        height,width=mask.shape
        py=np.clip((y+.5)/2-.5,0,height-1);px=np.clip((x+.5)/2-.5,0,width-1)
        ya,xa=int(np.floor(py)),int(np.floor(px));yb,xb=min(ya+1,height-1),min(xa+1,width-1)
        fy,fx=py-ya,px-xa
        value=np.zeros(25,dtype=np.float64);mask_value=0.
        for iy,ix,weight in [(ya,xa,(1-fy)*(1-fx)),(ya,xb,(1-fy)*fx),(yb,xa,fy*(1-fx)),(yb,xb,fy*fx)]:
            value+=np.asarray(data[:,iy,ix],dtype=np.float64)*weight;mask_value+=float(mask[iy,ix])*weight
        valid=mask_value>=1-1e-6
    else:raise ValueError('非法增强尺度')
    value=np.asarray(value,dtype=np.float32)/np.float32(normalization)
    if not valid:value[:]=0
    return value,valid


def independent_half_pixels(source,manifest):
    """用实际文件中的2×2像元核对基础图，仅允许训练和验证场景。"""
    if manifest['split']=='test':raise ValueError('独立审核也拒绝测试像元读取')
    data=np.load(manifest['base_resolution']['data_file'],mmap_mode='r')
    mask=np.load(manifest['valid_mask']['file'],mmap_mode='r');height,width=mask.shape
    points=[(0,0),(height//3,width//3),(height//2,width//2),(height-1,width-1)]
    values=np.zeros((25,len(points)),dtype=np.float64);valid=np.ones(len(points),dtype=bool)
    seen=np.zeros_like(values,dtype=bool)
    dataset=source['dataset']
    if dataset=='harvard':
        variables={n:s for n,s,k in whosmat(source['raw_file'],appendmat=False)}
        if variables['ref'][2]!=31:raise ValueError('Harvard实际谱轴不符')
        if (height,width)!=tuple(n//2 for n in variables['ref'][:2]) or manifest['wavelength_mapping']['selected_indices']!=list(range(25)):raise ValueError('Harvard半尺寸/谱索引不符')
        calibration=np.loadtxt(source['calibration_file'],dtype=np.float64).reshape(-1)
        labels=loadmat(source['raw_file'],variable_names=['lbl'],appendmat=False)['lbl']
        for i,(y,x) in enumerate(points):
            block=labels[2*y:2*y+2,2*x:2*x+2];valid[i]=bool(((block!=0)&np.isfinite(block)).all())
        for band,x,block in matlab_ref_blocks(source['raw_file'],list(range(25))):
            for i,(oy,ox) in enumerate(points):
                px=2*ox
                if x<=px and px+2<=x+block.shape[1]:
                    raw=np.asarray(block[2*oy:2*oy+2,px-x:px-x+2],dtype=np.float64)
                    good=np.isfinite(raw)&(raw>=0)
                    valid[i]&=bool(good.all())
                    values[band,i]=float(np.where(good,raw,0).sum())/(4*calibration[band])
                    seen[band,i]=True
    elif dataset=='icvl':
        import h5py
        with h5py.File(source['raw_file'],'r') as file:
            wave=np.asarray(file['bands']).ravel();indices=[int(np.flatnonzero(wave==w)[0]) for w in range(420,661,10)]
            if indices!=manifest['wavelength_mapping']['selected_indices']:raise ValueError('ICVL独立波长映射不符')
            if (height,width)!=(file['rad'].shape[2]//2,file['rad'].shape[1]//2):raise ValueError('ICVL实际空间轴/半尺寸不符')
            for i,(y,x) in enumerate(points):
                raw=np.asarray(file['rad'][indices,2*x:2*x+2,2*y:2*y+2],dtype=np.float64)
                good=np.isfinite(raw)&(raw>=0);valid[i]=bool(good.all())
                values[:,i]=np.where(good,raw,0).mean((1,2))
                seen[:,i]=True
    elif dataset=='kaist':
        import OpenEXR,Imath
        file=OpenEXR.InputFile(source['raw_file']);header=file.header();box=header['dataWindow'];original_width=box.max.x-box.min.x+1
        if (height,width)!=((box.max.y-box.min.y+1)//2,original_width//2):raise ValueError('KAIST实际半尺寸不符')
        try:
            names=['w'+str(w)+'nm' for w in range(420,661,10)]
            if names!=manifest['wavelength_mapping']['channel_names']:raise ValueError('KAIST独立名称映射不符')
            for j,name in enumerate(names):
                for i,(y,x) in enumerate(points):
                    raw=np.frombuffer(file.channel(name,Imath.PixelType(Imath.PixelType.FLOAT),box.min.y+2*y,box.min.y+2*y+1),dtype=np.float32).reshape(2,original_width)[:,2*x:2*x+2]
                    good=np.isfinite(raw)&(raw>=0);valid[i]&=bool(good.all());values[j,i]=np.where(good,raw,0).mean(dtype=np.float64)
                    seen[j,i]=True
        finally:file.close()
    else:raise ValueError('未知数据集')
    if not seen.all():raise ValueError('独立基础图核对未覆盖全部指定谱带/像元')
    values[:,~valid]=0;expected=values.astype(np.float32)
    actual=np.stack([data[:,y,x] for y,x in points],axis=1)
    actual_mask=np.asarray([mask[y,x] for y,x in points])
    scale=max(float(np.max(np.abs(expected))),1e-30)
    error=float(np.max(np.abs(actual-expected))/scale)
    if error>2e-6 or not np.array_equal(actual_mask,valid):raise ValueError('实际基础图面积平均/校正/掩码独立核对失败：'+source['scene_id'])
    return {'scene_id':source['scene_id'],'split':manifest['split'],'fixed_points_yx':points,'relative_max_error':error,'mask_exact':True}


def independent_header(row):
    """所有候选只读取格式/通道头，包括测试；不读取其图像像元。"""
    metadata=row['metadata'];dataset=row['dataset']
    if dataset=='harvard':
        variables={n:s for n,s,k in whosmat(row['raw_file'],appendmat=False)}
        calibration=np.loadtxt(row['calibration_file']).reshape(-1)
        if list(variables['ref'])!=metadata['shape_hwc'] or variables['lbl']!=variables['ref'][:2]:raise ValueError('Harvard实际变量维度与清单不符')
        if variables['ref'][2]!=31 or not (np.isfinite(calibration).all() and (calibration>0).all()) or len(calibration)!=31:raise ValueError('Harvard校正向量不符')
        if metadata['wavelengths_nm']!=list(range(420,721,10)) or metadata['selected_indices']!=list(range(25)) or metadata['calibration']!='divide_sensitivity_no_photon_conversion':raise ValueError('Harvard波长/能量校正语义不符')
    elif dataset=='icvl':
        import h5py
        with h5py.File(row['raw_file'],'r') as file:
            wave=np.asarray(file['bands']).ravel();shape=file['rad'].shape
            indices=[int(np.flatnonzero(wave==w)[0]) for w in range(420,661,10)]
            if not np.array_equal(wave,np.arange(400,701,10)) or indices!=metadata['selected_indices'] or list(wave)!=metadata['wavelengths_nm']:raise ValueError('ICVL实际bands与清单不符')
            if [shape[2],shape[1],shape[0]]!=metadata['shape_hwc']:raise ValueError('ICVL实际轴与清单不符')
        if row['upstream_sha_unavailable'] or row['source_revision']!='d2cf6714224029431cf4cec551ca6753ed59bc52':raise ValueError('ICVL缺固定官方版本摘要')
    elif dataset=='kaist':
        import OpenEXR
        file=OpenEXR.InputFile(row['raw_file'])
        try:
            header=file.header();box=header['dataWindow'];names=['w'+str(w)+'nm' for w in range(420,661,10)]
            if not all(name in header['channels'] for name in names) or names!=metadata['channel_names']:raise ValueError('KAIST真实谱通道名不符')
            if [box.max.y-box.min.y+1,box.max.x-box.min.x+1,25]!=metadata['shape_hwc']:raise ValueError('KAIST真实数据窗口不符')
        finally:file.close()
    else:raise ValueError('未知候选数据集')


def audit_run(run):
    run=Path(run).resolve();report=read_json(run/'audit.json')
    if report['status']!='passed':raise ValueError('正式M3未通过，不能写独立验收通过')
    config=report['effective_config']['config'];grouped=read_json(run/'group_audit.json')['source_records']
    scenes=read_json(run/'scene_manifest.json');by_id={r['scene_id']:r for r in grouped}
    counts=dict(collections.Counter(r['dataset'] for r in grouped))
    if counts!={'harvard':77,'icvl':202,'kaist':30}:raise ValueError('全部实际候选来源不符')
    # 每个源文件只重算一次SHA；所有候选均参与，含封存测试仅完整性核对。
    source_files={file:digest for row in grouped for file,digest in row['source_sha256'].items()}
    for file,digest in source_files.items():
        if sha256(file)!=digest:raise ValueError('候选原始文件SHA变化：'+file)
    for row in grouped:independent_header(row)
    groups=collections.defaultdict(list)
    for row in grouped:groups[row['scene_group_id']].append(row)
    for group,rows in groups.items():
        expected='group:'+hashlib.sha256('\n'.join(sorted(r['scene_id'] for r in rows)).encode()).hexdigest()
        if expected!=group:raise ValueError('组ID不是稳定成员哈希')
    reader_sha=scenes[0]['preprocessing_version']['reader_sha256'];descriptors={}
    for row in grouped:
        folder=safe_path(config['data_dir'])/'metadata'/fingerprint({'raw':row['source_sha256'],'reader':reader_sha})
        record=folder/'record.json'
        if not record.exists():
            candidates=sorted(folder.rglob('record.json'))
            if not candidates:raise ValueError('实际重复描述元数据缓存缺失')
            record=candidates[-1]
        info=read_json(record)
        if sha256(record.parent/'descriptor.npy')!=info['descriptor_sha256']:raise ValueError('重复元数据SHA变化')
        descriptors[row['scene_id']]=np.load(record.parent/'descriptor.npy')
    relationships=0;near_duplicate_pairs=0
    for i,a in enumerate(grouped):
        for b in grouped[:i]:
            related=a['raw_sha256']==b['raw_sha256']
            if a['dataset']==b['dataset']=='icvl':
                # 独立实现采集键，不调用生产分组函数。
                prefix_a='Master_kelvin_lighting_variants' if re.fullmatch(r'Master\d+k(?:_\d+k)*',a['stem'],re.I) else re.sub(r'-\d{4}(?:-\d+)?$','',a['stem'])
                prefix_b='Master_kelvin_lighting_variants' if re.fullmatch(r'Master\d+k(?:_\d+k)*',b['stem'],re.I) else re.sub(r'-\d{4}(?:-\d+)?$','',b['stem'])
                related|=prefix_a==prefix_b
            if float(descriptors[a['scene_id']]@descriptors[b['scene_id']])>=.9999:
                related=True;near_duplicate_pairs+=1
            if related:
                relationships+=1
                if a['scene_group_id']!=b['scene_group_id']:raise ValueError('重复SHA/同捕获前缀跨关联组')
    membership=collections.defaultdict(set)
    for row in scenes:
        membership[row['scene_group_id']].add(row['split'])
        if row['scene_group_id']!=by_id[row['scene_id']]['scene_group_id']:raise ValueError('交接组ID不符')
        if by_id[row['scene_id']].get('excluded_reason') or by_id[row['scene_id']].get('legacy_diagnostic'):raise ValueError('历史诊断/排除组进入正式集合')
    if any(len(v)>1 for v in membership.values()):raise ValueError('实际场景组跨集')
    pixel_checks=[]
    for dataset in ['harvard','icvl','kaist']:
        for split in ['train','validation']:
            chosen=sorted([r for r in scenes if r['dataset']==dataset and r['split']==split],key=lambda r:r['scene_id'])[0]
            pixel_checks.append(independent_half_pixels(by_id[chosen['scene_id']],chosen))
    prepared=read_json(run/'prepared_scenes.json');norm=read_json(run/'normalization_audit.json');gain=read_json(run/'gain_audit.json')
    reader=BoundedReader(prepared,norm['scales']);selected=gain['selected_indices']
    index=[json.loads(line) for line in (run/'patch_index.jsonl').read_text(encoding='utf-8').splitlines()]
    index_mask_audit=independent_index_masks(prepared,index)
    independent_selection={}
    for row in index:
        digest=hashlib.sha256((str(row['seed'])+':'+json.dumps(row,sort_keys=True)).encode()).hexdigest()
        scene=row['scene_id']
        if scene not in independent_selection or digest<independent_selection[scene][0]:independent_selection[scene]=(digest,row)
    if {scene:r[1] for scene,r in independent_selection.items()}!=selected:raise ValueError('共同增益不是预登记的逐训练场景哈希最小索引')
    physical_checks=[];augmentation_checks=[]
    parent=safe_path(config['m2_run']);package=read_json(parent/'jeon_package.json')
    coeff=np.load(parent/package['spectral_response'])*np.asarray(package['spectral_weights']['values'])[None,:]
    training={r['scene_id']:r for r in prepared if r['split']=='train'}
    for dataset in ['harvard','icvl','kaist']:
        scene=sorted(r['scene_id'] for r in prepared if r['dataset']==dataset and r['split']=='train')[0];base=training[scene]
        data=np.load(base['data_file'],mmap_mode='r');mask=np.load(base['mask_file'],mmap_mode='r')
        for scale in [.5,1.,2.]:
            if min(int(data.shape[1]*scale),int(data.shape[2]*scale))<256:continue
            image,valid=reader.read(scene,scale,0,0)
            for y,x in [(0,0),(127,129),(255,255)]:
                value,good=independent_augmented_pixel(data,mask,scale,y,x,norm['scales'][dataset])
                if not np.allclose(value,image[:,y,x],rtol=2e-6,atol=1e-8) or good!=bool(valid[y,x]):raise ValueError('真实训练在线增强/掩码独立核对失败')
            augmentation_checks.append({'scene_id':scene,'scale':scale,'mask_checked':True})
        row=selected[scene];image,valid=reader.read(scene,row['scale'],row['y'],row['x']);points=[(0,0),(127,129),(255,255)]
        for method in ['jeon','fresnel']:
            package=read_json(parent/(method+'_package.json'));kernels=np.load(parent/package['kernels_training'])
            expected=direct_rgb_points(image,kernels,coeff,points)
            blurred=np.stack([fftconvolve(np.asarray(b,dtype=np.float64),k,mode='same') for b,k in zip(image,kernels)])
            rgb=np.einsum('cl,lhw->chw',coeff,blurred)
            actual=np.stack([rgb[:,y,x] for y,x in points],axis=1)
            error=float(np.max(np.abs(expected-actual))/max(float(np.max(np.abs(expected))),1e-30))
            if error>1e-10:raise ValueError('真实训练共同增益物理RGB的FFT/直接核求和不一致')
            physical_checks.append({'scene_id':scene,'method':method,'relative_max_error':error})
    return {'milestone':'M3','status':'passed','scope':'全部候选SHA及关联组、六幅训练/验证基础图独立像元、真实训练增强、共同增益哈希选择及两器件直接核求和核对',
            'formal_run':str(run),'candidate_counts':counts,'source_files_sha_checked':len(source_files),'capture_or_sha_relationships_checked':relationships,
            'actual_headers_wavelengths_checked':len(grouped),'index_mask_audit':index_mask_audit,
            'near_duplicate_pairs_checked':near_duplicate_pairs,'groups_disjoint':True,'legacy_groups_excluded':True,'half_pixel_checks':pixel_checks,'augmentation_checks':augmentation_checks,'physical_gain_forward_checks':physical_checks,
            'sealed_test_pixels_read':False,'test_predictions_generated':False,'cache_measurement':reader.metrics(),'paper_alignment_passed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    result=audit_run(args.run)
    destination=ROOT/'outputs/jeon2019_optics/results/reproduction_v1/m3'/('review_'+uuid.uuid4().hex[:12]);destination.mkdir()
    save_json(destination/'independent_completion_audit.json',result)
    print(json.dumps({'status':result['status'],'report':str(destination/'independent_completion_audit.json')},ensure_ascii=False))
