"""M3场景关联组、确定性划分、按需增强和训练专用尺度/共同增益。"""
from collections import OrderedDict
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from m3_common import rank,fingerprint,save_json


def icvl_capture_key(stem):
    """明显的同采集照明变体保守同组，避免照明谱变化绕过近重复阈值。"""
    if re.fullmatch(r'Master\d+k(?:_\d+k)*',stem,re.IGNORECASE):return 'Master_kelvin_lighting_variants'
    return re.sub(r'-\d{4}(?:-\d+)?$','',stem)


def scene_groups(records,descriptors,cosine_min=.9999):
    """文件SHA、捕获前缀及整图曝光不变描述联合分组；跨库关联保守排除。"""
    n=len(records);parent=list(range(n))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b):parent[root(b)]=root(a)
    reasons=[]
    for i,a in enumerate(records):
        for j in range(i):
            b=records[j];reason=None
            if a['raw_sha256']==b['raw_sha256']:reason='原始内容SHA相同'
            elif a['dataset']==b['dataset']=='icvl' and icvl_capture_key(a['stem'])==icvl_capture_key(b['stem']):reason='ICVL同位置/日期捕获前缀或Master色温照明变体关联'
            elif a['scene_id'] in descriptors and b['scene_id'] in descriptors:
                similarity=float(np.dot(descriptors[a['scene_id']],descriptors[b['scene_id']]))
                if similarity>=cosine_min:reason=f'固定整图重复元数据曝光不变余弦{similarity:.9f}'
            if reason:join(i,j);reasons.append({'a':a['scene_id'],'b':b['scene_id'],'reason':reason})
    groups={}
    for i,record in enumerate(records):groups.setdefault(root(i),[]).append(record)
    for rows in groups.values():
        group='group:'+hashlib.sha256('\n'.join(sorted(r['scene_id'] for r in rows)).encode()).hexdigest()
        datasets={r['dataset'] for r in rows}
        for r in rows:
            r['scene_group_id']=group
            if len(datasets)>1:r['excluded_reason']='跨库关联组待人工核查，不纳入本版拆分'
            if any(x.get('legacy_diagnostic') for x in rows):r['excluded_reason']='历史已查看诊断场景及其关联组，明确排除正式训练/验证/测试'
    return records,reasons


def split_groups(records,targets,seed):
    """哈希排序的完整组分配；留出集合先固定，绝不从测试补足训练。"""
    result=[];counts={}
    for dataset,target in targets.items():
        groups={}
        for r in records:
            if r['dataset']==dataset and not r.get('excluded_reason'):groups.setdefault(r['scene_group_id'],[]).append(r)
        order=sorted(groups,key=lambda g:(rank(seed,g),g));selected=[];total=0
        for group in order:
            if total+len(groups[group])<=target['total']:
                selected.append(group);total+=len(groups[group])
        assigned={};remaining=list(selected)
        for split in ['test','validation','train']:
            count=0
            for group in list(remaining):
                if count+len(groups[group])<=target[split]:
                    assigned[group]=split;count+=len(groups[group]);remaining.remove(group)
            counts.setdefault(dataset,{})[split]=count
        for group,split in assigned.items():
            for r in groups[group]:result.append(dict(r,split=split))
        counts[dataset]['available_scenes']=sum(len(v) for v in groups.values())
        counts[dataset]['selected_scenes']=sum(counts[dataset][s] for s in ['train','validation','test'])
        counts[dataset]['target']=target
        counts[dataset]['scale_difference']=any(counts[dataset][s]!=target[s] for s in ['train','validation','test'])
    result.sort(key=lambda r:r['scene_id'])
    assignments={}
    for r in result:assignments.setdefault(r['scene_group_id'],set()).add(r['split'])
    if any(len(v)!=1 for v in assignments.values()):raise ValueError('场景组跨集')
    return result,counts


def normalization_scale(records,block_rows=64):
    """预登记：每库训练集所有有效像元、全部25谱带的共同RMS，分块FP64累加。"""
    scales={};details={}
    for dataset in ['harvard','icvl']:
        train=[r for r in records if r['dataset']==dataset and r['split']=='train']
        if not train:raise ValueError('缺少真实训练场景：'+dataset)
        total=0.;count=0
        for r in train:
            data=np.load(r['data_file'],mmap_mode='r');mask=np.load(r['mask_file'],mmap_mode='r')
            for y in range(0,data.shape[1],block_rows):
                valid=np.asarray(mask[y:y+block_rows]);values=np.asarray(data[:,y:y+block_rows],dtype=np.float64)[:,valid]
                total+=float(np.sum(values*values,dtype=np.float64));count+=values.size
        scale=float(np.sqrt(total/count)) if count else 0
        if not np.isfinite(scale) or scale<=0:raise ValueError('训练共同RMS尺度非法')
        scales[dataset]=scale;details[dataset]={'estimator':'有效训练像元全部25谱带的RMS','count':count,'sum_squares':total,'value':scale,'training_scene_ids':[r['scene_id'] for r in train]}
    scales['kaist']=1.;details['kaist']={'value':1.,'fit':False,'units':'反射率保留原值'}
    return scales,details


class BoundedReader:
    """仅内存映射基础图，缓存实际patch数组；全局及每场景实测不超过512MiB。"""
    def __init__(self,records,scales,max_bytes=64*1024**2):
        if not 0<max_bytes<=512*1024**2:raise ValueError('缓存上限不符合512MiB协议')
        self.records={r['scene_id']:r for r in records};self.scales=scales;self.max_bytes=max_bytes
        self.cache=OrderedDict();self.bytes=0;self.peak_bytes=0;self.per_scene_peak={};self.peak_block_bytes=0;self.peak_rss=0
    def read(self,scene_id,scale,y,x,size=256,allow_test=False):
        r=self.records[scene_id]
        if r['split']=='test' and not allow_test:raise ValueError('M7前测试封存：拒绝读取预测/拟合patch')
        if scale not in [.5,1.,2.]:raise ValueError('增强尺度不符')
        cost=25*size*size*4+size*size
        if cost>self.max_bytes:raise ValueError('单patch超过实际缓存上限，读取前拒绝')
        key=(scene_id,scale,y,x,size)
        if key in self.cache:
            value=self.cache.pop(key);self.cache[key]=value;return value
        source=np.load(r['data_file'],mmap_mode='r');mask=np.load(r['mask_file'],mmap_mode='r')
        h,w=source.shape[1:];height=int(h*scale);width=int(w*scale)
        if not (0<=y<=height-size and 0<=x<=width-size):raise ValueError('patch越界')
        if scale==1:
            image=np.asarray(source[:,y:y+size,x:x+size]).copy();valid=np.asarray(mask[y:y+size,x:x+size]).copy()
        elif scale==.5:
            raw=np.asarray(source[:,2*y:2*(y+size),2*x:2*(x+size)])
            image=raw.reshape(25,size,2,size,2).mean((2,4),dtype=np.float64).astype(np.float32)
            valid=np.asarray(mask[2*y:2*(y+size),2*x:2*(x+size)]).reshape(size,2,size,2).all((1,3))
            self.peak_block_bytes=max(self.peak_block_bytes,raw.nbytes)
        else:
            yy=(np.arange(y,y+size)+.5)/2-.5;xx=(np.arange(x,x+size)+.5)/2-.5
            ya=max(0,int(np.floor(yy.min())));yb=min(h,int(np.ceil(yy.max()))+1)
            xa=max(0,int(np.floor(xx.min())));xb=min(w,int(np.ceil(xx.max()))+1)
            gy,gx=np.meshgrid(yy-ya,xx-xa,indexing='ij');points=np.stack([gy,gx])
            raw=np.asarray(source[:,ya:yb,xa:xb]);image=np.stack([map_coordinates(b,points,order=1,mode='nearest') for b in raw])
            valid=map_coordinates(np.asarray(mask[ya:yb,xa:xb],dtype=np.float32),points,order=1,mode='nearest')>=1-1e-6
            self.peak_block_bytes=max(self.peak_block_bytes,raw.nbytes)
        image/=self.scales[r['dataset']]
        image[:,~valid]=0
        while self.cache and self.bytes+cost>self.max_bytes:
            _,value=self.cache.popitem(last=False);self.bytes-=value[0].nbytes+value[1].nbytes
        self.cache[key]=(image,valid);self.bytes+=image.nbytes+valid.nbytes;self.peak_bytes=max(self.peak_bytes,self.bytes)
        scene_bytes=sum(a.nbytes+b.nbytes for k,(a,b) in self.cache.items() if k[0]==scene_id)
        self.per_scene_peak[scene_id]=max(self.per_scene_peak.get(scene_id,0),scene_bytes)
        try:
            import psutil
            self.peak_rss=max(self.peak_rss,psutil.Process().memory_info().rss)
        except ImportError:raise ValueError('缺少实际RSS测量依赖，不能宣称缓存实测完成')
        return image,valid
    def metrics(self):return {'peak_cache_bytes':self.peak_bytes,'per_scene_peak_bytes':self.per_scene_peak,'max_bytes':self.max_bytes,'peak_source_block_bytes':self.peak_block_bytes,'observed_process_rss_bytes':self.peak_rss,'whole_dataset_loaded':False}


def build_index(records,scales,seed,count=30000):
    """仅训练场景生成轻量索引，真实掩码审核每个位置，不复制完整patch。"""
    train=sorted([r for r in records if r['split']=='train'],key=lambda r:r['scene_id'])
    if not train:raise ValueError('没有训练场景')
    rng=np.random.default_rng(seed);reader=BoundedReader(train,scales);rows=[]
    options=[]
    for r in train:
        _,h,w=np.load(r['data_file'],mmap_mode='r').shape
        all_valid=bool(np.load(r['mask_file'],mmap_mode='r').all())
        for scale in [.5,1.,2.]:
            if int(h*scale)>=256 and int(w*scale)>=256:options.append((r,scale,int(h*scale),int(w*scale),all_valid))
    if not options:raise ValueError('没有符合256×256的实际增强图')
    if count<len(train):raise ValueError('索引数量不足以覆盖全部训练场景')
    by_scene={r['scene_id']:[o for o in options if o[0]['scene_id']==r['scene_id']] for r in train}
    if any(not value for value in by_scene.values()):raise ValueError('训练场景没有符合尺寸的增强图')
    for i in range(count):
        # 先为每个训练场景登记一条，其余随机；掩码随补丁保留，不将无效像元宣称为有效。
        eligible=by_scene[train[i]['scene_id']] if i<len(train) else options
        for attempt in range(1000):
            r,scale,h,w,all_valid=eligible[int(rng.integers(len(eligible)))];y=int(rng.integers(h-255));x=int(rng.integers(w-255))
            if all_valid:break
            # 索引只读mask，避免为30000个索引执行完整光谱增强。
            mask=np.load(r['mask_file'],mmap_mode='r')
            if scale==1:valid=np.asarray(mask[y:y+256,x:x+256])
            elif scale==.5:valid=np.asarray(mask[2*y:2*(y+256),2*x:2*(x+256)]).reshape(256,2,256,2).all((1,3))
            else:
                yy=(np.arange(y,y+256)+.5)/2-.5;xx=(np.arange(x,x+256)+.5)/2-.5
                ya=max(0,int(np.floor(yy.min())));yb=min(mask.shape[0],int(np.ceil(yy.max()))+1)
                xa=max(0,int(np.floor(xx.min())));xb=min(mask.shape[1],int(np.ceil(xx.max()))+1)
                gy,gx=np.meshgrid(yy-ya,xx-xa,indexing='ij')
                valid=map_coordinates(np.asarray(mask[ya:yb,xa:xb],dtype=np.float32),np.stack([gy,gx]),order=1,mode='nearest')>=1-1e-6
            if valid.any():break
        else:raise ValueError('找不到含有效像元的256×256位置，禁止伪造掩码')
        rows.append({'index':i,'scene_id':r['scene_id'],'scene_group_id':r['scene_group_id'],'scale':scale,'y':y,'x':x,'size':256,'valid_mask':'保留真实逐像元mask；至少一个有效像元；拟合仅统计有效像元','seed':seed,'preprocessing_version':'m3-energy-area-v1'})
    return rows


def fit_gain(records,scales,index,kernels,response,weights,target_rms=.25):
    """预登记：每训练场景哈希选择一条索引，两器件RGB等权RMS映射至0.25。"""
    reader=BoundedReader(records,scales);selected={}
    for row in index:
        scene=row['scene_id']
        if scene not in selected or rank(row['seed'],json.dumps(row,sort_keys=True))<rank(selected[scene]['seed'],json.dumps(selected[scene],sort_keys=True)):selected[scene]=row
    expected={r['scene_id'] for r in records if r['split']=='train'}
    if set(selected)!=expected:raise ValueError('共同增益索引未覆盖所有训练场景')
    coeff=response*np.asarray(weights)[None,:];square=0.;count=0
    for scene,row in sorted(selected.items()):
        image,mask=reader.read(scene,row['scale'],row['y'],row['x'])
        for method in ['jeon','fresnel']:
            blurred=np.stack([fftconvolve(b.astype(np.float64),k,mode='same') for b,k in zip(image,kernels[method])])
            rgb=np.einsum('cl,lhw->chw',coeff,blurred)
            values=rgb[:,mask];square+=float(np.sum(values*values));count+=values.size
    rms=float(np.sqrt(square/count)) if count else 0
    if not np.isfinite(rms) or rms<=0:raise ValueError('共同训练测量RMS非法')
    gain=target_rms/rms
    return gain,{'estimator':'每训练场景索引哈希最小的一条256×256样本，两器件/三通道/有效像元等权RMS','target_rms':target_rms,'physical_rms':rms,'sum_squares':square,'count':count,'value':gain,'training_only':True,'same_gain_both_devices':True,'selected_indices':selected,'cache_measurement':reader.metrics()}
