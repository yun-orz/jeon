"""独立核对M3交付物；仅训练/验证基础图可读，测试保持封存。"""
from pathlib import Path
import json
import numpy as np
from audit import read_json, sha256
from m3_common import fingerprint


def validate_artifacts(dest, parents, operators):
    dest = Path(dest)
    scenes = read_json(dest/'scene_manifest.json')
    core = read_json(dest/'data_version.json')
    norm = read_json(dest/'normalization_audit.json')
    gain = read_json(dest/'gain_audit.json')
    seal = read_json(dest/'test_seal.json')
    checks = []
    def require(condition, message):
        if not condition:
            raise ValueError('独立审核失败：'+message)
        checks.append(message)
    require(core['fingerprint'] == fingerprint({k:v for k,v in core.items() if k!='fingerprint'}), '数据版本指纹')
    by_id = {r['scene_id']:r for r in scenes}
    require(len(by_id)==len(scenes), '场景ID唯一')
    groups = {}
    prepared_sha = {}
    for row in scenes:
        groups.setdefault(row['scene_group_id'], set()).add(row['split'])
        require(row['fingerprint']==fingerprint({k:v for k,v in row.items() if k!='fingerprint'}), '场景指纹：'+row['scene_id'])
        if row['split']=='test':
            require(row['base_resolution']['data_file'] is None and row['valid_mask']['file'] is None, '测试未预处理：'+row['scene_id'])
            continue
        path = Path(row['base_resolution']['data_file'])
        mask_path = Path(row['valid_mask']['file'])
        recorded = row['base_resolution']['sha256']
        require(sha256(path)==recorded['base.npy'] and sha256(mask_path)==recorded['mask.npy'], '基础图及掩码SHA：'+row['scene_id'])
        prepared_sha[row['scene_id']] = recorded
        data = np.load(path, mmap_mode='r')
        mask = np.load(mask_path, mmap_mode='r')
        require(data.dtype==np.float32 and data.shape[0]==25 and mask.dtype==bool and mask.shape==data.shape[1:], '基础图维度与类型：'+row['scene_id'])
        valid_count = 0
        for y in range(0, mask.shape[0], 32):
            block = data[:,y:y+32]
            valid = mask[y:y+32]
            require(bool(np.isfinite(block).all() and (block>=0).all()), '有限非负块：'+row['scene_id']+':'+str(y))
            require(bool((block[:,~valid]==0).all()), '无效像元置零且保留掩码：'+row['scene_id']+':'+str(y))
            valid_count += int(valid.sum())
        require(valid_count==row['base_resolution']['details']['valid_pixels'], '有效像元数：'+row['scene_id'])
    require(all(len(v)==1 for v in groups.values()), '组无跨集')
    require(core['preprocessed_sha256']==prepared_sha, '数据版本绑定基础图与掩码')
    training = {r['scene_id'] for r in scenes if r['split']=='train'}
    for dataset in ['harvard','icvl']:
        selected = [r for r in scenes if r['dataset']==dataset and r['split']=='train']
        expected = {r['scene_id'] for r in selected}
        info = norm['datasets'][dataset]
        require(set(info['training_scene_ids'])==expected and bool(expected), '尺度仅本库全部训练场景：'+dataset)
        total, count = 0., 0
        # 用不同块高重算RMS，避免仅信任生产路径中的汇总字段。
        for row in selected:
            data = np.load(row['base_resolution']['data_file'], mmap_mode='r')
            mask = np.load(row['valid_mask']['file'], mmap_mode='r')
            for y in range(0, mask.shape[0], 37):
                values = np.asarray(data[:,y:y+37], dtype=np.float64)[:,mask[y:y+37]]
                total += float(np.einsum('ij,ij->',values,values))
                count += values.size
        require(count==info['count'] and np.isclose(total,info['sum_squares'],rtol=1e-11), '独立重算训练平方和：'+dataset)
        require(np.isclose(np.sqrt(total/count),norm['scales'][dataset],rtol=1e-11), '独立重算训练尺度：'+dataset)
    require(norm['scales']['kaist']==1 and norm['datasets']['kaist']['fit'] is False, 'KAIST反射率未拟合尺度')
    rows = [json.loads(line) for line in (dest/'patch_index.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(rows)==30000 and sha256(dest/'patch_index.jsonl')==core['patch_index_sha256']==sha256(dest/'patch_index_replay.jsonl'), '三万索引与重放SHA')
    for i, row in enumerate(rows):
        require(row['index']==i and row['scene_id'] in training and row['scene_group_id']==by_id[row['scene_id']]['scene_group_id'], '索引训练归属：'+str(i))
        require(row['size']==256 and row['seed']==20261006 and row['scale'] in [.5,1.,2.], '索引参数：'+str(i))
        h,w = by_id[row['scene_id']]['base_resolution']['details']['shape_chw'][1:]
        require(0<=row['y']<=int(h*row['scale'])-256 and 0<=row['x']<=int(w*row['scale'])-256, '索引坐标：'+str(i))
    require(set(gain['selected_indices'])==training and gain['training_only'] is True, '共同增益覆盖所有训练场景')
    require(all(0<=r['index']<len(rows) and r==rows[r['index']] and r['scene_id']==scene for scene,r in gain['selected_indices'].items()), '共同增益样本属于冻结训练索引')
    require(gain['count']>0 and np.isclose(gain['physical_rms'],np.sqrt(gain['sum_squares']/gain['count']),rtol=1e-12), '增益RMS汇总')
    require(gain['value']>0 and np.isfinite(gain['value']) and np.isclose(gain['value']*gain['physical_rms'],.25,rtol=1e-12), '增益预登记目标')
    require(set(operators)==set(parents)=={'jeon','fresnel'}, '两个器件包齐全')
    for method, info in operators.items():
        parent = parents[method]
        folder = Path(info['file']).parent
        package = read_json(info['file'])
        require(package['fingerprint']==info['fingerprint']==fingerprint({k:v for k,v in package.items() if k!='fingerprint'}), '最终算子指纹：'+method)
        require(package['parent_fingerprint']==parent['fingerprint'] and package['data_fingerprint']==core['fingerprint'], '算子父来源及数据绑定：'+method)
        require(package['measurement_gain']['value']==gain['value'] and package['measurement_gain']['audit_sha256']==sha256(dest/'gain_audit.json'), '两器件同一增益与审核SHA：'+method)
        source = parent['sources']['response']
        files = {parent['kernels_native'],parent['kernels_training'],parent['spectral_response'],parent['mapping']['file'],source['raw_file'],source['license_file']}
        parent_dir = Path(package['sources']['parent_m2_run'])
        require(all(sha256(folder/name)==sha256(parent_dir/name) for name in files), '全部物理附件与M2逐字节一致：'+method)
    test_ids = {r['scene_id'] for r in scenes if r['split']=='test'}
    require(len(test_ids)==10 and test_ids==set(seal['scene_ids']) and not test_ids&training, '十幅测试与训练隔离')
    require(seal['source_sha256']=={r['scene_id']:r['source_sha256'] for r in scenes if r['split']=='test'}, '测试来源SHA绑定')
    require(set(seal['scene_groups'])=={r['scene_group_id'] for r in scenes if r['split']=='test'}, '测试场景组绑定')
    require(seal['operator_fingerprints']=={m:v['fingerprint'] for m,v in operators.items()}, '测试算子指纹绑定')
    require(seal['open_milestone']=='M7' and seal['preprocessed_before_M7'] is False and seal['data_version_fingerprint']==core['fingerprint'], '测试封条绑定及M7限制')
    require(seal['fingerprint']==fingerprint({k:v for k,v in seal.items() if k!='fingerprint'}), '测试封条指纹')
    cache = read_json(dest/'cache_audit.json')
    require(0<cache['peak_cache_bytes']<=cache['max_bytes']<=512*1024**2 and cache['observed_process_rss_bytes']>0, '真实缓存测量')
    # 保存计数和清单摘要，避免写出九万条重复的索引检查消息。
    return {'passed':True,'check_count':len(checks),'checks_sha256':fingerprint(checks),'training_scene_count':len(training),'test_scene_count':len(test_ids),'independent_rms_block_rows':37}
