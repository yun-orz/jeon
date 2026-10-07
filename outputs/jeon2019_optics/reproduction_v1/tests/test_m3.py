"""M3合成fixture与独立标量核对；fixture长期保留，不清理文件，不访问真实封存测试。"""
import hashlib
import json
import sys
import unittest
import uuid
from pathlib import Path
import numpy as np
from scipy.io import savemat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import ROOT
from m3_common import fingerprint,rank
from m3_readers import Harvard,ICVL,KAIST,inspect_scene,preprocess,wavelength_indices,exr_wavelength_names
from m3_index import scene_groups,split_groups,normalization_scale,BoundedReader,build_index,fit_gain

FIXTURE=ROOT/'work/datasets/reproduction_v1/fixtures'/('run_'+uuid.uuid4().hex)
FIXTURE.mkdir(parents=True)
(FIXTURE/'kind.json').write_text(json.dumps({'kind':'synthetic_fixture_only','formal_scenes':0}),encoding='utf-8')


class TestM3(unittest.TestCase):
    def test_sparse_masks_keep_scene_coverage(self):
        folder=FIXTURE/'sparse_mask';folder.mkdir()
        data=np.ones((25,256,256),dtype=np.float32)
        mask=np.zeros((256,256),dtype=bool);mask[100:110,100:110]=True
        data[:,~mask]=0
        np.save(folder/'base.npy',data);np.save(folder/'mask.npy',mask)
        records=[{'scene_id':'harvard:sparse'+str(i),'scene_group_id':'sparse'+str(i),'dataset':'harvard','split':'train','data_file':str(folder/'base.npy'),'mask_file':str(folder/'mask.npy')} for i in range(2)]
        rows=build_index(records,{'harvard':1.},20261006,2)
        self.assertEqual({r['scene_id'] for r in rows},{r['scene_id'] for r in records})
        reader=BoundedReader(records,{'harvard':1.})
        for row in rows:
            _,valid=reader.read(row['scene_id'],row['scale'],row['y'],row['x'])
            self.assertTrue(valid.any());self.assertFalse(valid.all())
        self.assertEqual(fingerprint(rows),fingerprint(build_index(records,{'harvard':1.},20261006,2)))

    def test_lighting_variants_are_one_capture_group(self):
        names=['Master2900k','Master5000K','Master5000K_2900K','Master20150112_f2_colorchecker']
        rows=[{'dataset':'icvl','scene_id':'icvl:'+name,'stem':name,'raw_sha256':str(i)} for i,name in enumerate(names)]
        descriptors={row['scene_id']:np.eye(4,dtype=np.float32)[i] for i,row in enumerate(rows)}
        grouped,_=scene_groups(rows,descriptors)
        self.assertEqual(len({row['scene_group_id'] for row in grouped[:3]}),1)
        self.assertNotEqual(grouped[0]['scene_group_id'],grouped[3]['scene_group_id'])

    def test_harvard_stream_calibration_and_area(self):
        p=FIXTURE/'harvard';p.mkdir()
        values=np.arange(64*66*31,dtype=float).reshape(64,66,31)%100+1
        labels=np.ones((64,66));labels[2,4]=0
        savemat(p/'scene.mat',{'ref':values,'lbl':labels},do_compression=True)
        calibration=np.arange(1,32,dtype=float);np.savetxt(p/'calib.txt',calibration)
        (p/'README.txt').write_text('ref 420:10:720 lbl sensitivity',encoding='utf-8')
        reader=Harvard(p/'scene.mat',p/'calib.txt',p/'README.txt')
        descriptor,meta=inspect_scene(reader)
        self.assertEqual(descriptor.shape,(25*24*24,));self.assertEqual(meta['selected_indices'],list(range(25)))
        result=preprocess(reader,p/'base.npy',p/'mask.npy')
        actual=np.load(p/'base.npy');mask=np.load(p/'mask.npy')
        for band in range(25):
            expected=(values[:,:,band]/calibration[band]).reshape(32,2,33,2).mean((1,3))
            expected[1,2]=0
            np.testing.assert_allclose(actual[band],expected,rtol=1e-7)
        self.assertFalse(mask[1,2]);self.assertEqual(int(mask.sum()),32*33-1)
        self.assertLess(result['peak_source_block_bytes'],512*1024**2)

    def test_icvl_actual_bands_and_axis(self):
        import h5py
        p=FIXTURE/'icvl.mat';values=np.arange(32*34*31,dtype=np.float32).reshape(32,34,31)+1
        with h5py.File(p,'w') as f:
            f.create_dataset('rad',data=values.transpose(2,1,0));f.create_dataset('bands',data=np.arange(400,701,10)[:,None])
        r=ICVL(p);out=FIXTURE/'icvl_base.npy';mask=FIXTURE/'icvl_mask.npy';preprocess(r,out,mask);r.close()
        expected=values[:,:,2:27].transpose(2,0,1).reshape(25,16,2,17,2).mean((2,4))
        np.testing.assert_allclose(np.load(out),expected)
        with self.assertRaises(ValueError):wavelength_indices(np.arange(430,661,10))

    def test_exr_channel_names_and_scanlines(self):
        import OpenEXR,Imath
        p=FIXTURE/'kaist.exr';h,w=32,34
        header=OpenEXR.Header(w,h);pt=Imath.PixelType(Imath.PixelType.FLOAT)
        names=['w%dnm'%v for v in reversed(range(420,721,10))]
        header['channels']={name:Imath.Channel(pt) for name in names}
        file=OpenEXR.OutputFile(str(p),header)
        file.writePixels({name:np.full((h,w),int(name[1:-2])/1000,dtype=np.float32).tobytes() for name in names});file.close()
        r=KAIST(p);preprocess(r,FIXTURE/'exr_base.npy',FIXTURE/'exr_mask.npy');r.close()
        np.testing.assert_allclose(np.load(FIXTURE/'exr_base.npy')[:,0,0],np.arange(420,661,10)/1000,rtol=1e-7)
        with self.assertRaises(ValueError):exr_wavelength_names(['R','G','B']+[f'channel{i}' for i in range(25)])

    def test_groups_exposure_and_legacy_no_crossing(self):
        records=[{'dataset':'harvard','stem':str(i),'scene_id':f'harvard:{i}','raw_sha256':str(i),'legacy_diagnostic':i==0} for i in range(8)]
        a=np.array([1.,0.],dtype=np.float32);b=np.array([0.,1.],dtype=np.float32)
        grouped,reasons=scene_groups(records,{'harvard:0':a,'harvard:1':a,'harvard:2':b})
        self.assertEqual(grouped[0]['scene_group_id'],grouped[1]['scene_group_id'])
        self.assertIn('excluded_reason',grouped[1])
        targets={'harvard':{'total':6,'train':4,'validation':2,'test':0}}
        split,counts=split_groups(grouped,targets,20261006)
        self.assertEqual(len(split),6)
        self.assertFalse(any(r['scene_id'] in ['harvard:0','harvard:1'] for r in split))
        membership={}
        for r in split:membership.setdefault(r['scene_group_id'],set()).add(r['split'])
        self.assertTrue(all(len(v)==1 for v in membership.values()))

    def fixtures(self):
        if hasattr(self.__class__,'_records'):return self.__class__._records
        rows=[]
        for dataset,value in [('harvard',1.),('icvl',2.),('kaist',.5)]:
            path=FIXTURE/(dataset+'_training.npy');mask=FIXTURE/(dataset+'_training_mask.npy')
            data=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(25,512,512));data[:]=value;data.flush()
            np.save(mask,np.ones((512,512),dtype=bool))
            rows.append({'dataset':dataset,'scene_id':dataset+':synthetic_train','scene_group_id':dataset+':synthetic_group','split':'train','data_file':str(path),'mask_file':str(mask)})
        self.__class__._records=rows
        return rows

    def test_train_only_scales_and_sealed_test(self):
        train=self.fixtures();test=dict(train[0],scene_id='harvard:sealed_fixture',split='test',data_file='must_not_be_read')
        validation=dict(train[1],scene_id='icvl:validation_fixture',split='validation',data_file='must_not_be_fit')
        scales,meta=normalization_scale(train+[test,validation])
        self.assertEqual(scales,{'harvard':1.,'icvl':2.,'kaist':1.})
        reader=BoundedReader(train+[test,validation],scales)
        with self.assertRaises(ValueError):reader.read(test['scene_id'],1,0,0)

    def test_30000_index_reproduction_and_cache(self):
        records=self.fixtures();scales={'harvard':1.,'icvl':2.,'kaist':1.}
        a=build_index(records,scales,20261006,30000);b=build_index(records,scales,20261006,30000)
        self.assertEqual(fingerprint(a),fingerprint(b));self.assertEqual(len(a),30000)
        self.assertEqual(set(r['scale'] for r in a),{.5,1.,2.})
        reader=BoundedReader(records,scales,max_bytes=16*1024**2)
        for row in a[:20]:
            image,mask=reader.read(row['scene_id'],row['scale'],row['y'],row['x'])
            self.assertEqual(image.shape,(25,256,256));self.assertTrue(mask.all())
        metrics=reader.metrics();self.assertLessEqual(metrics['peak_cache_bytes'],16*1024**2)
        self.assertGreater(metrics['observed_process_rss_bytes'],0)
        tiny=BoundedReader(records,scales,1024)
        with self.assertRaises(ValueError):tiny.read(records[0]['scene_id'],1,0,0)
        save={'kind':'synthetic_fixture_only','index_count':30000,'same_seed_sha':fingerprint(a),'cache':metrics}
        (FIXTURE/'index_cache_audit.json').write_text(json.dumps(save,indent=2),encoding='utf-8')

    def test_common_gain_independent_scalar(self):
        records=self.fixtures();scales={'harvard':1.,'icvl':2.,'kaist':1.}
        index=build_index(records,scales,20261006,200)
        kernels={'jeon':np.ones((25,1,1))*.8,'fresnel':np.ones((25,1,1))*.6}
        response=np.ones((3,25));weights=np.full(25,10.)
        gain,meta=fit_gain(records,scales,index,kernels,response,weights)
        expected=.25/np.sqrt(np.mean([(value*k*250)**2 for value in [1,1,.5] for k in [.8,.6]]))
        self.assertAlmostEqual(gain,expected,places=14);self.assertTrue(meta['training_only'])

    def test_common_gain_spatial_kernel_boundary(self):
        records=self.fixtures();scales={'harvard':1.,'icvl':2.,'kaist':1.}
        index=build_index(records,scales,20261006,200)
        spatial=np.array([[.01,.03,.02],[.11,.42,.07],[.05,.17,.12]],dtype=np.float64)
        kernels={method:np.broadcast_to(spatial*factor,(25,3,3)).copy() for method,factor in [('jeon',.8),('fresnel',.6)]}
        gain,_=fit_gain(records,scales,index,kernels,np.ones((3,25)),np.full(25,10.))
        # 直接按九个位置累加零边界卷积，独立于生产代码的FFT路径。
        padded=np.pad(np.ones((256,256)),1)
        blurred=np.zeros((256,256),dtype=np.float64)
        for y in range(3):
            for x in range(3):blurred+=spatial[2-y,2-x]*padded[y:y+256,x:x+256]
        expected_rms=np.sqrt(np.mean([(value*factor*250)**2 for value in [1,1,.5] for factor in [.8,.6]])*np.mean(blurred**2))
        self.assertAlmostEqual(gain,.25/expected_rms,places=14)


if __name__=='__main__':unittest.main()
