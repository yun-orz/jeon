"""外部独立审核算法的合成反例与直接卷积核对。"""
import unittest
import numpy as np
from scipy.signal import convolve2d
from audit_m3_result import direct_rgb_points,independent_augmented_pixel,independent_index_masks
import uuid
from audit import ROOT


class TestExternalAudit(unittest.TestCase):
    def test_all_index_masks_and_scene_coverage(self):
        folder=ROOT/'work/datasets/reproduction_v1/fixtures'/('mask_audit_'+uuid.uuid4().hex);folder.mkdir(parents=True)
        mask=np.ones((520,520),dtype=bool);mask[100:110,100:110]=False
        np.save(folder/'mask.npy',mask)
        scenes=[{'scene_id':'train','split':'train','mask_file':str(folder/'mask.npy')},{'scene_id':'test','split':'test','mask_file':'禁止读取测试mask'}]
        rows=[{'scene_id':'train','scale':s,'y':0,'x':0} for s in [.5,1.,2.]]
        result=independent_index_masks(scenes,rows)
        self.assertEqual(result['checked_indices'],3);self.assertEqual(result['partially_masked_indices'],3)
        # 十像元无效方块经双线性中心映射扩展为22×22的保守无效区域。
        self.assertEqual(result['minimum_valid_pixels'],256**2-22**2)
        with self.assertRaises(ValueError):independent_index_masks(scenes,[])

    def test_direct_kernel_indexing_and_zero_boundary(self):
        rng=np.random.default_rng(20261006)
        image=rng.normal(size=(25,9,11));kernels=rng.normal(size=(25,3,5));coeff=rng.normal(size=(3,25))
        points=[(0,0),(4,5),(8,10)]
        actual=direct_rgb_points(image,kernels,coeff,points)
        blurred=np.stack([convolve2d(b,k,mode='same') for b,k in zip(image,kernels)])
        rgb=np.einsum('cl,lhw->chw',coeff,blurred)
        expected=np.stack([rgb[:,y,x] for y,x in points],axis=1)
        np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)

    def test_scalar_area_and_center_mapping(self):
        data=np.arange(25*8*10,dtype=np.float32).reshape(25,8,10);mask=np.ones((8,10),dtype=bool)
        value,valid=independent_augmented_pixel(data,mask,.5,1,2,2.)
        expected=sum(data[:,y,x] for y in [2,3] for x in [4,5])/8
        np.testing.assert_array_equal(value,expected);self.assertTrue(valid)
        value,valid=independent_augmented_pixel(data,mask,2,3,5,2.)
        expected=(data[:,1,2]*9+data[:,1,3]*3+data[:,2,2]*3+data[:,2,3])/32
        np.testing.assert_array_equal(value,expected);self.assertTrue(valid)
        mask[1,2]=False
        value,valid=independent_augmented_pixel(data,mask,2,3,5,2.)
        self.assertFalse(valid);self.assertTrue((value==0).all())


if __name__=='__main__':unittest.main()
