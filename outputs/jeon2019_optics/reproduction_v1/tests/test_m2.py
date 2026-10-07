"""M2独立几何、边界方向、谱权重及增益状态核对；不清理任何文件。"""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from m2_operator import EnergyRGB, overlap_matrix, remap, mapping_audit, numerical_audit


class TestM2(unittest.TestCase):
    def test_area_geometry_and_mass(self):
        matrix, source, target, empty = overlap_matrix()
        self.assertEqual(matrix.shape,(49,97))
        self.assertAlmostEqual(float(source[0]),-48.5*6.22e-6)
        self.assertAlmostEqual(float(target[0]),-49*6.22e-6)
        self.assertAlmostEqual(float(empty.sum()),(98**2-97**2)*(6.22e-6)**2)
        rng = np.random.default_rng(3)
        kernels = rng.random((25,97,97)) / (97*97)
        self.assertTrue(mapping_audit(kernels,remap(kernels,matrix),matrix,6.22e-6)["passed"])

    def test_point_origin_and_asymmetric_convolution(self):
        k = np.zeros((25,3,3)); k[:,1,2]=.7
        response = np.ones((3,25)); weights=np.full(25,10.)
        op=EnergyRGB(k,response,weights)
        x=np.zeros((1,25,7,9)); x[0,0,3,4]=1
        result=op.numpy_forward(x)
        self.assertTrue(np.allclose(result[0,:,3,5],7.))
        self.assertEqual(np.count_nonzero(result),3)
        self.assertTrue(numerical_audit(k,response,weights)["passed"])

    def test_band_energy_and_density_equivalence(self):
        k=np.zeros((25,1,1));k[:,0,0]=.8
        response=np.arange(75).reshape(3,25)/75
        x=np.ones((1,25,3,4))
        density=EnergyRGB(k,response,np.full(25,10.)).numpy_forward(x)
        band=EnergyRGB(k,response,np.ones(25)).numpy_forward(10*x)
        np.testing.assert_allclose(density,band,rtol=1e-14)
        np.testing.assert_allclose(density[0,:,0,0],8*response.sum(1),rtol=1e-14)

    def test_pending_final_gain_rejected(self):
        k=np.ones((25,1,1)); r=np.ones((3,25));w=np.full(25,10.)
        with self.assertRaises(ValueError):EnergyRGB(k,r,w,final=True)
        x=np.ones((1,25,2,3))
        np.testing.assert_allclose(EnergyRGB(k,r,w,measurement_gain=2,final=True).numpy_forward(x),
                                   2*EnergyRGB(k,r,w).numpy_forward(x))

    def test_boundary_adjoint(self):
        rng=np.random.default_rng(88)
        k=rng.random((25,7,7))*.01
        self.assertTrue(numerical_audit(k,rng.random((3,25)),np.full(25,10.))["passed"])


if __name__=="__main__":unittest.main()
