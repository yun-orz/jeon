Hyperspectral Image Database of Real-world Scenes
- 50 Images in daylight illumination

http://vision.seas.harvard.edu/hyperspec/

This database contains a MATLAB .mat file for each image, with two
variables:

ref: NxMx31 - Image data for the 31 channels from 420:10:720 nm
lbl: NxM    - Mask data, a value of 0 implies that the pixel is masked
              out.

There is a calib.txt file which contains a 31x1 vector, corresponding
to sensitivity of the camera.

This database is being released for unrestricted research and academic
use. If you use this data in any publication, we ask that you cite the
following paper:

Ayan Chakrabarti and Todd Zickler, "Statistics of Real-World
Hyperspectral Images," in Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2011.

@conference{chakrabarti2011statistics,
  title={{Statistics of Real-World Hyperspectral Images}},
  author={Chakrabarti, A. and Zickler, T.},
  booktitle={Proc.~IEEE Conf.~on Computer Vision and Pattern Recognition (CVPR)},
  pages={193--200},
  year={2011}
}

For any questions, please contact:
   Ayan Chakrabarti at ayanc@eecs.harvard.edu
