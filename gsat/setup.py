from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gsat',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='ops.voxel_op',
            sources=[
                'ops/voxelization/voxelization.cpp',
                'ops/voxelization/voxelization_cpu.cpp',
                'ops/voxelization/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)
