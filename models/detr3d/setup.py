"""
Setup file in order to link the different modules of the project.
"""

import glob
import os.path as osp

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
# Assign a name to the extension
_ext_src_root = "_ext_src"
# List of source files
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob("{}/src/*.cu".format(_ext_src_root))
# List of header files
_ext_headers = glob.glob("{}/include/*.h".format(_ext_src_root))

setup(
    name="detr3d_pointnet2_ext",
    ext_modules=[
        CUDAExtension(
            name="_ext_src",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
