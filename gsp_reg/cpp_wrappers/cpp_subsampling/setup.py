from setuptools import setup, Extension
import numpy as np

SOURCES = ["../cpp_utils/cloud/cloud.cpp",
             "grid_subsampling/grid_subsampling.cpp",
             "wrapper.cpp"]

module = Extension(name="grid_subsampling",
                    sources=SOURCES,
                    extra_compile_args=['-std=c++11',
                                        '-D_GLIBCXX_USE_CXX11_ABI=0'])


setup(ext_modules=[module], include_dirs=[np.get_include()])








