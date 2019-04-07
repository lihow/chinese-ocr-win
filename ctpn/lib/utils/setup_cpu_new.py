import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os

PATH = os.environ.get('PATH')+ ';C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\\Build'
numpy_include = np.get_include()
setup(ext_modules=cythonize("bbox.pyx"),include_dirs=[numpy_include])
setup(ext_modules=cythonize("cython_nms.pyx"),include_dirs=[numpy_include])