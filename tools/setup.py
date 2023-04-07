from setuptools import setup,Extension
import numpy
from Cython.Build import cythonize

extensions = Extension(
    "base_dtw",
    sources=["base_dtw.pyx"],
    include_dirs= [numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API",)],
)

setup(
    name="base_dtw",
    ext_modules=cythonize(extensions)
)