from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

os.makedirs("monotonic_align", exist_ok=True)
open(os.path.join("monotonic_align", "__init__.py"), "a").close()

setup(
  name = 'monotonic_align',
  ext_modules = cythonize("core.pyx"),
  include_dirs=[numpy.get_include()]
)
