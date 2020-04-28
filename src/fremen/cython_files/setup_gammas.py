from distutils.core import setup
#from Cython.Distutils import Extension
from Cython.Distutils import build_ext
from distutils.extension import Extension
#import cython_gsl
from Cython.Build import cythonize
import numpy
#python setup_gammas.py build_ext -i

setup(
    name="cython speed",
    #include_dirs = [cython_gsl.get_include()],
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize([Extension("gammas_for_fremen",
                                ["gammas_for_fremen.pyx"],
                                extra_compile_args=['-march=native', '-fopenmp'],
                                extra_link_args=['-fopenmp'],
                                include_dirs=[numpy.get_include()])
                            ],
                                annotate=True
                            )
    )
