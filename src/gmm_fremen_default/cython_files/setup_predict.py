from distutils.core import setup
#from Cython.Distutils import Extension
from Cython.Distutils import build_ext
from distutils.extension import Extension
#import cython_gsl
import numpy
from Cython.Build import cythonize
#python setup_predict.py build_ext -i

setup(
    name="cython predict",
    #include_dirs = [cython_gsl.get_include()],
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize([Extension("fremen_predict",
                                ["fremen_predict.pyx"],
                                extra_compile_args=['-march=native'],
                                include_dirs = [numpy.get_include()])#, '-fopenmp'],
                                #extra_link_args=['-fopenmp'])

                            ],
                                annotate=True
                            )
    )
