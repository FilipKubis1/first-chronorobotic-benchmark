import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport cos
#from libc.math cimport sin
#from libc.math cimport sqrt
#from cython.parallel import prange
#cimport openmp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False) 
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline void predict(double [:] out, const double [:] alphas, const double [:] omegas, const double [:] phis, const double gamma_0, const double [:] times, const long lenG, const long lenT) nogil:
    cdef long i
    cdef long j
    cdef double tmp

    for i in xrange(lenT):
        tmp = gamma_0
        for j in xrange(lenG):
            tmp += alphas[j] * cos(omegas[j] * times[i] + phis[j])
        out[i] = tmp









@cython.boundscheck(False)
@cython.wraparound(False)
def calculate(double [:] alphas, double [:] omegas, double [:] times, double [:] phis, double gamma_0):
    cdef long lenG = len(alphas)
    cdef long lenT = len(times)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(lenT, dtype=np.float64)
    predict(out, alphas, omegas, phis, gamma_0, times, lenG, lenT)
    return out

