import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport cos
from libc.math cimport sin
from libc.math cimport sqrt
from cython.parallel import prange
cimport openmp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline void mean_gamma(double [:,:] out, const double [:] S, const double [:] T, const double [:] W, const long lenT, const long lenW, const double DlenT, const double meanS) nogil:
    cdef long i
    cdef long j
    cdef double gamma0
    cdef double gamma1

    for i in xrange(lenW):
        gamma0 = 0
        gamma1 = 0
        for j in prange(lenT, nogil=True, schedule='static'):
            gamma0 += (S[j] - meanS)*cos(W[i] * T[j])
            gamma1 -= (S[j] - meanS)*sin(W[i] * T[j])
        out[i, 0] = gamma0 / DlenT
        out[i, 1] = gamma1 / DlenT
                


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate(double [:] S, double [:] T, double [:] W, double PI2, double meanS):
    cdef long lenT = len(T)
    cdef long lenW = len(W)
    cdef double DlenT = lenT
    #cdef double meanS = np.mean(S)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.empty((lenW,2), dtype=np.float64)
    mean_gamma(out, S, T, W, lenT, lenW, DlenT, meanS)
    return out

