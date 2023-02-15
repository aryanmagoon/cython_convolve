import numpy as np
cimport numpy as np

cpdef convolve(np.ndarray x, np.ndarray h):
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t M = h.shape[0]
    cdef Py_ssize_t L = N + M - 1
    cdef np.ndarray y = np.zeros(L, dtype=np.result_type(x, h))
    cdef Py_ssize_t n, kmin, kmax, pad_length
    cdef np.ndarray x_part, h_part

    for n in range(L):
        kmin = max(0, n - (M - 1))
        kmax = min(N - 1, n)
        x_part = <np.ndarray> np.PyArray_DATA(x) + x.strides[0] * kmin
        h_part = <np.ndarray> np.PyArray_DATA(h) + h.strides[0] * (n - kmax)
        pad_length = max(0, len(x_part) - len(h_part))
        h_part = np.pad(h_part, (0, pad_length), 'constant')
        y[n] = np.sum(x_part[:len(h_part)] * h_part)

    return y